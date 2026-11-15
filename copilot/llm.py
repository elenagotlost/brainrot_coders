"""Ask the small language model for one letter, A to E, or for its probabilities."""
import math
import os
import sys

from .config import CONFIG
from .client import post

URL = f"http://127.0.0.1:{CONFIG['LLM_PORT']}/v1/chat/completions"

# Keep the system prompt stable: llama-server caches the start of the prompt,
# so text that never changes is only read once.
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Read the problem and the options, then reply with the letter of the correct option.")


def format_problem(problem, sees_image=False):
    lines = [problem["text"].strip()]
    options = problem.get("options") or {}
    for letter in "ABCDE":
        if letter in options:
            lines.append(f"{letter}) {options[letter]}")
    if not options:
        lines.append("(The options are in the picture.)" if sees_image else
                     "(The options are in a picture that you can't see. Pick the most likely letter.)")
    return "\n".join(lines)


def build_messages(problem, examples=(), image_b64=None):
    """image_b64 is a PNG as base64 text. Only a server started with --mmproj can see it."""
    messages = [{"role": "system", "content": SYSTEM}]
    for ex in examples:  # solved problems retrieved with RAG, if any
        messages.append({"role": "user", "content": format_problem(ex) + "\nAnswer:"})
        messages.append({"role": "assistant", "content": ex["label"]})
    text = format_problem(problem, sees_image=bool(image_b64)) + "\nAnswer:"
    if image_b64:
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + image_b64}},
                   {"type": "text", "text": text}]
    else:
        content = text
    messages.append({"role": "user", "content": content})
    return messages


def answer_letter(problem, examples=(), timeout=60, image_b64=None, url=URL):
    """Return (letter, timings). The grammar makes any other output impossible."""
    response = post(url, {
        "messages": build_messages(problem, examples, image_b64),
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=timeout)
    letter = response["choices"][0]["message"]["content"].strip()
    return letter, response.get("timings", {})


LETTERS = "ABCDE"
# Read deep: without a grammar the base model often wants to start a sentence, so a
# letter can sit far down the list. experiments/dump_probs.py reads the same depth, and the
# calibration is fitted on its output, so keep the two the same.
TOP_LOGPROBS = 5000
_debug_printed = False


def probs_messages(problem, prefix=None):
    """The messages answer_probs sends. prefix is fixed text added to the system
    prompt (worked examples, for instance): it must be the same for every problem,
    so the server reads it only once."""
    messages = build_messages(problem)
    if prefix:
        messages[0] = {"role": "system", "content": SYSTEM + "\n\n" + prefix}
    return messages


def letter_probs(top_logprobs):
    """Map the top tokens of the first position onto A-E and renormalize.

    Returns (probabilities or None, letter_mass). The same rule as
    experiments/dump_probs.py: "A", " A", "A)", "(A" and "**A" count as A, and their
    probabilities add up; a lowercase "a" does not count. None means no letter
    was among the tokens. letter_mass is the share of the listed probability
    that fell on letters before renormalizing.
    """
    p = {letter: 0.0 for letter in LETTERS}
    listed = 0.0
    for item in top_logprobs:
        value = math.exp(item["logprob"])
        listed += value
        token = item["token"].strip().strip("()*")
        if len(token) == 1 and token in p:
            p[token] += value
    mass = sum(p.values())
    if mass <= 0:
        return None, 0.0
    return {letter: value / mass for letter, value in p.items()}, mass / listed


def answer_probs(problem, mode="direct", timeout=60, url=URL, prefix=None):
    """Return ({"A": p, ..., "E": p} or None, timings).

    mode "direct": one call, one token, no grammar; the probabilities come from
    the top tokens of that position. None means no letter was among them.
    mode "reason:<budget>": not implemented yet.
    Set COPILOT_DEBUG_PROBS=1 to print the raw top tokens of the first call.
    """
    global _debug_printed
    if mode != "direct":
        raise NotImplementedError(f"mode {mode!r}: only 'direct' exists so far")
    response = post(url, {
        "messages": probs_messages(problem, prefix),
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
        "logprobs": True,
        "top_logprobs": TOP_LOGPROBS,
    }, timeout=timeout)
    top = response["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
    if os.environ.get("COPILOT_DEBUG_PROBS") and not _debug_printed:
        _debug_printed = True
        print("raw top tokens: " + ", ".join(f"{t['token']!r}:{math.exp(t['logprob']):.3f}" for t in top[:20]),
              file=sys.stderr)
    probs, letter_mass = letter_probs(top)
    timings = dict(response.get("timings", {}))
    timings["total_prompt_tokens"] = response.get("usage", {}).get("prompt_tokens")
    timings["letter_mass"] = letter_mass
    return probs, timings
