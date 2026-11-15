"""Reason, then compute: tool-integrated reasoning with the math model (TIR_* in config.env).

For a text-only problem with five numeric options, the model writes a line or two of
reasoning, then a short ```python block. It is stopped there, the code runs in
copilot/sandbox.py, and what it printed is pasted back as an ```output block (the format
Qwen2.5-Math-Instruct was trained on). The model then writes \\boxed{letter}.

The answer comes from, in this order:
  boxed   the model's own \\boxed{...} after reading the output
  code    the code printed a value equal to exactly one option (the model ran out of tokens)
  forced  "The answer is \\boxed{" is added and the model writes one letter

With a deadline, every call is sized so the problem fits in time: how many tokens the model
may still write comes from the speeds measured on the previous calls.

Standard library only: this runs on the board.
"""
import math
import re
import time
from collections import Counter
from fractions import Fraction

from . import sandbox
from .client import get, post
from .config import CONFIG
from .llm import format_problem

LETTERS = "ABCDE"
URL = f"http://127.0.0.1:{CONFIG.get('CODE_PORT', '8091')}"
MAX_TOKENS = int(CONFIG.get("CODE_MAX_TOKENS", "200"))
READ_TPS = float(CONFIG.get("CODE_READ_TPS", "6.0"))     # starting guesses; measured on every call
WRITE_TPS = float(CONFIG.get("CODE_WRITE_TPS", "3.5"))
FORCE = "\n\nThe answer is \\boxed{"
STOP = "```output"
# Qwen2.5-Math's own instruction for tool-integrated reasoning, plus "be brief".
SYSTEM = ("Please integrate natural language reasoning with programs to solve the problem above, "
          "and put your final answer within \\boxed{}. Be brief: one or two short lines of "
          "reasoning, then a short Python program that prints the result, then the letter of "
          "the option.")
# Made-up problems (not from M3Kang), in exactly the format we want. They come first in every
# prompt, so llama-server reads them once and reuses them from its cache.
EXAMPLES = [
    ("How many three-digit numbers have digits that add up to 5?\n"
     "A) 10\nB) 12\nC) 15\nD) 18\nE) 21",
     "Try every three-digit number and count those whose digits add up to 5.\n"
     "```python\nprint(sum(1 for n in range(100, 1000) if sum(map(int, str(n))) == 5))\n```\n"
     "```output\n15\n```\n15 is option C.\n\\boxed{C}"),
    ("Lea and Max have 25 stickers together. Lea has 7 more stickers than Max. "
     "How many stickers does Max have?\nA) 8\nB) 9\nC) 10\nD) 11\nE) 12",
     "Max has x, Lea has x + 7, and x + (x + 7) = 25. Try every x.\n"
     "```python\nprint([x for x in range(26) if x + (x + 7) == 25])\n```\n"
     "```output\n[9]\n```\n9 is option B.\n\\boxed{B}"),
]
CODE = re.compile(r"```python\s*\n(.*?)```", re.S)
BOXED = re.compile(r"\\boxed\{((?:[^{}]|\{[^{}]*\})*)\}")
CHARS_PER_TOKEN = 3.6

# --- options as numbers ----------------------------------------------------------------

# A unit or a short word after the number is ignored: "12 cm", "15 €", "3 years", "25%".
UNIT = r"(?:\s*[a-zA-Zàèéíòóúç°€$%²³\.]{1,12}){0,2}"
NUMBER = re.compile(
    r"^\s*([-−]?\d{1,3}(?:[ \u00a0\u202f]\d{3})+|[-−]?\d+(?:[.,]\d+)?)"   # 1 000 000, 12, 3,5, -2
    r"(?:\s*/\s*(\d+(?:[.,]\d+)?))?" + UNIT + r"\s*$")                     # optional /4


def _fraction(text):
    return Fraction(re.sub(r"[ \u00a0\u202f]", "", text).replace("−", "-").replace(",", "."))


def option_value(text):
    """'12', '3,5', '3/4', '25%', '12 cm', '1 000' as a Fraction; None when it is not one number."""
    m = NUMBER.match(str(text))
    if not m:
        return None
    value = _fraction(m.group(1))
    if m.group(2):
        den = _fraction(m.group(2))
        if den == 0:
            return None
        value /= den
    return value


def numeric_options(problem):
    """{letter: Fraction} when all five options are plain, different numbers; else None."""
    options = problem.get("options") or {}
    if any(k not in options for k in LETTERS):
        return None
    values = {k: option_value(options[k]) for k in LETTERS}
    if any(v is None for v in values.values()) or len(set(values.values())) != 5:
        return None
    return values


def matches(value, values):
    """Letters whose option equals the value (exactly, or within 1e-6 for decimals)."""
    if isinstance(value, str):
        value = option_value(value)
    if value is None or isinstance(value, bool):
        return []
    if isinstance(value, float):
        if not math.isfinite(value):
            return []
        return [k for k, v in values.items() if math.isclose(value, float(v), rel_tol=1e-6, abs_tol=1e-9)]
    return [k for k, v in values.items() if v == Fraction(value)]


def applies(problem):
    """True for the problems this method is for: text only, five numeric options."""
    return bool(numeric_options(problem)) and problem.get("text_only", True) is not False


def boxed_answer(text, values):
    """The letter of the option in the last \\boxed{...}, given as its letter or its value.
    Returns (letter or '', what was in the box or None)."""
    found = BOXED.findall(text or "")
    if not found:
        return "", None
    raw = found[-1].strip()
    # \frac{3}{4} -> 3/4; \text{...}, \, and units removed
    raw = re.sub(r"\\[dt]?frac\{([^{}]*)\}\{([^{}]*)\}", r"\1/\2", raw)
    raw = re.sub(r"\\text\{[^{}]*\}|\\,|\\!|\\%|\$|\\circ|\^\{?\\circ\}?", "", raw).strip()
    letter = re.fullmatch(r"\(?([A-E])\)?(?:\)|\.|:.*)?", raw)
    if letter:
        return letter.group(1), raw
    hits = matches(raw, values)
    return (hits[0] if len(hits) == 1 else ""), raw

# --- the model calls -------------------------------------------------------------------


def messages_for(problem):
    messages = [{"role": "system", "content": SYSTEM}]
    for question, answer in EXAMPLES:
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": answer}]
    return messages + [{"role": "user", "content": format_problem(problem)}]


def looped(text):
    """True when one non-trivial line appears three times or more: the model is going round."""
    lines = [x.strip() for x in text.splitlines() if len(x.strip()) > 8]
    return any(c >= 3 for c in Counter(lines).values())


class Speeds:
    """Reading and writing speed of the server, updated after every call (tokens per second)."""

    def __init__(self, read=READ_TPS, write=WRITE_TPS):
        self.read, self.write = read, write

    def update(self, timings):
        if timings.get("prompt_n", 0) >= 10 and timings.get("prompt_per_second"):
            self.read = 0.5 * self.read + 0.5 * timings["prompt_per_second"]
        if timings.get("predicted_n", 0) >= 10 and timings.get("predicted_per_second"):
            self.write = 0.5 * self.write + 0.5 * timings["predicted_per_second"]


def complete(url, prompt, n, stop=None, grammar=None, timeout=900, speeds=None, seed=42):
    payload = {"prompt": prompt, "n_predict": n, "temperature": 0, "seed": seed, "cache_prompt": True}
    if stop:
        payload["stop"] = stop
    if grammar:
        payload["grammar"] = grammar
    r = post(url + "/completion", payload, timeout=timeout)
    timings = r.get("timings", {})
    if speeds:
        speeds.update(timings)
    used = r.get("tokens_predicted") or timings.get("predicted_n", 0)
    return r["content"], used, r.get("stop_type", ""), timings.get("prompt_n", 0)


def solve(url, problem, budget=MAX_TOKENS, max_rounds=3, run_timeout=3.0, deadline=None, speeds=None,
          reserve=4.0):
    """Solve one problem. deadline (a time.time() value) caps the total time; reserve is kept
    for the final forced answer. Returns {"letter", "source", "code_letter", "rounds",
    "code_errors", "looped", "gen_tokens", "prompt_tokens", "text"}."""
    url = url.rstrip("/")
    values = numeric_options(problem)
    speeds = speeds or Speeds()
    prompt = post(url + "/apply-template", {"messages": messages_for(problem)}, timeout=60)["prompt"]
    text, used, rounds, read, outputs, first = "", 0, 0, None, [], True
    while used < budget:
        allowed = budget - used
        if deadline is not None:
            left = deadline - time.time() - reserve
            if first:  # the first call also reads the problem
                left -= (len(format_problem(problem)) / CHARS_PER_TOKEN + 10) / speeds.read
            allowed = min(allowed, int(left * speeds.write))
            if allowed < 8:
                break
        timeout = 900 if deadline is None else max(5.0, deadline - time.time())
        out, n, stop_type, prompt_n = complete(url, prompt + text, allowed, stop=[STOP],
                                               timeout=timeout, speeds=speeds)
        first = False
        read = prompt_n if read is None else read
        text += out
        used += n
        if looped(text) or boxed_answer(text, values)[1] is not None:
            break
        blocks = CODE.findall(text)
        if stop_type == "word" and blocks and rounds < max_rounds:
            printed, error = sandbox.run(blocks[-1], timeout=run_timeout)
            shown = ((printed or "") + ("\n" + error if error else "")).strip()[-200:]
            outputs.append({"printed": printed, "error": error})
            text += STOP + "\n" + (shown or "(nothing was printed)") + "\n```\n"
            rounds += 1
            continue
        break  # end of text, budget used, or a code block we won't run

    letter, _ = boxed_answer(text, values)
    source = "boxed" if letter else ""
    code_letter = ""
    for o in reversed(outputs):  # the last printed number that is exactly one option
        nums = re.findall(r"-?\d+(?:\.\d+)?(?:/\d+)?", o["printed"] or "")
        if nums:
            hits = matches(nums[-1], values)
            code_letter = hits[0] if len(hits) == 1 else ""
            break
    if not letter and code_letter:
        letter, source = code_letter, "code"
    if not letter:
        timeout = 60 if deadline is None else max(3.0, deadline - time.time() + reserve)
        forced, _, _, _ = complete(url, prompt + text + FORCE, 1, grammar="root ::= [A-E]",
                                   timeout=timeout, speeds=speeds)
        letter, source = forced.strip()[:1], "forced"
    return {"letter": letter, "source": source, "code_letter": code_letter, "rounds": rounds,
            "code_errors": sum(bool(o["error"]) for o in outputs), "looped": looped(text),
            "gen_tokens": used, "prompt_tokens": read or 0, "text": text}


def warm_up(url=URL):
    """One call at start-up: the server reads the fixed instructions and examples once, so no
    problem pays for them. Returns the seconds it took, or None when no server answers."""
    start = time.time()
    try:
        get(url + "/health", timeout=2)
        dummy = {"text": "What is 1 + 1?", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}
        solve(url, dummy, budget=1)
    except Exception:
        return None
    return time.time() - start
