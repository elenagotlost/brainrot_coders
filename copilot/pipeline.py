"""The copilot cascade: one problem in, one letter out.

predict.py and eval.py both call solve(), so whatever happens here is exactly
what the official run measures. Cheapest stage first; each stage either
answers or passes the problem on:

  1. twin lookup (only with --twins): the same problem, already solved, in
     the bank                                               route = twin
  2. no options: the model would guess blind, so the prior answers without
     a model call                                           route = empty
  3. prior over the five options (median value or medoid string); with
     --no-llm it answers                    route = prior-numeric / prior-text
  4. time guard: if reading the prompt would pass the time limit, the prior
     answers                                                route = prior-time
  5. the model gives P(A..E); its letter bias is divided out and the result
     is fused with the prior                                route = fused
     any error, timeout or missing letter: the prior answers
                                                            route = fallback

The old retrieval path (--rag K) still asks for one letter: route = llm.
"""
import math
import os
import socket
import time
import urllib.error

from . import priors, twins
from .config import CONFIG
from .llm import URL, answer_letter, answer_probs, probs_messages

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]  # only used if the prior itself fails
MARGIN = 3.0  # seconds kept free for Python and for writing the answer
LETTERS = priors.LETTERS


# How fast the model reads a prompt, for the time guard. 11 tokens per second is
# the board; set TOKENS_PER_S to test the guard on a faster machine.
TOKENS_PER_S = float(os.environ.get("TOKENS_PER_S", "11"))
CHARS_PER_TOKEN = 3.6
GENERATION_S = 2.0  # one token out, plus the request itself
DEFAULT_ALPHA = 0.5  # weight of the model against the prior without a calibration file


def is_letter(value):
    return isinstance(value, str) and len(value) == 1 and value in LETTERS


def estimated_prefill_s(messages):
    chars = sum(len(m["content"]) for m in messages if isinstance(m["content"], str))
    return chars / CHARS_PER_TOKEN / TOKENS_PER_S


def debias(p_llm, bias):
    """Divide out the model's taste for some letters. bias is its mean P(letter) on train."""
    if not bias:
        return dict(p_llm)
    q = {k: p_llm[k] / max(float(bias[k]), 1e-6) for k in LETTERS}
    total = sum(q.values())
    return {k: v / total for k, v in q.items()}


def fuse(p_llm, p_prior, alpha):
    """argmax of alpha * log p_llm + (1 - alpha) * log p_prior. alpha = 0 is the prior alone."""
    score = {k: alpha * math.log(p_llm[k] + 1e-6) + (1 - alpha) * math.log(p_prior[k] + 1e-6)
             for k in LETTERS}
    return max(LETTERS, key=lambda k: score[k])


def failure(error):
    """A short status for a failed model call."""
    if isinstance(error, (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(error, urllib.error.URLError):
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            return "timeout"
        return f"error: {error.reason}"
    return f"error: {type(error).__name__}"


def solve(problem, index=None, k=0, exclude_id=None, image_b64=None, llm_url=URL,
          time_limit=TIME_LIMIT, ctx=None):
    """Return {"answer", "seconds", "status", "route", ...} for one problem.

    ctx holds the run options, built once by copilot.options.build_ctx:
    {"twins": TwinIndex or None, "calib": dict or None, "no_llm": bool, "mode": str}.
    exclude_id keeps a problem from finding its own translation in the twin
    bank or the RAG index (used while evaluating).
    index and k turn on the old retrieval path: the k most similar solved
    problems are added to the prompt as examples.
    image_b64 and llm_url send the problem image to a vision server.
    """
    start = time.perf_counter()
    ctx = ctx or {}
    result = {"answer": FALLBACK, "status": "ok", "route": "", "prompt_tokens": 0,
              "retrieved": "", "examples": [], "twin": {}}

    def done(answer, route, status=None):
        result["answer"] = answer
        result["route"] = route
        if status:
            result["status"] = status
        result["seconds"] = round(time.perf_counter() - start, 2)
        return result

    # 1. twin lookup
    if ctx.get("twins") is not None:
        try:
            skip = (lambda d: d["id"] == exclude_id) if exclude_id is not None else (lambda d: False)
            letter, info = twins.answer(ctx["twins"], problem, exclude=skip)
            result["twin"] = info
            if is_letter(letter):
                return done(letter, "twin")
        except Exception as error:  # a broken lookup must never cost the answer
            result["status"] = f"twin error: {type(error).__name__}"

    # 2 and 3. the prior
    try:
        p_prior, kind = priors.prior(problem)
        prior_letter = max(LETTERS, key=lambda letter: p_prior[letter])
    except Exception as error:
        return done(FALLBACK, "fallback", f"prior error: {type(error).__name__}")
    if kind == "empty":
        return done(prior_letter, "empty")
    if ctx.get("no_llm"):
        return done(prior_letter, f"prior-{kind}")

    # the old retrieval path: examples in the prompt, one letter out
    if index is not None and k > 0:
        try:
            hits = index.search(problem, k=k, exclude_id=exclude_id)
            examples = [p for _score, p in hits]
            result["examples"] = hits
            result["retrieved"] = ";".join(f"{p['id']}:{s:.2f}" for s, p in hits)
            remaining = time_limit - (time.perf_counter() - start) - MARGIN
            letter, timings = answer_letter(problem, examples, timeout=max(remaining, 1.0),
                                            image_b64=image_b64, url=llm_url)
            result["prompt_tokens"] = timings.get("prompt_n", 0)
            if not is_letter(letter):
                return done(prior_letter, "fallback", "error: no letter from the model")
            return done(letter, "llm")
        except Exception as error:
            return done(prior_letter, "fallback", failure(error))

    # 4. time guard
    mode = ctx.get("mode") or "direct"
    prefix = ctx.get("prefix")
    elapsed = time.perf_counter() - start
    if elapsed + estimated_prefill_s(probs_messages(problem, prefix)) + GENERATION_S > time_limit - MARGIN:
        return done(prior_letter, "prior-time")

    # 5. the model's probabilities, without its letter bias, fused with the prior
    try:
        remaining = time_limit - (time.perf_counter() - start) - MARGIN
        p_llm, timings = answer_probs(problem, mode=mode, timeout=max(remaining, 1.0),
                                      url=llm_url, prefix=prefix)
    except NotImplementedError:
        raise  # an unknown mode is a mistake in the command, not a model failure
    except Exception as error:
        return done(prior_letter, "fallback", failure(error))
    result["prompt_tokens"] = timings.get("prompt_n", 0)
    result["letter_mass"] = timings.get("letter_mass")
    if p_llm is None:
        return done(prior_letter, "fallback", "error: no letter among the model's top tokens")
    for letter in LETTERS:
        result["p" + letter] = p_llm[letter]
    try:
        calib = ctx.get("calib") or {}
        alpha = float((calib.get("alpha") or {}).get(kind, DEFAULT_ALPHA))
        return done(fuse(debias(p_llm, calib.get("bias")), p_prior, alpha), "fused")
    except Exception as error:  # a malformed calibration file
        return done(prior_letter, "fallback", f"calib error: {type(error).__name__}")
