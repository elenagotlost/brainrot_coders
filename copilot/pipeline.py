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
  4. the model answers with one letter                      route = llm
     any error or timeout: the prior answers                route = fallback
"""
import socket
import time
import urllib.error

from . import priors, twins
from .config import CONFIG
from .llm import URL, answer_letter

TIME_LIMIT = float(CONFIG["TIME_LIMIT_SECONDS"])
FALLBACK = CONFIG["FALLBACK_ANSWER"]  # only used if the prior itself fails
MARGIN = 3.0  # seconds kept free for Python and for writing the answer
LETTERS = priors.LETTERS


def is_letter(value):
    return isinstance(value, str) and len(value) == 1 and value in LETTERS


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

    # 4. the model
    try:
        examples = []
        if index is not None and k > 0:
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
    except (socket.timeout, TimeoutError):
        return done(prior_letter, "fallback", "timeout")
    except urllib.error.URLError as error:
        if isinstance(error.reason, (socket.timeout, TimeoutError)):
            return done(prior_letter, "fallback", "timeout")
        return done(prior_letter, "fallback", f"error: {error.reason}")
    except Exception as error:  # a malformed response, a dropped connection, ...
        return done(prior_letter, "fallback", f"error: {type(error).__name__}")
