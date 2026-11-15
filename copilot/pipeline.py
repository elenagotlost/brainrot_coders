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
  3b. code stage (when a code model is serving, see copilot/coder.py): the
     model writes a short Python program, the board runs it, and the output
     must equal exactly one option            route = code, else code-prior
  4. time guard: if reading the prompt would pass the time limit, the prior
     answers                                                route = prior-time
  5. the model gives P(A..E); its letter bias is divided out and the result
     is fused with the prior                                route = fused
     any error, timeout or missing letter: the prior answers
                                                            route = fallback

Two optional extras sit in stage 5. --fewshot N (R6) adds N solved training
problems to the system prompt; the server reads them once. --perms N (R5) asks
the question N times with the options moved round and averages the
probabilities, mapped back to the original letters.

The old retrieval path (--rag K) still asks for one letter: route = llm.
"""
import math
import os
import socket
import time
import urllib.error

from . import coder, priors, twins
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


def shift_options(problem, shift):
    """R5: the same problem with its options moved `shift` places (A takes B's text, ...)."""
    if not shift:
        return problem
    options = problem["options"]
    return {**problem, "options": {LETTERS[i]: options[LETTERS[(i + shift) % 5]] for i in range(5)}}


def warm_up(ctx, llm_url=URL, timeout=300.0):
    """One call at start-up, so the server reads the fixed prefix before the first
    problem and every problem only pays for its own text. Returns the seconds it
    took, or None if it failed (the time guard then counts the prefix as unread)."""
    if not ctx.get("prefix"):
        return None
    dummy = {"text": "What is 1 + 1?", "options": {"A": "1", "B": "2", "C": "3", "D": "4", "E": "5"}}
    begin = time.perf_counter()
    try:
        answer_probs(dummy, mode=ctx.get("mode") or "direct", timeout=timeout, url=llm_url,
                     prefix=ctx["prefix"])
    except Exception:
        ctx["prefix_warm"] = False
        return None
    ctx["prefix_warm"] = True
    return time.perf_counter() - begin


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
    {"twins": TwinIndex or None, "calib": dict or None, "no_llm": bool, "mode": str,
     "probs": saved probabilities to replay instead of calling the model, or None,
     "prefix": fixed text for the system prompt (R6), "prefix_warm": it is already in
     the server's cache, "perms": how many option orders to ask and average (R5)}.
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

    # 3b. the code stage: the model writes a short program, the board runs it, and the
    # answer counts only if the output equals exactly one option. When it has no answer
    # the prior answers: the time is spent, and the letter-picking model adds nothing.
    if ctx.get("code"):
        try:
            left = time_limit - MARGIN - (time.perf_counter() - start)
            letter, info = coder.solve(problem, left, url=ctx.get("code_url") or coder.URL)
            result["prompt_tokens"] = info.get("prompt_tokens", 0)
            result["code_why"], result["code_tokens"] = info["why"], info.get("tokens", 0)
            # Only a program that computes and hits exactly one option is trusted. "nearest" and
            # "guess" were right 4 times in 38 on the practice set, the prior 10 times on the same rows.
            if is_letter(letter) and info["why"].startswith("accepted"):
                return done(letter, "code")
            return done(prior_letter, "code-prior")
        except Exception as error:
            result["code_why"] = failure(error)
            return done(prior_letter, "code-prior", failure(error))

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

    if ctx.get("probs") is not None:
        # replay: probabilities saved by laptop/dump_probs.py stand in for the model
        # call, so a whole split is scored in seconds. Latency means nothing here.
        saved = ctx["probs"].get((problem.get("id"), problem.get("language", "")))
        if saved is None:
            return done(prior_letter, "fallback", "error: no saved probabilities for this problem")
        p_llm, result["letter_mass"] = saved
    else:
        mode = ctx.get("mode") or "direct"
        prefix = ctx.get("prefix")
        # a prefix that the warm-up call already put in the server's cache costs no reading time
        counted_prefix = None if ctx.get("prefix_warm") else prefix
        total = {letter: 0.0 for letter in LETTERS}
        masses, used = [], 0
        for shift in range(max(1, int(ctx.get("perms") or 1))):
            asked = shift_options(problem, shift)
            # 4. time guard, before every call
            elapsed = time.perf_counter() - start
            reading = estimated_prefill_s(probs_messages(asked, counted_prefix))
            if elapsed + reading + GENERATION_S > time_limit - MARGIN:
                if shift == 0:
                    return done(prior_letter, "prior-time")
                break  # no time for another permutation: keep what we have
            # 5. the model's probabilities
            try:
                remaining = time_limit - (time.perf_counter() - start) - MARGIN
                p_shift, timings = answer_probs(asked, mode=mode, timeout=max(remaining, 1.0),
                                                url=llm_url, prefix=prefix)
            except NotImplementedError:
                raise  # an unknown mode is a mistake in the command, not a model failure
            except Exception as error:
                if shift == 0:
                    return done(prior_letter, "fallback", failure(error))
                break
            result["prompt_tokens"] += timings.get("prompt_n", 0) or 0
            if p_shift is None:
                continue
            for i, letter in enumerate(LETTERS):  # option i of the asked order is option i + shift of ours
                total[LETTERS[(i + shift) % 5]] += p_shift[letter]
            masses.append(timings.get("letter_mass") or 0.0)
            used += 1
        result["perms_used"] = used
        p_llm = {letter: total[letter] / used for letter in LETTERS} if used else None
        result["letter_mass"] = sum(masses) / used if used else None

    # the model's letter bias is divided out, then the result is fused with the prior
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
