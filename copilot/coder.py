"""The code stage: the model writes a short Python program, the board runs it,
and the answer is accepted only when the output equals exactly one option.

Why: the small models can't reason their way through a Kangaroo problem in the
tokens that fit in 60 seconds, but many problems are computations (count,
enumerate, last digit, divisibility) and a few lines of brute force solve them.
Code is much shorter than prose working, and Python does the arithmetic the
model gets wrong.

Measured on the 100 datathon practice rows: Qwen3.5 2B answers about 44 this
way and is right on half of them, against 20% for every letter-picking variant.
On the board the 2B reads 4.6 tokens/s and writes 2.6, so the number of tokens
it may write is set from the time left (see budget()).

Standard library only. The model's program runs in a separate Python process
with a time limit, no dangerous names and a short list of allowed imports.
"""
import re
import subprocess
import sys
import time
from fractions import Fraction

from .client import post
from .config import CONFIG

LETTERS = "ABCDE"
URL = CONFIG.get("CODE_URL") or f"http://127.0.0.1:{CONFIG.get('CODE_PORT', CONFIG['LLM_PORT'])}/v1/chat/completions"
READ_TPS = float(CONFIG.get("CODE_READ_TPS", "4.6"))    # the code model on the board
WRITE_TPS = float(CONFIG.get("CODE_WRITE_TPS", "2.6"))
MAX_TOKENS = int(CONFIG.get("CODE_MAX_TOKENS", "96"))
MIN_TOKENS = 20        # less than this is not a program: skip the stage
RUN_SECONDS = 5.0      # how long the program may run
SYSTEM = ("You solve math competition problems by writing Python. Write a very short Python 3 program, at most "
          "10 lines, that computes the answer by brute force and prints only the final answer with print(). "
          "No comments, no functions, no explanation. Never answer directly: always reply with code only.")
ALLOWED = {"math", "itertools", "fractions", "collections", "functools", "datetime"}
BANNED = re.compile(r"__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\binput\s*\(|\bos\b|\bsys\b|\bsubprocess\b|\bsocket\b")


def user_text(problem):
    text = re.sub(r"^\s*\d+\.\s*", "", problem["text"]).strip()
    return text + "\nOptions: " + "  ".join(f"{k}) {problem['options'][k]}" for k in LETTERS)


def messages(problem):
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user_text(problem)}]


def prompt_tokens(problem):
    """A safe estimate of the tokens the server must read (the system prompt is cached after the warm-up).
    Fitted on the practice set: tokens = 0.29 * characters + 31; this is that line, rounded up."""
    return 0.30 * len(user_text(problem)) + 40


def budget(problem, seconds_left):
    """How many tokens the model may write so that reading, writing and running fit in seconds_left."""
    writing = seconds_left - prompt_tokens(problem) / READ_TPS - RUN_SECONDS
    return max(0, min(MAX_TOKENS, int(writing * WRITE_TPS)))


def extract(reply):
    blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|\Z)", reply, flags=re.S)
    return (blocks[0] if blocks else reply).strip()


def run(code, timeout=RUN_SECONDS):
    """Return (the last line the program printed, why it failed or "")."""
    if BANNED.search(code):
        return "", "banned"
    for module in re.findall(r"^\s*(?:import|from)\s+(\w+)", code, flags=re.M):
        if module not in ALLOWED:
            return "", "import"
    try:
        done = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", "timeout"
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0 or not lines:
        return "", "error" if done.returncode else "no output"
    return lines[-1], ""


def number(text):
    text = text.strip().replace("−", "-").replace(",", ".").replace(" ", "")
    text = re.sub(r"(cm|m|kg|g|°|%|€|km|min|h|s)\.?$", "", text)
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def match(output, options):
    """The one letter whose value equals the output, or None."""
    value = number(output)
    hits = []
    for letter in LETTERS:
        option = options[letter]
        if value is not None and number(option) is not None:
            if abs(number(option) - value) < Fraction(1, 1000):
                hits.append(letter)
        elif re.sub(r"\s+", "", option.lower()) == re.sub(r"\s+", "", output.lower()):
            hits.append(letter)
    return hits[0] if len(hits) == 1 else None


def computes(code):
    """False for a "program" that only prints a number the model guessed."""
    return bool(re.search(r"\bfor\b|\bwhile\b|\bsum\(|\blen\(|\brange\(|%|\*\*|//|itertools|math\.", code))


def solve(problem, seconds_left, url=URL):
    """Return (letter or None, info). info: why, code, output, tokens, prompt_tokens, max_tokens, read_s, write_s.

    Never takes longer than seconds_left: the tokens are budgeted, and the request
    itself is cut off in time. Raises nothing the caller must handle except what
    post() raises (timeout, connection), which means "no answer".
    """
    info = {"why": "", "code": "", "output": "", "tokens": 0, "prompt_tokens": 0, "max_tokens": 0}
    options = problem.get("options") or {}
    if len(options) != 5 or any(k not in options for k in LETTERS):
        info["why"] = "no options"
        return None, info
    info["max_tokens"] = budget(problem, seconds_left)
    if info["max_tokens"] < MIN_TOKENS:
        info["why"] = "no time"
        return None, info
    start = time.perf_counter()
    response = post(url, {"messages": messages(problem), "max_tokens": info["max_tokens"], "temperature": 0,
                          "seed": int(CONFIG["SEED"]), "cache_prompt": True},
                    timeout=max(seconds_left - RUN_SECONDS, 1.0))
    timings = response.get("timings", {})
    info.update(tokens=timings.get("predicted_n", 0), prompt_tokens=timings.get("prompt_n", 0),
                read_s=timings.get("prompt_ms", 0) / 1000, write_s=timings.get("predicted_ms", 0) / 1000)
    info["code"] = extract(response["choices"][0]["message"]["content"])
    left = seconds_left - (time.perf_counter() - start)
    if left < 1.0:
        info["why"] = "no time"
        return None, info
    info["output"], info["why"] = run(info["code"], timeout=min(RUN_SECONDS, left))
    if info["why"]:
        return None, info
    letter = match(info["output"], options)
    info["why"] = "accepted" if letter else "no option"
    return letter, info


CHECK_SYSTEM = ("You check a proposed answer to a math competition problem. Work out whether the proposed "
                "answer is correct. Reply with one word: yes or no.")


def check(problem, letter, seconds_left, url=URL):
    """Ask the code model whether a looked-up answer is right. Returns True, False, or None
    when it could not be asked in time (no options, no time, server error).

    Measured on the 100 practice rows: the model says yes to 52 right answers and to 51 wrong
    ones, so this is a second opinion, not a proof. "yes" or the same letter confirms; "no"
    or another letter rejects."""
    options = problem.get("options") or {}
    if letter not in options or any(k not in options for k in LETTERS):
        return None
    user = (user_text(problem) + f"\nProposed answer: {letter}) {options[letter]}"
            "\nIs the proposed answer correct?")
    # measured on the practice set: 0.29 * characters + 31 tokens with a system prompt of this
    # size, which is read again every time because the code stage's prompt replaces it in the cache
    reading = (0.30 * len(user) + 40) / READ_TPS
    if reading + 3 / WRITE_TPS > seconds_left:
        return None
    try:
        response = post(url, {"messages": [{"role": "system", "content": CHECK_SYSTEM},
                                           {"role": "user", "content": user}],
                              "max_tokens": 3, "temperature": 0, "seed": int(CONFIG["SEED"]),
                              "cache_prompt": True}, timeout=max(seconds_left, 1.0))
        reply = response["choices"][0]["message"]["content"].strip().lower()
    except Exception:
        return None
    if reply.startswith("yes"):
        return True
    if reply.startswith("no"):
        return False
    named = re.match(r"([a-e])\b", reply)
    return (named.group(1).upper() == letter) if named else None


def warm_up(url=URL, timeout=120.0):
    """One call at start-up, so the server has read the system prompt before the first problem.
    Returns the seconds it took, or None when the code server is not there."""
    start = time.perf_counter()
    try:
        post(url, {"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": "1 + 1"}],
                   "max_tokens": 1, "temperature": 0, "cache_prompt": True}, timeout=timeout)
    except Exception:
        return None
    return time.perf_counter() - start
