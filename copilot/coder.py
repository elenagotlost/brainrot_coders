"""The code stage: the model writes a short Python program, the board runs it,
and the answer is accepted only when the output equals exactly one option.

Why: the small models can't reason their way through a Kangaroo problem in the
tokens that fit in 60 seconds, but many problems are computations (count,
enumerate, last digit, divisibility) and a few lines of brute force solve them.
Code is much shorter than prose working, and Python does the arithmetic the
model gets wrong.

The code model is Qwen2.5-Coder 1.5B (config.env). On the board it reads about
4.3 tokens/s and writes about 2.8, so the number of tokens it may write is set
from the time left (see budget()). Measure any change with laptop/code_lab.py.

Standard library only. The model's program runs in a separate Python process
with a time limit, no dangerous names and a short list of allowed imports.
"""
import os
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
SYSTEM = ("You solve math competition problems by writing a very short Python 3 program that finds the answer "
          "by brute force and prints only the final answer. math, itertools and Fraction are already imported. "
          "Reply with code only. Never reply with the answer itself: always compute it.")
# Solved examples, read once at warm-up and then cached by the server: they cost no time per
# problem. Written for this file (not M3Kang, not the practice set). They teach the style:
# enumerate, test the condition, print. Set CODE_FEWSHOT=0 to switch them off.
EXAMPLES = [
    ("How many whole numbers from 1 to 200 have digits that add up to 9?",
     "print(sum(1 for n in range(1,201) if sum(map(int,str(n)))==9))"),
    ("Marta has 12 coins. Some are worth 2 cents and the rest 5 cents. Together they are worth 39 cents. "
     "How many 5-cent coins does she have?",
     "for a in range(13):\n    if 5*a+2*(12-a)==39:print(a)"),
    ("Four friends stand in a queue. Ana is somewhere ahead of Bea. Carla is directly behind Dani. "
     "Bea is not last. Who is second?",
     'for p in permutations(["Ana","Bea","Carla","Dani"]):\n    i=p.index\n'
     '    if i("Ana")<i("Bea") and i("Carla")==i("Dani")+1 and i("Bea")!=3:print(p[1])'),
    ("Two fair dice are rolled. What is the probability that the product of the two numbers is even?",
     "r=[a*b%2==0 for a in range(1,7) for b in range(1,7)]\nprint(Fraction(sum(r),len(r)))"),
    ("What is the smallest positive whole number that is divisible by 6 and whose digits add up to 15?",
     "n=1\nwhile not(n%6==0 and sum(map(int,str(n)))==15):n+=1\nprint(n)"),
    ("A train leaves every 25 minutes; the first one leaves at 7:10. Pau arrives at 9:00. "
     "At what time does the next train leave?",
     't=7*60+10\nwhile t<9*60:t+=25\nprint(f"{t//60}:{t%60:02d}")'),
    ("The digits 2, 5, 7 and 8 are used once each to write two two-digit numbers. "
     "What is the smallest possible product of the two numbers?",
     "print(min((10*a+b)*(10*c+d) for a,b,c,d in permutations([2,5,7,8])))"),
    ("How many subsets of {1, 2, 3, 4, 5, 6} have a sum of 10?",
     "print(sum(1 for k in range(7) for c in combinations(range(1,7),k) if sum(c)==10))"),
    ("A number starts at 3. At each step it is doubled and then 1 is subtracted. What is it after 8 steps?",
     "x=3\nfor _ in range(8):x=2*x-1\nprint(x)"),
]
FEWSHOT = os.environ.get("CODE_FEWSHOT", "1") != "0"
# The first attempt does not see the choices: with them the model prints one instead of computing.
SHOW_OPTIONS = os.environ.get("CODE_OPTIONS", "0") != "0"
# Put in front of every program: the imports the model forgets cost it no tokens.
PREAMBLE = ("import math, itertools, fractions, collections, functools\nfrom math import *\n"
            "from itertools import *\nfrom fractions import Fraction\nfrom collections import *\n")
ALLOWED = {"math", "itertools", "fractions", "collections", "functools", "datetime"}
BANNED = re.compile(r"__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\binput\s*\(|\bos\b|\bsys\b|\bsubprocess\b|\bsocket\b")


def clean_option(text):
    """An option as the model should read it: OCR's long dash becomes a minus sign."""
    return re.sub(r"^[一—–−]\s*", "-", text.strip())


def user_text(problem):
    text = re.sub(r"^\s*\d+\.\s*", "", problem["text"]).strip()
    if not SHOW_OPTIONS:
        return text
    return text + "\nChoices: " + ", ".join(clean_option(problem["options"][k]) for k in LETTERS)


def prefix():
    """The part of the prompt that is the same for every problem: the server caches it."""
    turns = [{"role": "system", "content": SYSTEM}]
    if FEWSHOT:
        for question, code in EXAMPLES:
            turns += [{"role": "user", "content": question}, {"role": "assistant", "content": code}]
    return turns


def messages(problem):
    return prefix() + [{"role": "user", "content": user_text(problem)}]


TEMPLATE_TOKENS = 12   # the chat template around the problem's text: measured 10 on the practice set


def prompt_tokens(problem, url=None):
    """The tokens the server must read for this problem (the prefix is cached after the warm-up).

    Asks the server's tokenizer for the exact count (milliseconds, no model run). Without a
    server, an estimate fitted on the practice set (0.27 * characters + 10, worst row +25)."""
    text = user_text(problem)
    if url:
        try:
            base = url.split("/v1/")[0]
            return len(post(base + "/tokenize", {"content": text}, timeout=5.0)["tokens"]) + TEMPLATE_TOKENS
        except Exception:
            pass
    return 0.30 * len(text) + 35


def budget(problem, seconds_left, url=None):
    """How many tokens the model may write so that reading, writing and running fit in seconds_left."""
    writing = seconds_left - prompt_tokens(problem, url) / READ_TPS - RUN_SECONDS
    return max(0, min(MAX_TOKENS, int(writing * WRITE_TPS)))


def extract(reply):
    reply = reply.split("</think>")[-1]
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
        done = subprocess.run([sys.executable, "-I", "-c", PREAMBLE + code], capture_output=True, text=True,
                              timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", "timeout"
    lines = [line.strip() for line in done.stdout.splitlines() if line.strip()]
    if done.returncode != 0 or not lines:
        return "", "error" if done.returncode else "no output"
    return lines[-1], ""


def number(text):
    text = clean_option(str(text)).replace("−", "-").replace(",", ".").replace(" ", "")
    text = re.sub(r"(cm|mm|km|m|kg|g|°|%|€|min|h|s)\.?$", "", text)
    try:
        return Fraction(text)
    except (ValueError, ZeroDivisionError):
        return None


def option_numbers(options):
    """The five options as lists of possible numbers ([] where an option is not a number).

    OCR sometimes reads a stray digit before a number ("1 16" for 16, "2 4" for 24). The
    options of a problem are written in ascending order, so of the two readings (digits
    joined, or the last group alone) those that keep the order are kept."""
    single = [number(options[k]) for k in LETTERS]
    values = [[] if v is None else [v] for v in single]
    for i, letter in enumerate(LETTERS):
        raw = clean_option(options[letter])
        if not re.fullmatch(r"-?\d+\s+\d+", raw):
            continue
        low = next((v for v in reversed(single[:i]) if v is not None), None)
        high = next((v for v in single[i + 1:] if v is not None), None)
        fits = [v for v in (number(raw), number(raw.split()[-1]))
                if (low is None or v > low) and (high is None or v < high)]
        if fits:
            values[i] = fits
    return values


def same_text(a, b):
    """Text options: ignore case, spaces and the letters OCR confuses (l, I, 1)."""
    def squash(text):
        return re.sub(r"[il1|]", "l", re.sub(r"\s+", "", text.lower()))
    return squash(a) == squash(b)


def match(output, options):
    """The one letter whose value equals the output, or None."""
    value = number(output)
    hits = []
    for letter, option_value in zip(LETTERS, option_numbers(options)):
        if value is not None and option_value:
            if any(abs(v - value) < Fraction(1, 1000) for v in option_value):
                hits.append(letter)
        elif same_text(options[letter], output):
            hits.append(letter)
    return hits[0] if len(hits) == 1 else None


def computes(code):
    """False for a "program" that only prints a number the model guessed."""
    return bool(re.search(r"\bfor\b|\bwhile\b|\bsum\(|\blen\(|\brange\(|%|\*\*|//|itertools|math\.", code))


def nearest(output, options):
    """The letter of the numeric option closest to a numeric output, or None."""
    value, numbers = number(output), option_numbers(options)
    if value is None or not all(numbers):
        return None
    return min(LETTERS, key=lambda letter: min(abs(v - value) for v in numbers[LETTERS.index(letter)]))


def attempt(turns, max_tokens, timeout, url):
    """One call and one run. Returns (code, output, why, timings, simulated board seconds)."""
    start = time.perf_counter()
    # t_max_predict_ms: the server itself stops writing in time. Without it, a request the
    # client gave up on keeps the server busy, and the next problem waits behind it.
    write_ms = int(max(timeout, 1.0) * 1000)
    response = post(url, {"messages": turns, "max_tokens": max_tokens, "temperature": 0,
                          "seed": int(CONFIG["SEED"]), "cache_prompt": True, "t_max_predict_ms": write_ms},
                    timeout=max(timeout, 1.0))
    timings = response.get("timings", {})
    code = extract(response["choices"][0]["message"]["content"])
    run_start = time.perf_counter()
    output, why = run(code, timeout=RUN_SECONDS)
    run_s = time.perf_counter() - run_start
    board_s = timings.get("prompt_n", 0) / READ_TPS + timings.get("predicted_n", 0) / WRITE_TPS + run_s
    # on the board the wall clock is the truth; on a faster machine the simulated time is
    return code, output, why, timings, max(board_s, time.perf_counter() - start)


def solve(problem, seconds_left, url=URL):
    """Return (letter or None, info). info: why, code, output, tokens, prompt_tokens, max_tokens, board_s.

    why says how far to trust the letter:
      accepted   a program that computes printed exactly one option (first try)
      nearest    no program hit an option; the numeric option closest to its output
      guess      the program only printed a value, and it is an option
    Never takes longer than seconds_left on the board: the tokens of each try are budgeted
    from the time left. Raises only what post() raises (timeout, connection): "no answer".
    """
    info = {"why": "", "code": "", "output": "", "tokens": 0, "prompt_tokens": 0, "max_tokens": 0, "board_s": 0.0}
    options = problem.get("options") or {}
    if len(options) != 5 or any(k not in options for k in LETTERS):
        info["why"] = "no options"
        return None, info
    info["max_tokens"] = budget(problem, seconds_left, url)
    if info["max_tokens"] < MIN_TOKENS:
        info["why"] = "no time"
        return None, info
    turns = messages(problem)
    code, output, why, timings, spent = attempt(turns, info["max_tokens"], seconds_left - RUN_SECONDS, url)
    info.update(code=code, output=output, tokens=timings.get("predicted_n", 0),
                prompt_tokens=timings.get("prompt_n", 0), board_s=spent)
    letter = None if why else match(output, options)
    if letter and computes(code):
        info["why"] = "accepted"
        return letter, info
    # Not a trusted answer. The letter is still returned with its label, for the experiments;
    # the pipeline ignores it and lets the prior answer.
    first_why = why or ("guess" if letter else "no option")
    fallback = (letter, "guess") if letter else (nearest(output, options), "nearest") if not why else (None, first_why)
    info["why"] = fallback[1] if fallback[0] else first_why
    return fallback[0], info


def warm_up(url=URL, timeout=300.0):
    """One call at start-up, so the server has read the prefix before the first problem.
    Returns the seconds it took, or None when the code server is not there."""
    start = time.perf_counter()
    try:
        post(url, {"messages": prefix() + [{"role": "user", "content": "What is 1 + 1?"}],
                   "max_tokens": 1, "temperature": 0, "cache_prompt": True}, timeout=timeout)
    except Exception:
        return None
    return time.perf_counter() - start
