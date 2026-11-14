#!/usr/bin/env python3
"""M2, program-of-thought: the model writes one arithmetic expression, Python computes it.

Only for problems whose five options are all plain numbers. The model's output is limited by a
grammar to a short arithmetic expression (numbers, + - * / // % **, brackets and a few
functions). Python evaluates it, and the answer is ACCEPTED only when the value equals exactly
one option. Everything else is a miss and the cascade carries on with the next step.

    python3 laptop/pot.py --input data/train.jsonl --model q2b-q4 --sample 200 --url http://127.0.0.1:8092

Reports coverage (how often an answer is accepted) and precision (how often an accepted answer
is right). Precision is the number that matters: an accepted answer overrides everything else.
The grammar demands at least one operator, and an expression that contains the matching option
itself ("15 * 1") counts as "trivial" and is never accepted: that is naming an option, not computing.

Writes results/pot-<model>-<split>.csv. Standard library only.
"""
import argparse
import ast
import csv
import json
import math
import operator
import os
import re
import sys
import time
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copilot.client import post          # noqa: E402
from copilot.config import CONFIG        # noqa: E402
from copilot.llm import format_problem   # noqa: E402
from dump_probs import sample_ids        # noqa: E402

LETTERS = "ABCDE"
SYSTEM = ("You solve multiple-choice math competition problems for school students. "
          "Reply with one Python arithmetic expression that computes the answer, using the "
          "numbers from the problem. Reply with the expression only.")

# Two made-up examples (not from M3Kang) show the format. They sit in front of every problem,
# so llama-server reads them once and reuses them from its cache (checked in E0).
EXAMPLES = [
    ("Tom has 4 boxes with 6 pencils in each box. He gives away 5 pencils. How many pencils "
     "does he have left?\nA) 15\nB) 19\nC) 24\nD) 29\nE) 34", "4 * 6 - 5"),
    ("The sum of three consecutive whole numbers is 48. What is the largest of them?\n"
     "A) 15\nB) 16\nC) 17\nD) 18\nE) 24", "48 / 3 + 1"),
]

# At least one operator: a bare number would be the model naming an option, not computing.
GRAMMAR = r'''
root   ::= factor (ws op ws factor)+
expr   ::= factor (ws op ws factor)*
op     ::= "**" | "*" | "//" | "/" | "%" | "+" | "-"
factor ::= "-"? (number | "(" ws expr ws ")" | func "(" ws expr (ws "," ws expr)* ws ")")
number ::= [0-9]+ ("." [0-9]+)?
func   ::= "sqrt" | "min" | "max" | "abs" | "comb" | "factorial" | "gcd" | "lcm"
ws     ::= " "?
'''

BINARY = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
          ast.Div: operator.truediv, ast.FloorDiv: operator.floordiv, ast.Mod: operator.mod}
FUNCS = {"min": min, "max": max, "abs": abs, "comb": math.comb, "factorial": math.factorial,
         "gcd": math.gcd, "lcm": math.lcm}


def evaluate(expression):
    """Value of the expression as a Fraction (exact) or a float (after sqrt). Raises on anything odd."""
    def ev(node):
        if isinstance(node, ast.Constant) and type(node.value) in (int, float):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -ev(node.operand)
        if isinstance(node, ast.BinOp):
            a, b = ev(node.left), ev(node.right)
            if isinstance(node.op, ast.Pow):
                if abs(b) > 64 or abs(a) > 10 ** 6:
                    raise ValueError("power too large")
                return a ** b
            return BINARY[type(node.op)](a, b)
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
            args = [ev(a) for a in node.args]
            if node.func.id == "sqrt":
                return math.sqrt(args[0])
            if node.func.id in ("comb", "factorial", "gcd", "lcm"):
                if any(a.denominator != 1 or a < 0 or a > 1000 for a in args):
                    raise ValueError("whole numbers only")
                return Fraction(FUNCS[node.func.id](*[int(a) for a in args]))
            return FUNCS[node.func.id](*args)
        raise ValueError("not allowed")
    return ev(ast.parse(expression.strip(), mode="eval").body)


NUMBER = re.compile(r"^(-?\d+(?:[.,]\d+)?)(?:\s*/\s*(\d+(?:[.,]\d+)?))?\s*(%?)$")


def option_value(text):
    """'12', '3,5', '3/4', '25%' as a Fraction; None when the option is not a plain number."""
    m = NUMBER.match(text.strip().replace("−", "-"))
    if not m:
        return None
    value = Fraction(m.group(1).replace(",", "."))
    if m.group(2):
        if Fraction(m.group(2).replace(",", ".")) == 0:
            return None
        value /= Fraction(m.group(2).replace(",", "."))
    return value


def numeric_options(problem):
    """{letter: Fraction} when all five options are plain, different numbers; else None."""
    options = problem.get("options") or {}
    values = {k: option_value(str(options.get(k, ""))) for k in LETTERS}
    if any(v is None for v in values.values()) or len(set(values.values())) != 5:
        return None
    return values


def match(value, values):
    """The one letter whose option equals the value; '' when none or several do."""
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        hits = [k for k, v in values.items() if math.isclose(value, float(v), rel_tol=1e-9, abs_tol=1e-9)]
    else:
        hits = [k for k, v in values.items() if v == value]
    return hits[0] if len(hits) == 1 else ""


def messages_for(problem):
    messages = [{"role": "system", "content": SYSTEM}]
    for question, expression in EXAMPLES:
        messages += [{"role": "user", "content": question}, {"role": "assistant", "content": expression}]
    return messages + [{"role": "user", "content": format_problem(problem)}]


def literals(expression):
    return [Fraction(x) for x in re.findall(r"\d+(?:\.\d+)?", expression)]


def grounded(expression, problem):
    """True when every number in the expression (except 1, 2, 10, 100) is written in the problem text."""
    in_text = {Fraction(x.replace(",", ".")) for x in re.findall(r"\d+(?:[.,]\d+)?", problem["text"])}
    return all(x in in_text or x in (1, 2, 10, 100) for x in literals(expression))


def solve(url, problem, values, max_tokens=30, timeout=60):
    """Return (letter or '', expression, why, tokens). The letter is only set when accepted."""
    response = post(url + "/v1/chat/completions", {
        "messages": messages_for(problem),
        "grammar": GRAMMAR, "max_tokens": max_tokens, "temperature": 0,
        "seed": int(CONFIG["SEED"]), "cache_prompt": True,
    }, timeout=timeout)
    expression = response["choices"][0]["message"]["content"].strip()
    tokens = response.get("usage", {}).get("completion_tokens", 0)
    if response["choices"][0].get("finish_reason") == "length":
        return "", expression, "invalid", tokens  # cut off at max_tokens: not a whole expression
    try:
        value = evaluate(expression)
    except Exception:
        return "", expression, "invalid", tokens
    letter = match(value, values)
    if not letter:
        return "", expression, "no-option", tokens
    if any(x == values[letter] for x in literals(expression)):
        # "15 * 1", "14 + 0": the option itself is written in the expression, nothing was computed
        return "", expression, "trivial", tokens
    return letter, expression, "accepted", tokens


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default=f"http://127.0.0.1:{CONFIG['LLM_PORT']}")
    ap.add_argument("--sample", type=int, default=0, help="N different ids, alternating languages")
    ap.add_argument("--max-tokens", type=int, default=30)
    args = ap.parse_args()

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    with_options = sum(bool(p.get("options")) for p in problems)
    eligible = [p for p in problems if numeric_options(p)]
    print(f"{len(problems)} rows, {with_options} with options, {len(eligible)} with five plain numbers",
          file=sys.stderr)
    if args.sample:
        eligible = sample_ids(eligible, args.sample)

    split = os.path.splitext(os.path.basename(args.input))[0]
    out = f"results/pot-{args.model}-{split}.csv"
    os.makedirs("results", exist_ok=True)
    why_count, accepted, right, g_accepted, g_right, t_start = {}, 0, 0, 0, 0, time.time()
    with open(out, "w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["id", "language", "level", "label", "answer", "why", "correct", "grounded",
                    "expression", "tokens"])
        for i, p in enumerate(eligible, 1):
            try:
                letter, expression, why, tokens = solve(args.url, p, numeric_options(p), args.max_tokens)
            except Exception as e:
                letter, expression, why, tokens = "", "", f"error: {e}", 0
            why_count[why] = why_count.get(why, 0) + 1
            accepted += bool(letter)
            right += letter == p["label"]
            is_grounded = bool(expression) and grounded(expression, p)
            g_accepted += bool(letter) and is_grounded
            g_right += letter == p["label"] and is_grounded
            w.writerow([p["id"], p.get("language", ""), p.get("level", ""), p["label"], letter, why,
                        int(letter == p["label"]), int(is_grounded), expression, tokens])
            fh.flush()
            if i % 50 == 0 or i == len(eligible):
                print(f"  [{i}/{len(eligible)}] accepted {accepted}, right {right}", file=sys.stderr)
    n = len(eligible)
    print(f"\nCoverage:  {accepted}/{n} = {100 * accepted / n:.1f}% of numeric problems get an answer",
          file=sys.stderr)
    if accepted:
        print(f"Precision: {right}/{accepted} = {100 * right / accepted:.1f}% of those answers are right",
              file=sys.stderr)
    if g_accepted:
        print(f"Only expressions built from numbers in the problem text: {g_right}/{g_accepted} = "
              f"{100 * g_right / g_accepted:.1f}% right, coverage {100 * g_accepted / n:.1f}%", file=sys.stderr)
    print(f"Outcomes:  {why_count}   ({(time.time() - t_start) / n:.2f} s/problem here)", file=sys.stderr)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
