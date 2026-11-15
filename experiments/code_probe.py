#!/usr/bin/env python3
"""Probe: let the model write a short Python program, run it, and match the output to an option.

    python3 laptop/code_probe.py --input data/demo/train.jsonl --model q2b --url http://127.0.0.1:8091

The datathon practice problems are short and computational (count, enumerate,
last digit, divisibility): a few lines of brute force solve many of them, and
code is far shorter than prose reasoning. The answer is accepted only when the
program's output equals exactly one option; otherwise the problem is "not covered"
and the cascade would fall back.

Reports coverage (answered), precision (right among answered) and the tokens
written, which decide the time on the board. Writes results/code-<model>-<split>.csv.
Standard library only. The model's code runs in a separate Python process with a
time limit and a short list of allowed imports.
"""
import argparse
import csv
import json
import os
import re
import subprocess
import sys
import urllib.request
from collections import Counter
from fractions import Fraction

LETTERS = "ABCDE"
SYSTEM = ("You solve math competition problems by writing Python. Write a very short Python 3 program, at most "
          "10 lines, that computes the answer by brute force and prints only the final answer with print(). "
          "No comments, no functions, no explanation. Never answer directly: always reply with code only.")
ALLOWED = {"math", "itertools", "fractions", "collections", "functools", "datetime"}
BANNED = re.compile(r"__|\bopen\s*\(|\beval\s*\(|\bexec\s*\(|\binput\s*\(|\bos\b|\bsys\b|\bsubprocess\b|\bsocket\b")


def ask(url, problem, max_tokens, with_options):
    text = re.sub(r"^\s*\d+\.\s*", "", problem["text"]).strip()
    if with_options:
        text += "\nOptions: " + "  ".join(f"{k}) {v}" for k, v in problem["options"].items())
    body = json.dumps({"messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": text}],
                       "max_tokens": max_tokens, "temperature": 0, "seed": 42, "cache_prompt": True}).encode()
    request = urllib.request.Request(url.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=300) as response:
        data = json.loads(response.read())
    timings = data.get("timings", {})
    ask.last = timings  # read and write times, for the board
    return data["choices"][0]["message"]["content"], timings.get("prompt_n", 0), timings.get("predicted_n", 0)


def extract(reply):
    blocks = re.findall(r"```(?:python)?\n(.*?)(?:```|\Z)", reply, flags=re.S)
    return (blocks[0] if blocks else reply).strip()


def run(code, timeout):
    """Return (stdout's last line, why it failed or "")."""
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


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--model", required=True, help="short name for the output file")
    p.add_argument("--url", default="http://127.0.0.1:8081")
    p.add_argument("--max-tokens", type=int, default=128)
    p.add_argument("--no-options", action="store_true", help="don't show the options to the model")
    p.add_argument("--timeout", type=float, default=5.0, help="seconds the program may run")
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if len(x.get("options") or {}) == 5]
    if args.limit:
        problems = problems[:args.limit]
    split = os.path.splitext(os.path.basename(args.input))[0]
    out_path = f"results/code-{args.model}-{split}.csv"
    os.makedirs("results", exist_ok=True)
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "level", "text_only", "label", "answer", "correct", "why",
                         "output", "prompt_tokens", "tokens", "code"])
        for n, x in enumerate(problems, 1):
            reply, prompt_n, tokens = ask(args.url, x, args.max_tokens, not args.no_options)
            code = extract(reply)
            output, why = run(code, args.timeout)
            answer = match(output, x["options"]) if not why else None
            if not why and answer is None:
                why = "no option"
            row = {"answer": answer or "", "correct": int(answer == x["label"]), "why": why or "accepted",
                   "tokens": tokens, "prompt": prompt_n, **x,
                   "read_s": ask.last.get("prompt_ms", 0) / 1000, "write_s": ask.last.get("predicted_ms", 0) / 1000}
            print(f"    {x['id']} {x['language']}: read {prompt_n} tok in {row['read_s']:.1f} s, wrote {tokens} tok in "
                  f"{row['write_s']:.1f} s, total {row['read_s'] + row['write_s']:.1f} s, {row['why']}"
                  + (", RIGHT" if row["correct"] else ""), flush=True)
            rows.append(row)
            writer.writerow([x["id"], x["language"], x["level"], x.get("text_only"), x["label"], row["answer"],
                             row["correct"], row["why"], output[:60], prompt_n, tokens, code])
            out.flush()
            if n % 20 == 0 or n == len(problems):
                done = [r for r in rows if r["answer"]]
                print(f"  [{n}/{len(problems)}] answered {len(done)}, right {sum(r['correct'] for r in done)}", flush=True)

    answered = [r for r in rows if r["answer"]]
    right = sum(r["correct"] for r in answered)
    tokens = sorted(r["tokens"] for r in rows)
    print(f"\nCoverage:  {len(answered)}/{len(rows)} = {100 * len(answered) / len(rows):.0f}% get an answer")
    if answered:
        print(f"Precision: {right}/{len(answered)} = {100 * right / len(answered):.0f}% of those are right")
    print(f"Overall if the rest were guessed at 20%: {100 * (right + 0.2 * (len(rows) - len(answered))) / len(rows):.0f}%")
    print(f"Outcomes:  {dict(Counter(r['why'] for r in rows))}")
    print(f"Tokens written: median {tokens[len(tokens) // 2]}, worst {tokens[-1]}; "
          f"prompt tokens median {sorted(r['prompt'] for r in rows)[len(rows) // 2]}")
    total = sorted(r["read_s"] + r["write_s"] for r in rows)
    written = sum(r["tokens"] for r in rows) / max(sum(r["write_s"] for r in rows), 1e-9)
    read = sum(r["prompt"] for r in rows) / max(sum(r["read_s"] for r in rows), 1e-9)
    print(f"Model time: median {total[len(total) // 2]:.1f} s, worst {total[-1]:.1f} s; "
          f"reads {read:.1f} tokens/s, writes {written:.1f} tokens/s")
    for key, title in (("language", "By language"), ("level", "By level"), ("text_only", "Text only")):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        print(title + ": " + "  ".join(
            f"{k}: {sum(r['correct'] for r in v)}/{sum(bool(r['answer']) for r in v)}/{len(v)}"
            for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))) + "   (right/answered/all)")
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
