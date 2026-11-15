#!/usr/bin/env python3
"""Code-stage lab: compare prompt variants of the code stage on a laptop, under the board's budget.

    python laptop/code_lab.py --input data/demo/train.jsonl --variant fewshot --name fs1

Every variant runs through the same runner and the same acceptance rule, and the tokens the
model may write are capped the way copilot/coder.py caps them on the board (57 s, the board's
read and write speeds). For each row it records the simulated board time:
    uncached prompt tokens / READ_TPS + written tokens / WRITE_TPS + program run time.

Writes results/lab-<name>.csv and prints coverage, precision and the worst simulated time.
Standard library only.
"""
import argparse
import csv
import json
import os
import re
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copilot import coder          # noqa: E402
from copilot.client import post    # noqa: E402

LETTERS = "ABCDE"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", required=True)
    p.add_argument("--name", required=True)
    p.add_argument("--url", default="http://127.0.0.1:8091/v1/chat/completions")
    p.add_argument("--languages", default="")
    p.add_argument("--text-only", action="store_true")
    p.add_argument("--limit", type=int, default=0)
    p.add_argument("--seconds", type=float, default=57.0, help="time the code stage may use on the board")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if len(x.get("options") or {}) == 5]
    if args.languages:
        problems = [x for x in problems if x["language"] in args.languages.split(",")]
    if args.text_only:
        problems = [x for x in problems if x.get("text_only")]
    if args.limit:
        problems = problems[:args.limit]

    warm = coder.warm_up(args.url)
    print(f"warm-up {warm:.1f} s", flush=True)
    os.makedirs("results", exist_ok=True)
    out_path = f"results/lab-{args.name}.csv"
    rows = []
    with open(out_path, "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "level", "text_only", "label", "answer", "correct", "why", "output",
                         "prompt_tokens", "tokens", "max_tokens", "board_s", "code"])
        for n, x in enumerate(problems, 1):
            try:
                letter, info = coder.solve(x, args.seconds, url=args.url)
            except Exception as error:  # a timeout or a server error is a miss
                letter, info = None, {"why": type(error).__name__, "code": "", "output": "", "tokens": 0,
                                      "prompt_tokens": 0, "max_tokens": 0}
            board_s = info.get("board_s", 0.0)
            row = {**x, "answer": letter or "", "correct": int(letter == x["label"]), "why": info["why"],
                   "tokens": info.get("tokens", 0), "board_s": board_s}
            rows.append(row)
            writer.writerow([x["id"], x["language"], x["level"], x.get("text_only"), x["label"], row["answer"],
                             row["correct"], info["why"], str(info.get("output", ""))[:60],
                             info.get("prompt_tokens", 0), info.get("tokens", 0), info.get("max_tokens", 0),
                             f"{board_s:.1f}", info.get("code", "")])
            out.flush()
            if n % 25 == 0 or n == len(problems):
                done = [r for r in rows if r["answer"]]
                print(f"  [{n}/{len(problems)}] answered {len(done)}, right {sum(r['correct'] for r in done)}",
                      flush=True)

    by_why = {}
    for r in rows:
        by_why.setdefault(r["why"], []).append(r["correct"])
    print("\nRight/all by outcome: " + "  ".join(f"{k}: {sum(v)}/{len(v)}" for k, v in sorted(by_why.items())))
    answered = [r for r in rows if r["answer"]]
    right = sum(r["correct"] for r in answered)
    wrong = len(answered) - right
    print(f"\n{args.name}: answered {len(answered)}/{len(rows)}, right {right}, wrong {wrong}, "
          f"precision {100 * right / max(len(answered), 1):.0f}%")
    print(f"Expected score if misses are guessed at 20%: "
          f"{100 * (right + 0.2 * (len(rows) - len(answered))) / len(rows):.1f}%")
    print(f"Outcomes: {dict(Counter(r['why'] for r in rows))}")
    board = sorted(r["board_s"] for r in rows)
    tokens = sorted(r["tokens"] for r in rows)
    print(f"Board time (simulated): median {board[len(board) // 2]:.0f} s, worst {board[-1]:.0f} s, "
          f"over 57 s: {sum(b > 57 for b in board)}; tokens written: median {tokens[len(tokens) // 2]}, "
          f"worst {tokens[-1]}")
    for key in ("language", "level", "text_only"):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r)
        print(f"By {key}: " + "  ".join(f"{k}: {sum(r['correct'] for r in v)}/{sum(bool(r['answer']) for r in v)}/{len(v)}"
                                         for k, v in sorted(groups.items(), key=lambda kv: str(kv[0]))))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
