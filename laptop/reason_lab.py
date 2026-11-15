#!/usr/bin/env python3
"""Reasoning lab: how well does a model answer by reasoning in words, for a given token budget?

    python laptop/reason_lab.py --name math15-128 --url http://127.0.0.1:8094/v1/chat/completions --tokens 128
    python laptop/reason_lab.py --name math15-0 --url http://127.0.0.1:8094/v1/chat/completions --tokens 0

--tokens N: the model may write N tokens of working, then it is asked for one letter (forced
by a grammar). --tokens 0 asks for the letter straight away. The tokens written are what
cost time on the board, so try several budgets and compare.

Writes results/reason-<name>.csv (id, language, label, answer, correct, tokens, working).
Standard library only. The counterpart of laptop/code_lab.py for models that reason instead
of writing code.
"""
import argparse
import csv
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copilot.client import post    # noqa: E402

LETTERS = "ABCDE"
SYSTEM = ("You solve multiple-choice math competition problems. Reason step by step, briefly, "
          "then finish with a line 'Answer: X' where X is the letter of the correct option.")
DIRECT = ("You solve multiple-choice math competition problems. "
          "Reply with the letter of the correct option only.")
ASK = "Which option is correct? Reply with one letter."


def question(problem):
    text = re.sub(r"^\s*\d+\.\s*", "", problem["text"]).strip()
    return text + "\n" + "\n".join(f"{k}) {problem['options'][k]}" for k in LETTERS)


def letter_call(url, turns):
    response = post(url, {"messages": turns, "max_tokens": 1, "temperature": 0, "seed": 42,
                          "grammar": "root ::= [A-E]", "cache_prompt": True}, timeout=300)
    return response["choices"][0]["message"]["content"].strip()[:1]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default="data/demo/train.jsonl")
    p.add_argument("--name", required=True)
    p.add_argument("--url", required=True)
    p.add_argument("--tokens", type=int, default=128)
    p.add_argument("--limit", type=int, default=0)
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [x for x in problems if len(x.get("options") or {}) == 5]
    if args.limit:
        problems = problems[:args.limit]
    os.makedirs("results", exist_ok=True)
    rows = []
    with open(f"results/reason-{args.name}.csv", "w", newline="", encoding="utf-8") as out:
        writer = csv.writer(out)
        writer.writerow(["id", "language", "level", "text_only", "label", "answer", "correct", "finished",
                         "prompt_tokens", "tokens", "working"])
        for n, x in enumerate(problems, 1):
            working, tokens, prompt_n, finished = "", 0, 0, False
            if args.tokens:
                turns = [{"role": "system", "content": SYSTEM}, {"role": "user", "content": question(x)}]
                response = post(args.url, {"messages": turns, "max_tokens": args.tokens, "temperature": 0,
                                           "seed": 42, "cache_prompt": True}, timeout=600)
                working = response["choices"][0]["message"]["content"]
                tokens = response.get("timings", {}).get("predicted_n", 0)
                prompt_n = response.get("timings", {}).get("prompt_n", 0)
                found = re.findall(r"Answer:\s*\(?([A-E])\b", working)
                finished = bool(found)
                answer = found[-1] if found else letter_call(
                    args.url, turns + [{"role": "assistant", "content": working}, {"role": "user", "content": ASK}])
            else:
                answer = letter_call(args.url, [{"role": "system", "content": DIRECT},
                                                {"role": "user", "content": question(x)}])
            correct = int(answer == x["label"])
            rows.append({**x, "correct": correct, "tokens": tokens, "finished": finished})
            writer.writerow([x["id"], x["language"], x["level"], x.get("text_only"), x["label"], answer, correct,
                             int(finished), prompt_n, tokens, working])
            out.flush()
            if n % 25 == 0 or n == len(problems):
                print(f"  [{n}/{len(problems)}] right {sum(r['correct'] for r in rows)}", flush=True)

    right = sum(r["correct"] for r in rows)
    tokens = sorted(r["tokens"] for r in rows)
    print(f"\n{args.name}: {right}/{len(rows)} right; finished its working within the budget: "
          f"{sum(r['finished'] for r in rows)}; tokens written: median {tokens[len(tokens) // 2]}, worst {tokens[-1]}")
    for key in ("language", "text_only"):
        groups = {}
        for r in rows:
            groups.setdefault(r[key], []).append(r["correct"])
        print(f"By {key}: " + "  ".join(f"{k}: {sum(v)}/{len(v)}" for k, v in sorted(groups.items(), key=str)))


if __name__ == "__main__":
    main()
