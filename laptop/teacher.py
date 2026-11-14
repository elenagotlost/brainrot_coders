#!/usr/bin/env python3
"""M3: have a larger local model write short solutions for the TRAIN problems.

Two modes, both local (nothing from M3Kang leaves the machine):

  explain  the teacher is told the correct answer and writes the short solution that leads
           to it. One sample, temperature 0. Every finished sample ends at the right letter.
  solve    the teacher solves blind, --samples times at temperature 0.7. Only samples that
           reach the M3Kang label are worth keeping (filter on "correct" later).

    bash laptop/serve.sh is NOT used for the teacher. Start it like this (4 slots):
    ~/llama.cpp/build/bin/llama-server -m ~/models/Qwen_Qwen3.5-9B-Q4_K_M.gguf --port 8093 \\
        --ctx-size 16384 --parallel 4 --cache-ram 0 --jinja
    python3 laptop/teacher.py --mode explain

Refuses any input file that is not called train.jsonl: val ids must never reach the fine-tune.
Appends to results/teacher-<mode>.jsonl (M3Kang text: never commit). Resumable. The problems
are taken in a fixed random order, so a run stopped half-way is still a fair sample.
"""
import argparse
import json
import os
import random
import re
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copilot.client import post          # noqa: E402
from copilot.llm import format_problem   # noqa: E402

SOLVE = ("You are a maths teacher. Solve the multiple-choice problem. Write only the key steps "
         "and calculations, in at most 60 words, no introduction. Finish with a line 'Answer: X' "
         "where X is the letter of the correct option.")
EXPLAIN = ("You are a maths teacher writing model solutions. You are given a multiple-choice "
           "problem and its correct answer. Write the solution a strong student would write: only "
           "the key steps and calculations that lead to that answer, in at most 60 words, no "
           "introduction, and never mention that you were told the answer. Finish with a line "
           "'Answer: X'.")
ANSWER = re.compile(r"Answer:\s*\**\(?([A-E])\b")


def ask(url, mode, problem, sample, max_tokens):
    user = format_problem(problem)
    if mode == "explain":
        user += f"\n\nCorrect answer: {problem['label']}) {problem['options'][problem['label']]}"
    response = post(url + "/v1/chat/completions", {
        "messages": [{"role": "system", "content": EXPLAIN if mode == "explain" else SOLVE},
                     {"role": "user", "content": user}],
        "max_tokens": max_tokens, "temperature": 0 if mode == "explain" else 0.7,
        "seed": 42 + sample, "chat_template_kwargs": {"enable_thinking": False},
    }, timeout=1200)
    text = response["choices"][0]["message"]["content"]
    found = ANSWER.findall(text)
    return {"id": problem["id"], "language": problem["language"], "label": problem["label"],
            "mode": mode, "sample": sample, "answer": found[-1] if found else "",
            "correct": bool(found) and found[-1] == problem["label"],
            "tokens": response["usage"]["completion_tokens"],
            "finished": response["choices"][0]["finish_reason"] == "stop", "text": text}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default="data/train.jsonl")
    ap.add_argument("--mode", required=True, choices=["explain", "solve"])
    ap.add_argument("--url", default="http://127.0.0.1:8093")
    ap.add_argument("--samples", type=int, default=0, help="default: 1 for explain, 4 for solve")
    ap.add_argument("--max-tokens", type=int, default=260)
    ap.add_argument("--parallel", type=int, default=4, help="the server's --parallel")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--include-low", action="store_true", help="also rows with options_quality low")
    args = ap.parse_args()
    if os.path.basename(args.input) != "train.jsonl":
        sys.exit("teacher.py only reads train.jsonl: val ids must never reach the fine-tune")
    samples = args.samples or (1 if args.mode == "explain" else 4)

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [p for p in problems if p.get("options") and p["label"] in p["options"]
                and (args.include_low or p.get("options_quality", "ok") != "low")]
    ids = sorted({p["id"] for p in problems})
    random.Random(42).shuffle(ids)
    order = {pid: i for i, pid in enumerate(ids)}
    problems.sort(key=lambda p: (order[p["id"]], p["language"]))
    if args.limit:
        problems = problems[:args.limit]

    out = f"results/teacher-{args.mode}.jsonl"
    done = set()
    if os.path.exists(out):
        with open(out, encoding="utf-8") as f:
            done = {(r["id"], r["language"], r["sample"]) for r in map(json.loads, f)}
    jobs = [(p, s) for p in problems for s in range(samples) if (p["id"], p["language"], s) not in done]
    print(f"{len(problems)} problems x {samples} samples, {len(done)} already done, {len(jobs)} to go",
          file=sys.stderr)

    lock, stats, t0 = threading.Lock(), {"n": 0, "finished": 0, "correct": 0, "tokens": 0}, time.time()
    with open(out, "a", encoding="utf-8") as fh:
        def work(job):
            problem, sample = job
            try:
                row = ask(args.url, args.mode, problem, sample, args.max_tokens)
            except Exception as e:
                print(f"  {problem['id']} {problem['language']}: {e}", file=sys.stderr)
                return
            with lock:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
                fh.flush()
                stats["n"] += 1
                stats["finished"] += bool(row["answer"])
                stats["correct"] += row["correct"]
                stats["tokens"] += row["tokens"]
                if stats["n"] % 50 == 0 or stats["n"] == len(jobs):
                    rate = (time.time() - t0) / stats["n"]
                    print(f"  [{stats['n']}/{len(jobs)}] answered {stats['finished']}, correct "
                          f"{stats['correct']}, {rate:.1f} s each, about "
                          f"{rate * (len(jobs) - stats['n']) / 60:.0f} min left", file=sys.stderr)
        with ThreadPoolExecutor(args.parallel) as pool:
            list(pool.map(work, jobs))
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
