#!/usr/bin/env python3
"""Measure the math-model stage (copilot/tir.py) on training problems, on the laptop.

For each problem the model writes a line or two of reasoning, then a short ```python block.
It is stopped there, the code runs in copilot/sandbox.py, what it printed is pasted back as an
```output block, and the model continues to \\boxed{letter}. The model does the understanding;
Python does the counting. Same code as predict.py's stage 3b, without the time limit.

The answer comes from, in this order:
  boxed   the model's own \\boxed{...} after reading the output
  code    the code printed a value that equals exactly one option (the model ran out of budget)
  forced  "The answer is \\boxed{" is added and the model writes one letter

Start the math model on the laptop with the board's flags (see the CODE_* lines in config.env):

    ~/llama.cpp/build/bin/llama-server -m ~/models/Qwen2.5-Math-1.5B-Instruct-Q4_K_M.gguf \\
        --port 8091 --threads 4 --ctx-size 4096 --parallel 1 --cache-ram 0 --ctx-checkpoints 2 \\
        --jinja --reasoning off --reasoning-budget 0
    python3 laptop/tir.py --model qmath --limit 150
    python3 laptop/tir.py --model qmath --limit 150 --budget 300

--budget is the total number of tokens the model may write for one problem (code included).
Pool: English, text-only, five numeric options, in a fixed order (seed 42), so --limit N is
always the same N problems. Board time uses a GUESS of the board's speed: measure it there.

Writes results/tir-<model>-<budget>.csv (ids and numbers) and .jsonl (the reasoning: generated
from M3Kang, never commit). Standard library only.
"""
import argparse
import csv
import json
import os
import random
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copilot import priors                       # noqa: E402
from copilot.tir import applies, solve           # noqa: E402  (the same code as predict.py)


class Tee:
    """Write to the screen and to a log file at the same time."""

    def __init__(self, stream, path):
        self.stream, self.file = stream, open(path, "w", encoding="utf-8")

    def write(self, text):
        self.stream.write(text)
        self.stream.flush()
        self.file.write(text)
        self.file.flush()

    def flush(self):
        self.stream.flush()
        self.file.flush()


def clock(seconds):
    seconds = int(seconds)
    return f"{seconds // 3600}:{seconds % 3600 // 60:02d}:{seconds % 60:02d}"


def load_pool(path, languages):
    with open(path, encoding="utf-8") as f:
        rows = [json.loads(line) for line in f if line.strip()]
    pool = [r for r in rows if r.get("language") in languages and applies(r) and r.get("text_only")]
    pool.sort(key=lambda r: (r["id"], r["language"]))
    random.Random(42).shuffle(pool)
    return pool

def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", default=os.path.join("data", "v2", "train.jsonl"))
    ap.add_argument("--model", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8091")
    ap.add_argument("--languages", default="eng")
    ap.add_argument("--limit", type=int, default=150)
    ap.add_argument("--budget", type=int, default=200, help="tokens the model may write per problem")
    ap.add_argument("--board-read", type=float, default=7.0, help="tokens/s (guess until measured)")
    ap.add_argument("--board-write", type=float, default=4.5, help="tokens/s (guess until measured)")
    ap.add_argument("--log")
    args = ap.parse_args()
    if args.log:
        sys.stderr = Tee(sys.stderr, args.log)
    if os.path.basename(args.input) != "train.jsonl":
        print("Note: not a train.jsonl. Don't tune prompts on validation problems.", file=sys.stderr)

    url = args.url.rstrip("/")
    pool = load_pool(args.input, set(args.languages.split(",")))
    pool = pool[:args.limit] if args.limit else pool
    print(f"{len(pool)} problems, budget {args.budget} written tokens per problem", file=sys.stderr)

    stem = os.path.join(ROOT, "results", f"tir-{args.model}-{args.budget}")
    os.makedirs(os.path.dirname(stem), exist_ok=True)
    stats, by_source = Counter(), {}
    tokens_out, tokens_in, t_start = [], [], time.time()
    with open(stem + ".csv", "w", newline="", encoding="utf-8") as fc, \
            open(stem + ".jsonl", "w", encoding="utf-8") as fj:
        w = csv.writer(fc)
        w.writerow(["id", "language", "level", "topic", "label", "answer", "source", "correct",
                    "code_letter", "rounds", "code_errors", "looped", "prompt_tokens", "gen_tokens",
                    "seconds", "prior_letter"])
        for i, p in enumerate(pool, 1):
            t0 = time.time()
            try:
                r = solve(url, p, args.budget)
            except Exception as e:  # noqa: BLE001  server down, timeout...
                r = {"letter": "", "source": "request-failed", "code_letter": "", "rounds": 0,
                     "code_errors": 0, "looped": False, "gen_tokens": 0, "prompt_tokens": 0,
                     "text": str(e)[:200]}
            seconds = time.time() - t0
            prior_p, _ = priors.prior(p)
            prior_letter = max(prior_p, key=prior_p.get)
            correct = r["letter"] == p["label"]
            stats["right"] += correct
            stats["prior"] += prior_letter == p["label"]
            stats["code_ran"] += r["rounds"] > 0
            stats["looped"] += r["looped"]
            stats["code_answered"] += bool(r["code_letter"])
            stats["code_right"] += r["code_letter"] == p["label"]
            src = by_source.setdefault(r["source"], Counter())
            src["n"] += 1
            src["right"] += correct
            tokens_out.append(r["gen_tokens"])
            tokens_in.append(r["prompt_tokens"])
            w.writerow([p["id"], p["language"], p.get("level", ""), p.get("topic", ""), p["label"],
                        r["letter"], r["source"], int(correct), r["code_letter"], r["rounds"],
                        r["code_errors"], int(r["looped"]), r["prompt_tokens"], r["gen_tokens"],
                        round(seconds, 1), prior_letter])
            fc.flush()
            fj.write(json.dumps({"id": p["id"], "language": p["language"], "label": p["label"],
                                 "answer": r["letter"], "source": r["source"],
                                 "code_letter": r["code_letter"], "text": r["text"]},
                                ensure_ascii=False) + "\n")
            fj.flush()
            mark = "OK " if correct else ("BAD" if r["letter"] else " - ")
            elapsed = time.time() - t_start
            print(f"[{i}/{len(pool)}] {mark} {p['id']:<16} {r['source']:<7} code x{r['rounds']}"
                  f"{' LOOP' if r['looped'] else ''} {r['gen_tokens']:>4} tok {seconds:>6.1f} s | "
                  f"right so far {stats['right']} (priors {stats['prior']}) | elapsed {clock(elapsed)}, "
                  f"about {clock(elapsed / i * (len(pool) - i))} left", file=sys.stderr)

    n = len(pool) or 1
    written = sorted(tokens_out)
    mean_in, mean_out = sum(tokens_in) / n, sum(tokens_out) / n
    print(f"\n== {args.model} / tool-integrated reasoning, budget {args.budget}: {len(pool)} problems",
          file=sys.stderr)
    print(f"Math model:   {stats['right']}/{len(pool)} right ({100 * stats['right'] / n:.1f}%)",
          file=sys.stderr)
    print(f"Yardstick:    {stats['prior']}/{len(pool)} right with no model (the priors)", file=sys.stderr)
    print("Where the answer came from (n, right):", file=sys.stderr)
    for s, c in sorted(by_source.items(), key=lambda kv: -kv[1]["n"]):
        print(f"  {s:<15} {c['n']:>4} {c['right']:>4}", file=sys.stderr)
    print(f"Code ran on {stats['code_ran']} problems; its printed value matched one option on "
          f"{stats['code_answered']}, right on {stats['code_right']}", file=sys.stderr)
    print(f"Loops: {stats['looped']}", file=sys.stderr)
    print(f"Tokens written: median {written[len(written) // 2]}, mean {mean_out:.0f}; read {mean_in:.0f} "
          f"-> board guess {mean_in / args.board_read + mean_out / args.board_write:.0f} s per problem "
          f"({args.board_read:g} read / {args.board_write:g} write tokens/s)", file=sys.stderr)
    print(f"Laptop: {(time.time() - t_start) / n:.1f} s per problem", file=sys.stderr)
    print(f"Wrote {stem}.csv and {stem}.jsonl", file=sys.stderr)


if __name__ == "__main__":
    main()
