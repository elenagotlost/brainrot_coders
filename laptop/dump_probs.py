#!/usr/bin/env python3
"""M1: dump the model's probability for each letter A-E, for every problem.

This is the hand-off file between the two laptops: the model track makes it,
the pipeline track fits the letter debias and the fusion weight on it.

    python3 laptop/dump_probs.py --input data/val.jsonl   --model q08-q8
    python3 laptop/dump_probs.py --input data/train.jsonl --model q2b-q4 --url http://127.0.0.1:8091

Writes results/probs-<model>-direct-<split>.csv with the columns
    id, language, level, route, label, pA..pE, letter_mass, argmax, prompt_tokens, seconds, status
route: "options" (text options read) or "empty" (no options: the cascade answers the prior there).
pA..pE are renormalised over the five letters (the same as a softmax under the A-E grammar).
letter_mass: how much of the first-token probability fell on A-E before renormalising. The base
0.8B wants to start an explanation ("To solve...") and gives letters well under 50%, so use it as
a confidence feature, not as an error flag.

Resumable: rows already in the output file are skipped. Standard library only.
"""
import argparse
import csv
import json
import math
import os
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copilot.client import post          # noqa: E402
from copilot.config import CONFIG        # noqa: E402
from copilot.llm import build_messages   # noqa: E402

LETTERS = "ABCDE"
FIELDS = ["id", "language", "level", "route", "label", "pA", "pB", "pC", "pD", "pE",
          "letter_mass", "argmax", "prompt_tokens", "seconds", "status"]


def letter_probs(url, problem, top_k, timeout):
    """One forward pass: the first answer token's top-k log probabilities, folded onto A-E."""
    response = post(url + "/v1/chat/completions", {
        "messages": build_messages(problem),
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "logprobs": True,
        "top_logprobs": top_k,
        "cache_prompt": True,
    }, timeout=timeout)
    probs = dict.fromkeys(LETTERS, 0.0)
    total = 0.0
    content = (response["choices"][0].get("logprobs") or {}).get("content") or []
    for cand in (content[0].get("top_logprobs", []) if content else []):
        p = math.exp(cand["logprob"])
        total += p
        tok = cand["token"].strip().strip("()*")
        if len(tok) == 1 and tok in probs:  # "A", " A", "A)", "(A", "**A" count as A; "a" does not
            probs[tok] += p
    mass = sum(probs.values())
    timings = response.get("timings", {})
    return probs, mass, (mass / total if total else 0.0), timings.get("prompt_n", 0)


def forced_letter(url, problem, timeout):
    """Fallback when no letter shows up in the top-k: the baseline's grammar-forced letter."""
    response = post(url + "/v1/chat/completions", {
        "messages": build_messages(problem), "grammar": "root ::= [A-E]", "max_tokens": 1,
        "temperature": 0, "seed": int(CONFIG["SEED"]), "cache_prompt": True,
    }, timeout=timeout)
    return response["choices"][0]["message"]["content"].strip()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--model", required=True, help="short name used in the file name, e.g. q08-q8, q2b-q4")
    ap.add_argument("--url", default=f"http://127.0.0.1:{CONFIG['LLM_PORT']}")
    ap.add_argument("--out", default="", help="default: results/probs-<model>-direct-<split>.csv")
    ap.add_argument("--top-k", type=int, default=5000,
                    help="how many first-token candidates to read. Keep it large: the base model puts "
                         "under 5%% of its mass on letters, so a letter can sit at rank 100+ (costs ~0 s)")
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--languages", default="", help="e.g. eng,cat (default: all)")
    ap.add_argument("--timeout", type=float, default=120)
    args = ap.parse_args()

    split = os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or f"results/probs-{args.model}-direct-{split}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    langs = set(filter(None, args.languages.split(",")))
    if langs:
        problems = [p for p in problems if p.get("language") in langs]
    if args.limit:
        problems = problems[:args.limit]

    done = set()
    if os.path.exists(out):
        with open(out) as f:
            done = {(r["id"], r["language"]) for r in csv.DictReader(f)}
    new_file = not done
    todo = [p for p in problems if (p["id"], p.get("language", "")) not in done]
    print(f"{len(problems)} problems, {len(done)} already in {out}, {len(todo)} to go", file=sys.stderr)

    hits, n, picks, low_mass = 0, 0, Counter(), 0
    t_start = time.time()
    with open(out, "a", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        if new_file:
            w.writeheader()
        for i, p in enumerate(todo, 1):
            t0 = time.perf_counter()
            row = {"id": p["id"], "language": p.get("language", ""), "level": p.get("level", ""),
                   "route": "options" if p.get("options") else "empty", "label": p.get("label", ""),
                   "status": "ok"}
            try:
                probs, mass, share, prompt_n = letter_probs(args.url, p, args.top_k, args.timeout)
                if mass == 0:  # no letter in the top-k: fall back to the forced letter, one-hot
                    letter = forced_letter(args.url, p, args.timeout)
                    probs = {k: (1.0 if k == letter else 0.0) for k in LETTERS}
                    share, row["status"] = 0.0, "forced"
                else:
                    probs = {k: v / mass for k, v in probs.items()}
                if share < 0.5:
                    low_mass += 1
            except Exception as e:  # keep going; the row records what happened
                probs, share, prompt_n, row["status"] = dict.fromkeys(LETTERS, 0.2), 0.0, 0, f"error: {e}"
            best = max(LETTERS, key=lambda k: probs[k])
            row.update({f"p{k}": f"{probs[k]:.6f}" for k in LETTERS})
            row.update(letter_mass=f"{share:.4f}", argmax=best, prompt_tokens=prompt_n,
                       seconds=f"{time.perf_counter() - t0:.2f}")
            w.writerow(row)
            fh.flush()
            n += 1
            picks[best] += 1
            hits += best == p.get("label")
            if i % 25 == 0 or i == len(todo):
                rate = (time.time() - t_start) / i
                print(f"  [{i}/{len(todo)}] argmax acc {100 * hits / n:.1f}%  "
                      f"{rate:.2f} s/problem  low letter mass {low_mass}", file=sys.stderr)

    if n:
        print(f"\nArgmax accuracy (this run): {hits}/{n} = {100 * hits / n:.1f}%", file=sys.stderr)
        print("Letters picked: " + "  ".join(f"{k} {picks[k]}" for k in LETTERS), file=sys.stderr)
        print(f"Rows with letter mass < 50% of the top-{args.top_k}: {low_mass}", file=sys.stderr)
    print(f"Wrote {out}", file=sys.stderr)


if __name__ == "__main__":
    main()
