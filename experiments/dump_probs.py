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

M2, reasoning budget: let the model write its working for at most N tokens, then force a letter.

    python3 laptop/dump_probs.py --input data/train.jsonl --model q2b-q4 --mode reason \
        --budgets 0,64,128,192 --sample 200 --url http://127.0.0.1:8092

The working is generated once at the largest budget (temperature 0, so a smaller budget is
just a prefix of it). For each budget the working is cut there, "Answer:" is added, and the
next-token probabilities are read. Writes one file per budget,
results/probs-<model>-reason<N>-<split>.csv, with two more columns: reason_tokens (tokens of
working actually used) and finished (1 if the model reached its own "Answer:" within the budget).
The working itself goes to results/reason-<model>-<split>.jsonl (M3Kang text: never commit).

Resumable: rows already in the output file are skipped. Standard library only.
"""
import argparse
import csv
import json
import math
import os
import random
import sys
import time
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copilot.client import post          # noqa: E402
from copilot.config import CONFIG        # noqa: E402
from copilot.llm import build_messages, format_problem   # noqa: E402

LETTERS = "ABCDE"
FIELDS = ["id", "language", "level", "route", "label", "pA", "pB", "pC", "pD", "pE",
          "letter_mass", "argmax", "prompt_tokens", "seconds", "status"]


REASON_SYSTEM = ("You solve multiple-choice math competition problems for school students. "
                 "Reason step by step, very briefly, then finish with a line 'Answer: X' "
                 "where X is the letter of the correct option.")


def reason_messages(problem):
    return [{"role": "system", "content": REASON_SYSTEM},
            {"role": "user", "content": format_problem(problem)}]


def generate_working(url, problem, max_tokens, timeout):
    """The model's working as a list of token texts, stopped just before its own 'Answer:'."""
    t0 = time.perf_counter()
    response = post(url + "/v1/chat/completions", {
        "messages": reason_messages(problem), "max_tokens": max_tokens, "stop": ["Answer:"],
        "temperature": 0, "seed": int(CONFIG["SEED"]), "logprobs": True, "cache_prompt": True,
    }, timeout=timeout)
    choice = response["choices"][0]
    tokens = [t["token"] for t in (choice.get("logprobs") or {}).get("content") or []]
    return tokens, choice.get("finish_reason") == "stop", time.perf_counter() - t0


def letter_probs(url, problem, top_k, timeout, messages=None):
    """One forward pass: the first answer token's top-k log probabilities, folded onto A-E."""
    response = post(url + "/v1/chat/completions", {
        "messages": messages or build_messages(problem),
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


def sample_ids(problems, n, seed=42):
    """n different ids with options; language i % (number of languages), so no id counts twice."""
    by_id = {}
    for p in problems:
        if p.get("options"):
            by_id.setdefault(p["id"], {})[p.get("language", "")] = p
    ids = sorted(by_id)
    random.Random(seed).shuffle(ids)
    languages = sorted({lang for v in by_id.values() for lang in v})
    picked = []
    for pid in ids:
        want = languages[len(picked) % len(languages)]
        if want in by_id[pid]:
            picked.append(by_id[pid][want])
        if len(picked) == n:
            break
    return picked


def run_reason(args, problems, split):
    budgets = sorted({int(b) for b in args.budgets.split(",")})
    fields = FIELDS + ["reason_tokens", "finished"]
    paths = {b: f"results/probs-{args.model}-reason{b}-{split}.csv" for b in budgets}
    text_path = f"results/reason-{args.model}-{split}.jsonl"
    os.makedirs("results", exist_ok=True)
    done = None
    for path in paths.values():
        rows = set()
        if os.path.exists(path):
            with open(path) as f:
                rows = {(r["id"], r["language"]) for r in csv.DictReader(f)}
        done = rows if done is None else done & rows
    if not done:  # start clean, so the files of all budgets always hold the same rows
        for path in paths.values():
            with open(path, "w", newline="") as f:
                csv.DictWriter(f, fieldnames=fields).writeheader()
    todo = [p for p in problems if (p["id"], p.get("language", "")) not in done]
    print(f"{len(problems)} problems, {len(done)} already done, {len(todo)} to go; "
          f"budgets {budgets}", file=sys.stderr)

    hits, used, n = Counter(), Counter(), 0
    files = {b: open(path, "a", newline="") for b, path in paths.items()}
    writers = {b: csv.DictWriter(fh, fieldnames=fields) for b, fh in files.items()}
    with open(text_path, "a") as text_out:
        for i, p in enumerate(todo, 1):
            base = {"id": p["id"], "language": p.get("language", ""), "level": p.get("level", ""),
                    "route": "options" if p.get("options") else "empty", "label": p.get("label", "")}
            try:
                tokens, stopped, gen_seconds = generate_working(args.url, p, max(budgets), args.timeout)
                error = ""
            except Exception as e:
                tokens, stopped, gen_seconds, error = [], False, 0.0, f"error: {e}"
            text_out.write(json.dumps({**base, "finished": stopped, "tokens": tokens},
                                      ensure_ascii=False) + "\n")
            text_out.flush()
            for b in budgets:
                t0 = time.perf_counter()
                working = "".join(tokens[:b]).rstrip()
                prefill = (working + "\n" if working else "") + "Answer:"
                row = dict(base, status=error or "ok", reason_tokens=min(b, len(tokens)),
                           finished=int(stopped and len(tokens) <= b))
                probs, share, prompt_n = dict.fromkeys(LETTERS, 0.2), 0.0, 0
                if not error:
                    try:
                        messages = reason_messages(p) + [{"role": "assistant", "content": prefill}]
                        probs, mass, share, prompt_n = letter_probs(args.url, p, args.top_k,
                                                                    args.timeout, messages)
                        if mass == 0:
                            probs, row["status"] = dict.fromkeys(LETTERS, 0.2), "no-letter"
                        else:
                            probs = {k: v / mass for k, v in probs.items()}
                    except Exception as e:
                        row["status"] = f"error: {e}"
                best = max(LETTERS, key=lambda k: probs[k])
                # seconds: the share of the generation this budget used, plus the letter call
                share_of_gen = min(b, len(tokens)) / len(tokens) if tokens else 0.0
                row.update({f"p{k}": f"{probs[k]:.6f}" for k in LETTERS})
                row.update(letter_mass=f"{share:.4f}", argmax=best, prompt_tokens=prompt_n,
                           seconds=f"{gen_seconds * share_of_gen + time.perf_counter() - t0:.2f}")
                writers[b].writerow(row)
                files[b].flush()
                hits[b] += best == p.get("label")
                used[b] += row["reason_tokens"]
            n += 1
            if i % 25 == 0 or i == len(todo):
                print(f"  [{i}/{len(todo)}] " + "  ".join(
                    f"b{b}: {100 * hits[b] / n:.1f}%" for b in budgets), file=sys.stderr)
    for fh in files.values():
        fh.close()
    if n:
        print("\nbudget  accuracy  mean tokens used   (this run, n = %d)" % n, file=sys.stderr)
        for b in budgets:
            print(f"{b:6}  {100 * hits[b] / n:7.1f}%  {used[b] / n:8.1f}", file=sys.stderr)
    print("Wrote " + ", ".join(paths.values()), file=sys.stderr)


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
    ap.add_argument("--mode", default="direct", choices=["direct", "reason"])
    ap.add_argument("--budgets", default="0,64,128,192", help="reason mode: token budgets to measure")
    ap.add_argument("--sample", type=int, default=0,
                    help="take N different problem ids that have options, alternating languages "
                         "(same sample every run)")
    args = ap.parse_args()

    split = os.path.splitext(os.path.basename(args.input))[0]
    out = args.out or f"results/probs-{args.model}-direct-{split}.csv"
    os.makedirs(os.path.dirname(out) or ".", exist_ok=True)

    with open(args.input) as f:
        problems = [json.loads(line) for line in f if line.strip()]
    langs = set(filter(None, args.languages.split(",")))
    if langs:
        problems = [p for p in problems if p.get("language") in langs]
    if args.sample:
        problems = sample_ids(problems, args.sample)
    if args.limit:
        problems = problems[:args.limit]
    if args.mode == "reason":
        return run_reason(args, problems, split)

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
