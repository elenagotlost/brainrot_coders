#!/usr/bin/env python3
"""Fit the no-model prior tables on the training split only.

    python laptop/fit_priors.py [--train data/v2/train.jsonl] [--val data/v2/val.jsonl]
                                [--out results/priors.json]

Writes results/priors.json with three tables:
  rank_prior    P(correct | rank of the value among the sorted numeric options)
  empty_prior   P(letter) for problems without options
  letter_prior  P(letter) for all problems
copilot/priors.py reads that file when it is imported.

It also prints the accuracy of the prior on the validation split with the new
tables next to the first tables, which were fitted on train + val together.
Validation rows are never used for fitting. Standard library only.
"""
import argparse
import json
import os
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot import priors  # noqa: E402
from copilot.priors import LETTERS, option_values  # noqa: E402

# The first tables, fitted on all 1,747 English problems (train + val): kept
# only to show how little the train-only fit changes.
ALL_DATA = {
    "rank_prior": [0.084, 0.221, 0.311, 0.277, 0.108],
    "empty_prior": {"A": 0.164, "B": 0.178, "C": 0.212, "D": 0.274, "E": 0.172},
    "letter_prior": {"A": 0.160, "B": 0.199, "C": 0.218, "D": 0.247, "E": 0.176},
}


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def shares(counter, keys):
    total = sum(counter[k] for k in keys)
    return [round(counter[k] / total, 4) for k in keys]


def fit(rows):
    rank, empty, letter = Counter(), Counter(), Counter()
    n = Counter()
    for r in rows:
        options = r.get("options") or {}
        letter[r["label"]] += 1
        if len(options) != 5:
            empty[r["label"]] += 1
            n["empty"] += 1
            continue
        values = option_values(options)
        if values:
            order = sorted(range(5), key=lambda i: values[i])
            rank[order.index(LETTERS.index(r["label"]))] += 1
            n["numeric"] += 1
        else:
            n["text"] += 1
    return {
        "rank_prior": shares(rank, range(5)),
        "empty_prior": dict(zip(LETTERS, shares(empty, LETTERS))),
        "letter_prior": dict(zip(LETTERS, shares(letter, LETTERS))),
        "n_train": len(rows),
        "n_by_kind": dict(n),
    }


def use(tables):
    priors.RANK_PRIOR = list(tables["rank_prior"])
    priors.EMPTY_PRIOR = dict(tables["empty_prior"])
    priors.LETTER_PRIOR = dict(tables["letter_prior"])


def accuracy(rows):
    groups = defaultdict(list)
    for r in rows:
        letter, kind = priors.best_letter(r)
        groups[kind].append(letter == r["label"])
        groups["all"].append(letter == r["label"])
    return groups


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train", default="data/v2/train.jsonl")
    p.add_argument("--val", default="data/v2/val.jsonl")
    p.add_argument("--out", default="results/priors.json")
    args = p.parse_args()

    train, val = load(args.train), load(args.val)
    overlap = {r["id"] for r in train} & {r["id"] for r in val}
    if overlap:
        sys.exit(f"{len(overlap)} ids are in both files: the fit would see validation problems.")

    tables = fit(train)
    tables["fitted_on"] = args.train.replace("\\", "/")
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        json.dump(tables, f, indent=1)
        f.write("\n")

    print(f"Fitted on {args.train}: {tables['n_train']} rows {tables['n_by_kind']}")
    print(f"\n{'table':14} {'all data (old)':38} train only (new)")
    print(f"{'rank_prior':14} {str(ALL_DATA['rank_prior']):38} {tables['rank_prior']}")
    for name in ("empty_prior", "letter_prior"):
        old = " ".join(f"{k}:{ALL_DATA[name][k]:.3f}" for k in LETTERS)
        new = " ".join(f"{k}:{tables[name][k]:.3f}" for k in LETTERS)
        print(f"{name:14} {old:38} {new}")

    print(f"\nAccuracy of the prior on {args.val} ({len(val)} rows)")
    print(f"  {'kind':10} {'n':>5} {'all-data tables':>17} {'train-only tables':>19}")
    use(ALL_DATA)
    old = accuracy(val)
    use(tables)
    new = accuracy(val)
    for kind in ("numeric", "text", "empty", "all"):
        if kind in new:
            n = len(new[kind])
            print(f"  {kind:10} {n:>5} {sum(old[kind]):>9}  {100 * sum(old[kind]) / n:5.1f}%"
                  f" {sum(new[kind]):>11}  {100 * sum(new[kind]) / n:5.1f}%")
    by_lang = defaultdict(list)
    for r in val:
        by_lang[r["language"]].append(priors.best_letter(r)[0] == r["label"])
    print("  by language, train-only tables: "
          + ", ".join(f"{lang} {100 * sum(h) / len(h):.1f}%" for lang, h in sorted(by_lang.items())))
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
