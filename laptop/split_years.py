#!/usr/bin/env python3
"""A second, harder split: hold out the most recent years as "new problems".

    python laptop/split_years.py [--in data/v2] [--out data/v2_years] [--val-years 2023,2024]

Joins <in>/train.jsonl and <in>/val.jsonl and writes them again by year:
<out>/train.jsonl has every other year, <out>/val.jsonl has the held-out
years. Rows are not changed. No OCR and no download.

This split is a separate experiment. Fit the priors, the calibration and the
twin bank on its own train.jsonl before scoring its val.jsonl:

    python laptop/fit_priors.py --train data/v2_years/train.jsonl --val data/v2_years/val.jsonl \\
           --out results/priors-years.json
    python tools/fit_calib.py --probs results/probs-<model>-direct-train.csv results/probs-<model>-direct-val.csv \\
           --data data/v2_years/train.jsonl --out results/calib-<model>-years.json
    python eval.py --input data/v2_years/val.jsonl --limit 0 --priors results/priors-years.json ...

Standard library only.
"""
import argparse
import json
import os
import sys
from collections import Counter


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", default="data/v2")
    p.add_argument("--out", default="data/v2_years")
    p.add_argument("--val-years", default="2023,2024", help="years to hold out, e.g. 2023,2024")
    args = p.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.out):
        sys.exit("--out must be a different folder from --in.")
    held_out = {int(y) for y in args.val_years.split(",") if y.strip()}

    rows = []
    for split in ("train", "val"):
        with open(os.path.join(args.src, f"{split}.jsonl"), encoding="utf-8") as f:
            rows += [json.loads(line) for line in f if line.strip()]
    missing = sum(1 for r in rows if not isinstance(r.get("year"), int))
    if missing:
        sys.exit(f"{missing} rows have no year: this split needs it on every row.")

    parts = {"train": [r for r in rows if r["year"] not in held_out],
             "val": [r for r in rows if r["year"] in held_out]}
    if {r["id"] for r in parts["train"]} & {r["id"] for r in parts["val"]}:
        sys.exit("An id has rows in both parts: ids must carry one year.")

    os.makedirs(args.out, exist_ok=True)
    for split, part in parts.items():
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for r in part:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"{args.src} -> {args.out}, held-out years {sorted(held_out)}")
    print(f"\n{'part':6} {'rows':>6} {'ids':>6} {'years':>11} {'with options':>13}   by language")
    for split, part in parts.items():
        years = sorted({r["year"] for r in part})
        langs = Counter(r["language"] for r in part)
        with_options = sum(len(r.get("options") or {}) == 5 for r in part)
        print(f"{split:6} {len(part):>6} {len({r['id'] for r in part}):>6} {years[0]:>5}-{years[-1]} "
              f"{with_options:>13}   " + ", ".join(f"{k} {v}" for k, v in sorted(langs.items())))
    print("\nheld-out rows by level: " + ", ".join(
        f"{level}: {n}" for level, n in sorted(Counter(r["level"] for r in parts["val"]).items())))
    print("held-out labels: " + ", ".join(
        f"{letter} {100 * n / len(parts['val']):.1f}%" for letter, n in
        sorted(Counter(r["label"] for r in parts["val"]).items())))


if __name__ == "__main__":
    main()
