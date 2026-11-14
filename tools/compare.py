#!/usr/bin/env python3
"""Compare two eval.py runs problem by problem.

    python tools/compare.py results/eval-A.csv results/eval-B.csv [--by route,level,language,topic]

Rows are joined on (id, language). The report has:
  - accuracy of A and B, the difference, and the exact two-sided McNemar
    p-value on the discordant pairs (b: only A right, c: only B right)
  - the same per problem id: an id counts 1 when most of its languages are
    right, 0.5 on a tie, 0 otherwise. eng and cat are the same problems, so
    this is the honest headline (n = 281 on the full validation set)
  - accuracy tables by each --by column

Standard library only.
"""
import argparse
import csv
import math
import sys
from collections import Counter, defaultdict


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{path} has no rows.")
    for needed in ("id", "language", "correct"):
        if needed not in rows[0]:
            sys.exit(f"{path} has no '{needed}' column: is it an eval.py CSV?")
    table = {}
    for r in rows:
        key = (r["id"], r["language"])
        if key in table:
            sys.exit(f"{path} has {key} twice.")
        r["correct"] = int(r["correct"])
        table[key] = r
    return table


def mcnemar(b, c):
    """Exact two-sided p-value: a binomial test on the b + c discordant pairs."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
    return min(1.0, 2 * tail / 2 ** n)


def id_score(flags):
    """1 when most languages of the id are right, 0.5 on a tie, 0 otherwise."""
    share = sum(flags) / len(flags)
    return 1.0 if share > 0.5 else 0.5 if share == 0.5 else 0.0


def pct(x):
    return f"{100 * x:5.1f}%"


def headline(title, n, acc_a, acc_b, b, c, note=""):
    print(f"\n{title}")
    print(f"  n = {n}   A {pct(acc_a)}   B {pct(acc_b)}   B - A {100 * (acc_b - acc_a):+5.1f} pp")
    print(f"  only A right: {b}   only B right: {c}   McNemar exact p = {mcnemar(b, c):.4f}{note}")


def breakdown(title, keys, a, b, value_of):
    groups = defaultdict(list)
    for key in keys:
        groups[value_of(key)].append(key)
    print(f"\n{title}")
    print(f"  {'':18} {'n':>5} {'A':>7} {'B':>7} {'B - A':>8}")
    for value in sorted(groups, key=str):
        ks = groups[value]
        acc_a = sum(a[k]["correct"] for k in ks) / len(ks)
        acc_b = sum(b[k]["correct"] for k in ks) / len(ks)
        label = value if value != "" else "(none)"
        print(f"  {label:18} {len(ks):>5} {pct(acc_a):>7} {pct(acc_b):>7} {100 * (acc_b - acc_a):+7.1f}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("a", help="eval CSV of run A")
    p.add_argument("b", help="eval CSV of run B")
    p.add_argument("--by", default="route,level,language,topic", help="columns for the breakdown tables")
    args = p.parse_args()

    a, b = load(args.a), load(args.b)
    print(f"A: {args.a}\nB: {args.b}")

    keys = sorted(set(a) & set(b))
    only_a, only_b = len(set(a) - set(b)), len(set(b) - set(a))
    if only_a or only_b:
        print(f"\nWARNING: the runs don't cover the same (id, language) set: {only_a} rows only in A, "
              f"{only_b} only in B. Comparing the {len(keys)} shared rows.")
    if not keys:
        sys.exit("The two runs share no (id, language) rows.")
    labels = sum(1 for k in keys if a[k].get("label") != b[k].get("label"))
    if labels:
        print(f"\nWARNING: {labels} shared rows have a different label in A and B.")

    acc_a = sum(a[k]["correct"] for k in keys) / len(keys)
    acc_b = sum(b[k]["correct"] for k in keys) / len(keys)
    disc_b = sum(1 for k in keys if a[k]["correct"] and not b[k]["correct"])
    disc_c = sum(1 for k in keys if b[k]["correct"] and not a[k]["correct"])
    headline("Per row (id, language)", len(keys), acc_a, acc_b, disc_b, disc_c)

    by_id = defaultdict(list)
    for k in keys:
        by_id[k[0]].append(k)
    score_a = {i: id_score([a[k]["correct"] for k in ks]) for i, ks in by_id.items()}
    score_b = {i: id_score([b[k]["correct"] for k in ks]) for i, ks in by_id.items()}
    ids = sorted(by_id)
    headline("Per problem id (majority over its languages, ties count 0.5): the honest headline",
             len(ids), sum(score_a.values()) / len(ids), sum(score_b.values()) / len(ids),
             sum(1 for i in ids if score_a[i] > score_b[i]),
             sum(1 for i in ids if score_b[i] > score_a[i]),
             note="   (discordant: ids where one run scores higher)")
    per_id = Counter(len(ks) for ks in by_id.values())
    print("  languages per id: " + ", ".join(f"{n} lang: {c} ids" for n, c in sorted(per_id.items())))

    for column in filter(None, (c.strip() for c in args.by.split(","))):
        in_a = any(a[k].get(column) for k in keys)
        in_b = any(b[k].get(column) for k in keys)
        if not in_a and not in_b:
            print(f"\nBy {column}: no values in either run, skipped.")
            continue
        same = all((a[k].get(column) or "") == (b[k].get(column) or "") for k in keys)
        if same:
            breakdown(f"By {column}", keys, a, b, lambda k: a[k].get(column) or "")
        else:  # e.g. route: the two runs took different paths
            breakdown(f"By {column} of run A", keys, a, b, lambda k: a[k].get(column) or "")
            breakdown(f"By {column} of run B", keys, a, b, lambda k: b[k].get(column) or "")


if __name__ == "__main__":
    main()
