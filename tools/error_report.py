#!/usr/bin/env python3
"""Turn one eval.py run into markdown tables for the write-up.

    python tools/error_report.py results/eval-<time>-<name>.csv [--data data/v2/val.jsonl] [--out report.md]

Prints the accuracy overall, per problem id, and by route, kind of options,
quality of options, topic, level and language, plus the letter confusion
matrix (rows: the true letter, columns: the answer). --data fills in the
problem fields that the run's CSV doesn't have (options_quality, or topic
when the run was on v1 data). The tables hold ids-free numbers only, so they
can go in the README. Standard library only.
"""
import argparse
import csv
import json
import sys
from collections import defaultdict

LETTERS = "ABCDE"
FROM_DATA = ("topic", "options_kind", "options_quality", "options_source", "year")


def id_score(flags):
    share = sum(flags) / len(flags)
    return 1.0 if share > 0.5 else 0.5 if share == 0.5 else 0.0


def table(out, title, rows, key, order=None):
    groups = defaultdict(list)
    for r in rows:
        value = r.get(key)
        groups["(none)" if value in (None, "") else str(value)].append(r["correct"])
    if set(groups) == {"(none)"}:
        return
    names = sorted(groups, key=order or (lambda v: (v == "(none)", v)))
    out.append(f"\n**By {title}**\n")
    out.append(f"| {title.capitalize()} | n | Correct | Accuracy |")
    out.append("|---|---|---|---|")
    for name in names:
        hits = groups[name]
        out.append(f"| {name} | {len(hits)} | {sum(hits)} | {100 * sum(hits) / len(hits):.1f} % |")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("eval_csv")
    p.add_argument("--data", default="", help="the jsonl the run was made on, to fill in problem fields")
    p.add_argument("--out", default="", help="also write the report to this file")
    args = p.parse_args()

    with open(args.eval_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit(f"{args.eval_csv} has no rows.")
    for r in rows:
        r["correct"] = int(r["correct"])

    if args.data:
        data = {}
        with open(args.data, encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    q = json.loads(line)
                    data[q["id"], q["language"]] = q
        missing = 0
        for r in rows:
            q = data.get((r["id"], r["language"]))
            if q is None:
                missing += 1
                continue
            for field in FROM_DATA:
                if not r.get(field) and q.get(field) not in (None, ""):
                    r[field] = q[field]
        if missing:
            print(f"WARNING: {missing} rows of the run are not in {args.data}", file=sys.stderr)

    n = len(rows)
    hits = sum(r["correct"] for r in rows)
    by_id = defaultdict(list)
    for r in rows:
        by_id[r["id"]].append(r["correct"])
    id_total = sum(id_score(flags) for flags in by_id.values())
    always = {k: sum(r["label"] == k for r in rows) for k in LETTERS}
    best = max(LETTERS, key=lambda k: always[k])

    out = [f"### {args.eval_csv.replace(chr(92), '/').split('/')[-1]}", ""]
    out.append("| Measure | Value |")
    out.append("|---|---|")
    out.append(f"| Accuracy per row | {hits}/{n} = {100 * hits / n:.1f} % |")
    out.append(f"| Accuracy per problem id (majority over its languages) | "
               f"{id_total:g}/{len(by_id)} = {100 * id_total / len(by_id):.1f} % |")
    out.append(f"| Always {best} on these rows | {100 * always[best] / n:.1f} % |")
    out.append("| Random guess | 20.0 % |")

    table(out, "route", rows, "route")
    table(out, "kind of options", rows, "options_kind")
    table(out, "quality of options", rows, "options_quality")
    table(out, "topic", rows, "topic")
    table(out, "level", rows, "level", order=lambda v: (v == "(none)", int(v) if v.isdigit() else 99))
    table(out, "language", rows, "language")

    out.append("\n**Letters: true answer (rows) against the copilot's answer (columns)**\n")
    out.append("| True | " + " | ".join(LETTERS) + " | n | Recall |")
    out.append("|---|" + "---|" * 7)
    for true in LETTERS:
        line = [sum(1 for r in rows if r["label"] == true and r["answer"] == a) for a in LETTERS]
        total = sum(line)
        recall = f"{100 * line[LETTERS.index(true)] / total:.1f} %" if total else "n/a"
        out.append(f"| {true} | " + " | ".join(str(x) for x in line) + f" | {total} | {recall} |")
    answered = [sum(1 for r in rows if r["answer"] == a) for a in LETTERS]
    out.append("| **Answered** | " + " | ".join(str(x) for x in answered) + f" | {n} | |")

    report = "\n".join(out)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8", newline="\n") as f:
            f.write(report + "\n")


if __name__ == "__main__":
    main()
