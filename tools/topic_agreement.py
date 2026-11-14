#!/usr/bin/env python3
"""Compare human topic labels with the automatic tagger.

    python tools/topic_agreement.py [--labels results/topic_labels_todo.csv] [--data data/v2]

Reads the file written by tools/sample_for_labels.py once its human_topic
column is filled in, and prints:
  - the agreement on the exact topic
  - the agreement on four coarse families (number, geometry, spatial,
    logic-combinatorics), which is what reports and routing should use until
    the exact agreement reaches 80 %
  - the confusion matrix (rows: human, columns: tagger)
  - the ids where they disagree (ids only)

Standard library only.
"""
import argparse
import csv
import json
import os
import sys
from collections import Counter

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.topics import TOPICS  # noqa: E402

TOPIC_NAMES = sorted(TOPICS) + ["other"]
FAMILY = {
    "arithmetic": "number", "number_theory": "number", "algebra": "number",
    "sequences_patterns": "number", "measurement_time": "number", "other": "number",
    "plane_geometry": "geometry", "solid_geometry": "geometry",
    "spatial_visual": "spatial", "grids_tables": "spatial", "graphs_paths": "spatial",
    "logic": "logic-combinatorics", "combinatorics": "logic-combinatorics",
    "probability": "logic-combinatorics",
}


def matrix(pairs, names):
    counts = Counter(pairs)
    used = [n for n in names if any(n in pair for pair in counts)]
    width = max(len(n) for n in used) + 1
    print(f"  {'human / tagger':{width}}" + "".join(f"{n[:9]:>10}" for n in used))
    for human in used:
        print(f"  {human:{width}}" + "".join(f"{counts[human, tagger] or '.':>10}" for tagger in used))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--labels", default="results/topic_labels_todo.csv")
    p.add_argument("--data", default="data/v2")
    args = p.parse_args()

    tagger = {}
    for split in ("train", "val"):
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    tagger[row["id"]] = row["topic"]

    with open(args.labels, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    labelled, unknown_ids, bad_names = [], [], Counter()
    for r in rows:
        human = (r.get("human_topic") or "").strip().lower().replace(" ", "_")
        if not human:
            continue
        if r["id"] not in tagger:
            unknown_ids.append(r["id"])
        elif human not in FAMILY:
            bad_names[human] += 1
        else:
            labelled.append((r["id"], human, tagger[r["id"]]))
    print(f"{args.labels}: {len(rows)} ids, {len(labelled)} labelled")
    if unknown_ids:
        print(f"WARNING: {len(unknown_ids)} ids are not in {args.data}: {', '.join(unknown_ids[:5])}")
    if bad_names:
        print("WARNING: unknown topic names, skipped: " + ", ".join(f"{k} ({v})" for k, v in bad_names.items()))
        print("  valid names: " + ", ".join(TOPIC_NAMES))
    if not labelled:
        sys.exit("Nothing to compare yet: fill in the human_topic column first.")

    exact = sum(h == t for _, h, t in labelled)
    coarse = sum(FAMILY[h] == FAMILY[t] for _, h, t in labelled)
    n = len(labelled)
    print(f"\nExact topic agreement:   {exact}/{n} = {100 * exact / n:.1f}%")
    print(f"Coarse family agreement: {coarse}/{n} = {100 * coarse / n:.1f}%")
    print("  (use the exact topics only once the exact agreement is 80% or more)")

    print("\nConfusion matrix, exact topics")
    matrix([(h, t) for _, h, t in labelled], TOPIC_NAMES)
    print("\nConfusion matrix, coarse families")
    matrix([(FAMILY[h], FAMILY[t]) for _, h, t in labelled], sorted(set(FAMILY.values())))

    wrong = [(pid, h, t) for pid, h, t in labelled if h != t]
    if wrong:
        print("\nDisagreements (id: human, tagger)")
        for pid, h, t in wrong:
            print(f"  {pid}: {h}, {t}")


if __name__ == "__main__":
    main()
