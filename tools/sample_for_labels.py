#!/usr/bin/env python3
"""Pick problems for a human to label by topic, to check the automatic tagger.

    python tools/sample_for_labels.py [--data data/v2] [--n 60] [--out results/topic_labels_todo.csv]

Writes ids only (no problem text) with an empty human_topic column. The ids
are spread over the tagger's topics: every topic gets at least two, the rest
in proportion to its size. The tagger's own topic is NOT in the file, so the
person labelling isn't steered by it. The same ids every run.

Open each id in the explorer, write one topic name in human_topic, then run
tools/topic_agreement.py. Standard library only.
"""
import argparse
import csv
import json
import os
import random
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.topics import TOPICS  # noqa: E402

TOPIC_NAMES = sorted(TOPICS) + ["other"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--data", default="data/v2", help="folder with the v2 train.jsonl and val.jsonl")
    p.add_argument("--n", type=int, default=60)
    p.add_argument("--out", default="results/topic_labels_todo.csv")
    p.add_argument("--force", action="store_true", help="overwrite an existing file (labels in it are lost)")
    args = p.parse_args()

    if os.path.exists(args.out) and not args.force:
        sys.exit(f"{args.out} already exists and may hold labels. Use --force to overwrite it.")

    by_topic = defaultdict(set)
    for split in ("train", "val"):
        with open(os.path.join(args.data, f"{split}.jsonl"), encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    row = json.loads(line)
                    by_topic[row["topic"]].add(row["id"])
    total = sum(len(ids) for ids in by_topic.values())
    if args.n < 2 * len(by_topic):
        sys.exit(f"--n must be at least {2 * len(by_topic)}: two per topic.")

    # two per topic, then the rest in proportion to size (largest remainders first)
    quota = {t: min(2, len(ids)) for t, ids in by_topic.items()}
    left = args.n - sum(quota.values())
    share = {t: left * len(ids) / total for t, ids in by_topic.items()}
    for t in by_topic:
        quota[t] += int(share[t])
    for t in sorted(by_topic, key=lambda t: (-(share[t] - int(share[t])), t))[:args.n - sum(quota.values())]:
        quota[t] += 1

    rng = random.Random(42)
    picked = []
    for topic in sorted(by_topic):
        ids = sorted(by_topic[topic])
        rng.shuffle(ids)
        picked += ids[:quota[topic]]
    rng.shuffle(picked)  # so the file isn't grouped by the tagger's topic

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["id", "human_topic"])
        for pid in picked:
            writer.writerow([pid, ""])

    print(f"{len(picked)} ids written to {args.out}")
    print("ids per tagger topic: " + ", ".join(f"{t} {quota[t]}" for t in sorted(quota)))
    print("\nWrite one of these names in human_topic:")
    print("  " + ", ".join(TOPIC_NAMES))


if __name__ == "__main__":
    main()
