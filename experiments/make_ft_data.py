#!/usr/bin/env python3
"""M4: turn the teacher's solutions into fine-tuning data for mlx-lm.

    python3 laptop/make_ft_data.py --format reason --out ~/ft/data-reason
    python3 laptop/make_ft_data.py --format letter --out ~/ft/data-letter

reason  system prompt of the reasoning mode, the problem, then the teacher's short solution
        ending in "Answer: X". Only solutions that finished, reach the M3Kang label and are
        at most --max-tokens long.
letter  the baseline's prompt (copilot.llm.build_messages), then the letter alone. Every train
        row that has options.

Writes train.jsonl and valid.jsonl (5% of the TRAIN ids, to watch the loss) in --out.
The output holds M3Kang text: keep it outside the repository. Stops if any val id shows up.
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from copilot.llm import build_messages   # noqa: E402
from dump_probs import reason_messages   # noqa: E402


def load(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean(text, label):
    """The solution with exactly one final line 'Answer: X'. None when it looks wrong."""
    body = re.split(r"\n?\**Answer:", text)[0].strip()
    if not body or "correct answer" in body.lower() or "given answer" in body.lower():
        return None  # empty, or it talks about having been told the answer
    return f"{body}\nAnswer: {label}"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--format", required=True, choices=["reason", "letter"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--teacher", default="results/teacher-explain.jsonl")
    ap.add_argument("--max-tokens", type=int, default=160, help="reason: longest solution kept")
    ap.add_argument("--include-low", action="store_true")
    args = ap.parse_args()

    train = {(p["id"], p["language"]): p for p in load("data/train.jsonl")}
    val_ids = {p["id"] for p in load("data/val.jsonl")}
    if val_ids & {pid for pid, _ in train}:
        sys.exit("train and val share ids: stop")

    examples, skipped = [], {}
    if args.format == "reason":
        for row in load(args.teacher):
            key = (row["id"], row["language"])
            why = ("not a train row" if key not in train else
                   "not finished or wrong letter" if not row["correct"] else
                   "too long" if row["tokens"] > args.max_tokens else "")
            solution = None if why else clean(row["text"], row["label"])
            if not why and solution is None:
                why = "mentions the given answer"
            if why:
                skipped[why] = skipped.get(why, 0) + 1
                continue
            examples.append((row["id"], {"messages": reason_messages(train[key]) + [
                {"role": "assistant", "content": solution}]}))
    else:
        for (pid, _), p in train.items():
            if not p.get("options") or (p.get("options_quality") == "low" and not args.include_low):
                continue
            examples.append((pid, {"messages": build_messages(p) + [
                {"role": "assistant", "content": p["label"]}]}))

    leaked = {pid for pid, _ in examples} & val_ids
    if leaked:
        sys.exit(f"{len(leaked)} val ids in the training data: stop")
    ids = sorted({pid for pid, _ in examples})
    random.Random(42).shuffle(ids)
    held = set(ids[:max(1, len(ids) // 20)])
    out = os.path.expanduser(args.out)
    os.makedirs(out, exist_ok=True)
    counts = {}
    for name, keep in (("train", lambda pid: pid not in held), ("valid", lambda pid: pid in held)):
        rows = [e for pid, e in examples if keep(pid)]
        random.Random(42).shuffle(rows)
        counts[name] = len(rows)
        with open(os.path.join(out, name + ".jsonl"), "w", encoding="utf-8") as f:
            for e in rows:
                f.write(json.dumps(e, ensure_ascii=False) + "\n")
    letters = {}
    for _, e in examples:
        letter = e["messages"][-1]["content"][-1]
        letters[letter] = letters.get(letter, 0) + 1
    print(f"{args.format}: {counts['train']} train + {counts['valid']} valid examples from {len(ids)} ids "
          f"-> {out}")
    print(f"  answers: {dict(sorted(letters.items()))}")
    if skipped:
        print(f"  skipped: {skipped}")


if __name__ == "__main__":
    main()
