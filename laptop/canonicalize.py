#!/usr/bin/env python3
"""Canonical data v2: the same problems and the same split, with extra fields.

    python laptop/canonicalize.py --in data --out data/v2 [--no-merge] [--topics topics_by_id.csv]

Reads <in>/train.jsonl and <in>/val.jsonl (written by prepare_m3kang.py) and
writes the same rows to <out>/. No OCR, no download: it only recombines what
is already there. Each row gets four new fields:

  options_source   own | merged:<lang> | none
      K1: a row without options takes the five options that another language
      of the same id has (skipped with --no-merge).
  options_kind     numeric | text | ocr_failed | picture
      ocr_failed: no options and text_only is true (the options were text).
      picture:    no options and text_only is false.
  options_quality  ok | low | none
      low when any of these holds:
        - fewer than five distinct option strings
        - numeric options that disagree with another language of the same id
        - one option has several numbers while another has one (OCR debris)
        - numeric options where an option has several numbers, so the cleanup
          in option_values had to choose one
        - text options copied from another language (nothing to check them against)
  topic            from the rule-based tagger on the English text of the id,
                   or from the --topics CSV (id, topic, ...)

id, language, level, year, text_only, text, label and the split never change.
Standard library only.
"""
import argparse
import csv
import json
import os
import re
import sys
from collections import Counter, defaultdict

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.priors import LETTERS, NUM, option_values  # noqa: E402
from copilot.topics import tag  # noqa: E402

SPLITS = ("train", "val")
NEW_FIELDS = ("options_kind", "options_quality", "options_source", "topic")


def has_options(row):
    return len(row.get("options") or {}) == 5


def normalized(options):
    """Option strings made comparable: lowercase, no spaces, comma as point."""
    return [re.sub(r"\s+", "", options[k].lower().replace(",", ".")) for k in LETTERS]


def numbers_per_option(options):
    return [len(NUM.findall(options[k].replace("−", "-"))) for k in LETTERS]


def pick_donor(row, siblings):
    """K1: the row of another language whose options this row takes, or None."""
    donors = sorted((s for s in siblings if s is not row and has_options(s)),
                    key=lambda s: s["language"])
    numeric = [s for s in donors if option_values(s["options"])]
    if numeric:
        return numeric[0]
    if len(donors) == 1:
        return donors[0]
    # several languages have text options: copy only when they all agree
    if donors and all(normalized(d["options"]) == normalized(donors[0]["options"]) for d in donors):
        return donors[0]
    return None


def kind_of(options, text_only):
    if len(options) != 5:
        return "ocr_failed" if text_only else "picture"
    return "numeric" if option_values(options) else "text"


def quality_of(options, source, siblings, row):
    """K2. siblings are the original rows of the same id (their own OCR)."""
    if len(options) != 5:
        return "none", []
    reasons = []
    if len(set(options.values())) < 5:
        reasons.append("duplicates")
    values = option_values(options)
    counts = numbers_per_option(options)
    if any(c > 1 for c in counts) and any(c == 1 for c in counts):
        reasons.append("mixed-number-counts")
    if values:
        if any(c > 1 for c in counts):
            reasons.append("cleanup-chose")
        for other in siblings:
            if other is row or not has_options(other):
                continue
            other_values = option_values(other["options"])
            if other_values and other_values != values:
                reasons.append("languages-disagree")
                break
    elif source.startswith("merged:"):
        reasons.append("merged-text")
    return ("low" if reasons else "ok"), reasons


def load_topics(path):
    with open(path, newline="", encoding="utf-8") as f:
        return {r["id"]: r["topic"] for r in csv.DictReader(f)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--in", dest="src", default="data", help="folder with train.jsonl and val.jsonl")
    p.add_argument("--out", default="data/v2")
    p.add_argument("--no-merge", action="store_true", help="keep every row's own options (skip K1)")
    p.add_argument("--topics", default="", metavar="CSV", help="id,topic file to use instead of the tagger")
    args = p.parse_args()

    if os.path.abspath(args.src) == os.path.abspath(args.out):
        sys.exit("--out must be a different folder from --in: the v1 files are never overwritten.")

    data = {}
    for split in SPLITS:
        with open(os.path.join(args.src, f"{split}.jsonl"), encoding="utf-8") as f:
            data[split] = [json.loads(line) for line in f if line.strip()]

    by_id = defaultdict(list)
    for split in SPLITS:
        for row in data[split]:
            by_id[row["id"]].append(row)

    csv_topics = load_topics(args.topics) if args.topics else {}
    topics = {}
    for pid, siblings in by_id.items():
        if pid in csv_topics:
            topics[pid] = csv_topics[pid]
            continue
        # the tagger's rules are English; without an English row the tag is a rough guess
        source = next((s for s in siblings if s["language"] == "eng"), siblings[0])
        topics[pid] = tag(source["text"], source.get("text_only", True))[0]

    stats = Counter()
    reasons_count = Counter()
    out_rows = {split: [] for split in SPLITS}
    for split in SPLITS:
        for row in data[split]:
            siblings = by_id[row["id"]]
            lang = row["language"]
            options = row.get("options") or {}
            source = "own" if has_options(row) else "none"
            stats[split, lang, "rows"] += 1
            stats[split, lang, "options before"] += has_options(row)
            if not has_options(row) and not args.no_merge:
                donor = pick_donor(row, siblings)
                if donor is not None:
                    options = dict(donor["options"])
                    source = f"merged:{donor['language']}"
            kind = kind_of(options, row.get("text_only"))
            quality, reasons = quality_of(options, source, siblings, row)
            new = dict(row)
            new["options"] = options
            new["options_kind"] = kind
            new["options_quality"] = quality
            new["options_source"] = source
            new["topic"] = topics[row["id"]]
            out_rows[split].append(new)
            stats[split, lang, "options after"] += len(options) == 5
            stats[lang, "kind", kind] += 1
            stats[lang, "quality", quality] += 1
            stats[lang, "source", source.split(":")[0]] += 1
            for reason in reasons:
                reasons_count[lang, reason] += 1

    os.makedirs(args.out, exist_ok=True)
    for split in SPLITS:
        with open(os.path.join(args.out, f"{split}.jsonl"), "w", encoding="utf-8", newline="\n") as f:
            for row in out_rows[split]:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    languages = sorted({row["language"] for split in SPLITS for row in data[split]})
    print(f"{args.src} -> {args.out}" + ("   (--no-merge)" if args.no_merge else ""))
    print(f"\n{'split':6} {'lang':5} {'rows':>6} {'options before':>15} {'after':>7} {'recovered':>10}")
    for split in SPLITS + ("all",):
        for lang in languages:
            pick = SPLITS if split == "all" else (split,)
            rows = sum(stats[s, lang, "rows"] for s in pick)
            before = sum(stats[s, lang, "options before"] for s in pick)
            after = sum(stats[s, lang, "options after"] for s in pick)
            print(f"{split:6} {lang:5} {rows:>6} {before:>15} {after:>7} {after - before:>10}")
    for field in ("kind", "quality", "source"):
        print(f"\noptions_{field}")
        values = sorted({k[2] for k in stats if len(k) == 3 and k[1] == field and k[0] in languages})
        print(f"  {'':12}" + "".join(f"{lang:>8}" for lang in languages))
        for value in values:
            print(f"  {value:12}" + "".join(f"{stats[lang, field, value]:>8}" for lang in languages))
    print("\nwhy low (a row can have several reasons)")
    for reason in sorted({k[1] for k in reasons_count}):
        print(f"  {reason:20}" + "".join(f"{reasons_count[lang, reason]:>8}" for lang in languages))
    print("\ntopic (problem ids)")
    for topic, n in Counter(topics.values()).most_common():
        print(f"  {topic:20}{n:>6}")


if __name__ == "__main__":
    main()
