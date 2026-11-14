#!/usr/bin/env python3
"""Measure the teaching half: the hint ladder and the adaptive trainer.

    python3 tools/teach_report.py                       # every number below
    python3 tools/teach_report.py --sample 20           # also 20 problems to judge by hand

Standard library only: run it on the laptop or on the board, from the
repository root, after laptop/build_teach_bank.py.

  1. the bank: how many problems by language, level and topic
  2. rung 1 (a similar, easier problem): how often there is one, how often it
     is really at a lower level and on the same topic, and that it is never a twin
  3. rung 3 (the model's hint): how many hints the code check kept, why the
     others were rejected, and the leak rate of what is shown
  4. the time of each rung on this machine
  5. simulated students: does the trainer settle at the student's level, and
     does it stay on topic after a miss?
"""
import argparse
import os
import random
import sqlite3
import statistics
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "app", "math-matrix", "python"))

import tutor  # noqa: E402


def pct(part, whole):
    return f"{part}/{whole} = {100 * part / whole:.1f}%" if whole else "0/0"


def simulate(bank, lang, ability, start, steps, seed, noise=0.0):
    """A student who solves alone everything up to level `ability`, and nothing above it."""
    rng = random.Random(seed)
    t = tutor.Tutor(bank=bank, history=tutor.History(":memory:"), seed=seed)
    t.start(f"sim{ability}", lang, start)
    levels, picks, on_topic, after_miss = [], Counter(), 0, 0
    for _ in range(steps):
        problem = t.problem
        levels.append(problem["level"])
        knows = problem["level"] <= ability
        if rng.random() < noise:
            knows = not knows
        if knows:
            t.answer(problem["label"])
        missed, topic = not knows, problem["topic"]
        t.next()  # an unanswered problem counts as "not yet"
        picks[t.picked_by] += 1
        if missed:
            after_miss += 1
            on_topic += t.problem["topic"] == topic
    return levels, picks, on_topic, after_miss


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bank", default=tutor.BANK_PATH)
    p.add_argument("--sample", type=int, default=0, help="write this many rung-1 pairs and hints to judge by hand")
    p.add_argument("--out", default="results/teach-sample.txt")
    p.add_argument("--steps", type=int, default=40)
    args = p.parse_args()
    bank = tutor.Bank(args.bank)
    problems = list(bank.problems.values())

    print("1. THE BANK")
    print(f"  {len(problems)} problems, {len({x['id'] for x in problems})} ids, languages: "
          + ", ".join(f"{lang} {len(bank.of(lang))}" for lang in bank.languages))
    ref = bank.of(bank.languages[-1] if "eng" not in bank.languages else "eng")
    print("  by level: " + "  ".join(f"{k}:{v}" for k, v in sorted(Counter(x["level"] for x in ref).items())))
    print("  by topic: " + "  ".join(f"{k}:{v}" for k, v in Counter(x["topic"] for x in ref).most_common()))

    print("\n2. RUNG 1: a similar, easier problem")
    found = easier = same_topic = twins = same_value = 0
    scores, times, pairs = [], [], []
    for x in problems:
        begin = time.perf_counter()
        example, info = bank.easier_example(x)
        times.append(1000 * (time.perf_counter() - begin))
        if not example:
            continue
        found += 1
        easier += info["easier"]
        same_topic += info["same_topic"]
        twins += info["score"] >= tutor.TWIN_CAP
        same_value += example["options"][example["label"]] == x["options"][x["label"]]
        scores.append(info["score"])
        pairs.append((x, example, info))
    print(f"  has an example:        {pct(found, len(problems))}")
    print(f"  at a lower level:      {pct(easier, found)}   (the rest are at the same level)")
    print(f"  on the same topic:     {pct(same_topic, found)}")
    print(f"  similarity:            median {statistics.median(scores):.2f}, lowest {min(scores):.2f}, highest {max(scores):.2f}")
    print(f"  twins shown (must be 0): {twins}")
    print(f"  example's answer value equals this problem's: {pct(same_value, found)}")

    print("\n3. RUNG 3: one sentence from the small model, checked in code")
    db = sqlite3.connect(args.bank)
    log = db.execute("SELECT h.id, h.lang, h.hint, h.rejected, h.seconds FROM hint_log h "
                     "JOIN problems p ON p.id = h.id AND p.lang = h.lang").fetchall()
    db.close()
    if not log:
        print("  no hints generated yet: run laptop/build_teach_bank.py --hints")
    for lang in bank.languages:
        rows = [r for r in log if r[1] == lang]
        if not rows:
            continue
        kept = [r for r in rows if not r[3]]
        print(f"  {lang}: generated {len(rows)}, kept {pct(len(kept), len(rows))}")
        for reason, n in Counter(r[3] for r in rows if r[3]).most_common():
            print(f"       rejected, {reason}: {pct(n, len(rows))}")
    raw_leaks = sum(bool(tutor.leak(r[2], bank.get(r[0], r[1]))) for r in log)
    shown = [(bank.get(pid, lang), hint) for (pid, lang), hint in bank.hints.items() if bank.get(pid, lang)]
    shown_leaks = sum(bool(tutor.leak(hint, x)) for x, hint in shown)
    checkable = sum(tutor.verifiable(x) for x, _hint in shown)
    if log:
        print(f"  leak rate of what the model wrote:   {pct(raw_leaks, len(log))}")
        print(f"  leak rate of what the student sees:  {pct(shown_leaks, len(shown))}")
        print(f"  problems where the check can tell (the answer value is not in the question): {pct(checkable, len(shown))}")
        print(f"  problems in the bank with a stored hint: {pct(len(shown), len(problems))}")
        print(f"  words per hint: median {statistics.median(len(h.split()) for _x, h in shown):.0f}")

    print("\n4. TIME PER RUNG, on this machine")
    begin = time.perf_counter()
    for x in problems:
        tutor.method(x["topic"], x["language"])
    print(f"  rung 1, similar problem: median {statistics.median(times):.2f} ms, worst {max(times):.2f} ms")
    print(f"  rung 2, topic and method: {1000 * (time.perf_counter() - begin) / len(problems):.4f} ms")
    print("  rung 3, stored hint: a dictionary lookup")
    if log:
        seconds = [r[4] for r in log]
        print(f"  rung 3, generated: median {statistics.median(seconds):.2f} s, worst {max(seconds):.2f} s here. "
              "On the board, measure it: about 11 tokens/s read, 6 written.")

    print(f"\n5. SIMULATED STUDENTS, {args.steps} problems each")
    print("  A student who solves alone up to their level, and nothing above it.")
    print("  ability  start  language  mean level, last 15   on topic after a miss   how the next problem was found")
    for lang in bank.languages:
        for ability in (1, 3, 5):
            for start in (0, 7):
                levels, picks, on_topic, misses = simulate(bank, lang, ability, start, args.steps, seed=ability * 10 + start)
                tail = levels[-15:]
                print(f"  {ability:^7}  {start:^5}  {lang:^8}  {statistics.mean(tail):^20.1f}  {pct(on_topic, misses):^22}  "
                      + ", ".join(f"{k} {v}" for k, v in picks.most_common()))
    print("  A good trainer hovers between the student's level and one above it: it keeps offering a stretch.")

    if args.sample:
        rng = random.Random(42)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        with open(args.out, "w", encoding="utf-8") as f:
            f.write("Judge by hand. For each pair: is the example really easier (E)? about the same idea (S)?\n"
                    "For each hint: is it useful (U)? misleading (M)?\n")
            for x, example, info in rng.sample(pairs, min(args.sample, len(pairs))):
                f.write(f"\n--- {x['id']} {x['language']} level {x['level']} {x['topic']}\n{x['text']}\n"
                        f"EXAMPLE {example['id']} level {example['level']} {example['topic']} score {info['score']}\n"
                        f"{example['text']}\nHINT: {bank.hints.get((x['id'], x['language']), '(none)')}\n"
                        "E [ ]  S [ ]  U [ ]  M [ ]\n")
        print(f"\nWrote {args.out} (M3Kang text: don't commit it)")


if __name__ == "__main__":
    main()
