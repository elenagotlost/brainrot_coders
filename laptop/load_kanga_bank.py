#!/usr/bin/env python3
"""Check Kanga's scripts and store them in teach.db, where the Teach Kanga mode reads them.

    python3 laptop/load_kanga_bank.py                       # data/kanga_bank.json -> teach.db
    python3 laptop/load_kanga_bank.py --check               # only check, change nothing

A script is written for one problem (data/kanga_bank.json, by hand or with a large model):

  answer    the right letter: must be the problem's label, in every language
  steps     the key steps of a solution, in order: a question that leads to the step, and the
            numbers that show the student has done it ("needs": any one of them)
  wrong     for every wrong option, a question aimed at the mistake that leads to it
  solution  the worked solution, shown only when the student asks for it

The rule that keeps Kanga from giving the answer away, checked here for every question in
every language: it may only use numbers of the problem and, for a wrong option, that option's
own value. So no question can state the answer or a step's result. No letter is named either.

Teach Kanga then only serves the problems that have a script. The scripts are derived from
M3Kang: keep them in data/ and teach.db, which are never committed. Standard library only.
"""
import argparse
import json
import os
import re
import sqlite3
import sys

APP = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "math-matrix", "python")
sys.path.insert(0, APP)
from teachback import LETTER_SAID, NUMBER_WORDS, OPTION_LETTER, number_set  # noqa: E402

LETTERS = "ABCDE"


def problems_of(db):
    found = {}
    for pid, lang, text, options, label in db.execute("SELECT id, lang, text, options, label FROM problems"):
        found.setdefault(pid, {})[lang] = {"text": text, "options": json.loads(options), "label": label}
    return found


def line_problems(line, problem, allowed_value=None):
    """Why a line of Kanga's may not be shown, or ""."""
    if not isinstance(line, str) or len(line.strip()) < 8:
        return "missing"
    if line.strip()[-1] not in "?.!":
        return "does not end a sentence"
    if LETTER_SAID.search(line) or OPTION_LETTER.search(line):
        return "names an option letter"
    allowed = number_set(problem["text"]) | (number_set(allowed_value) if allowed_value else set())
    extra = number_set(line) - allowed
    if extra:
        return f"uses numbers that are not in the problem: {sorted(extra)}"
    answer = problem["options"][problem["label"]]
    for number in number_set(answer):
        if number.isdigit() and int(number) <= 20:
            for names in NUMBER_WORDS.values():
                word = names[int(number)]
                if len(word) > 2 and re.search(rf"\b{word}\b", line.lower()) \
                        and not re.search(rf"\b{word}\b", problem["text"].lower()):
                    return f"says the answer in words ({word})"
    return ""


def check(script, problems, languages):
    """A list of problems with one script, empty when it can be used."""
    out = []
    pid = script.get("id")
    if pid not in problems:
        return [f"{pid}: not in the bank"]
    versions = problems[pid]
    for lang in languages:
        if lang not in versions:
            out.append(f"{pid}: no {lang} version in the bank")
    for lang, problem in versions.items():
        if lang not in languages:
            continue
        if script.get("answer") != problem["label"]:
            out.append(f"{pid} {lang}: answer {script.get('answer')} but the label is {problem['label']}")
            continue
        answer_numbers = number_set(problem["options"][problem["label"]])
        for i, step in enumerate(script.get("steps") or []):
            if not step.get("needs"):
                out.append(f"{pid} step {i + 1}: no numbers that show it is done")
            if answer_numbers & set(step.get("needs") or []):
                out.append(f"{pid} step {i + 1}: the answer itself can't show a step is done")
            why = line_problems(step.get(lang), problem)
            if why:
                out.append(f"{pid} {lang} step {i + 1}: {why}")
        if not script.get("steps"):
            out.append(f"{pid}: no steps")
        wrong = script.get("wrong") or {}
        for letter in LETTERS:
            if letter == problem["label"] or letter not in problem["options"]:
                continue
            if letter not in wrong:
                out.append(f"{pid}: no question for the wrong option {letter}")
                continue
            why = line_problems(wrong[letter].get(lang), problem, problem["options"][letter])
            if why:
                out.append(f"{pid} {lang} wrong {letter}: {why}")
        if problem["label"] in wrong:
            out.append(f"{pid}: a 'wrong' question for the right answer")
        if not (script.get("solution") or {}).get(lang):
            out.append(f"{pid} {lang}: no solution")
    return out


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", default=os.path.join("data", "kanga_bank.json"))
    p.add_argument("--db", default=os.path.join(APP, "teach.db"))
    p.add_argument("--languages", default="eng,cat")
    p.add_argument("--check", action="store_true", help="only check the scripts")
    args = p.parse_args()

    with open(args.input, encoding="utf-8") as f:
        scripts = json.load(f)["problems"]
    db = sqlite3.connect(args.db)
    problems = problems_of(db)
    languages = args.languages.split(",")
    good, failed = [], 0
    for script in scripts:
        issues = check(script, problems, languages)
        if issues:
            failed += 1
            print("REFUSED " + "\n        ".join(issues))
        else:
            good.append(script)
    print(f"{len(good)} of {len(scripts)} scripts pass the checks")
    if args.check:
        sys.exit(1 if failed else 0)
    db.execute("CREATE TABLE IF NOT EXISTS kanga (id TEXT PRIMARY KEY, script TEXT NOT NULL)")
    db.execute("DELETE FROM kanga")
    db.executemany("INSERT INTO kanga VALUES (?, ?)", [(s["id"], json.dumps(s, ensure_ascii=False)) for s in good])
    db.commit()
    db.close()
    print(f"Stored {len(good)} scripts in {args.db}: Teach Kanga now serves these problems")


if __name__ == "__main__":
    main()
