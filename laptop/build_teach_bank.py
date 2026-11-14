#!/usr/bin/env python3
"""Build teach.db, the bank behind the hint ladder and the adaptive trainer.

    python3 laptop/build_teach_bank.py                          # problems + the similarity map
    python3 laptop/build_teach_bank.py --hints                  # also one model hint per problem
    python3 laptop/build_teach_bank.py --hints --hint-limit 40  # a few, to try it

Standard library only: it runs on the laptop or on the board. Run it from the
repository root, with the language model server up if you pass --hints (the
same 0.8B model and flags as the board: bash laptop/serve.sh <model>).

What goes in:
  problems      text-only M3Kang problems whose five options were read cleanly,
                so the student never needs a picture
  neighbours    for each problem, the 40 most similar ones (character n-gram
                TF-IDF from copilot/twins.py, on the English text when there is one)
  hints         rung 3: one sentence from the small model, kept ONLY if the code
                check in tutor.py says it does not contain the answer
  explanations  optional worked solutions (laptop/teacher.py output), kept only
                when the teacher reached the right letter

The output holds M3Kang text: it is ignored by Git (*.db). Never commit it.
"""
import argparse
import json
import os
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
APP = os.path.join(ROOT, "app", "math-matrix", "python")
sys.path.insert(0, ROOT)
sys.path.insert(0, APP)

import tutor  # noqa: E402
from copilot import topics  # noqa: E402
from copilot.twins import TwinIndex  # noqa: E402

NEIGHBOURS = 40


def usable(problem):
    return (problem.get("text_only") and len(problem.get("options") or {}) == 5
            and problem.get("options_quality", "ok") == "ok" and problem.get("label") in tutor.LETTERS
            and len(set(problem["options"].values())) == 5)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--input", nargs="+", default=["data/train.jsonl", "data/val.jsonl"])
    p.add_argument("--output", default=os.path.join(APP, "teach.db"))
    p.add_argument("--hints", action="store_true", help="generate rung-3 hints with the model server")
    p.add_argument("--hint-limit", type=int, default=0, help="at most this many new hints (0: all)")
    p.add_argument("--llm", default="http://127.0.0.1:8081")
    p.add_argument("--explanations", default="", metavar="JSONL", help="laptop/teacher.py --mode explain output")
    args = p.parse_args()

    problems = []
    for path in args.input:
        with open(path, encoding="utf-8") as f:
            problems += [json.loads(line) for line in f if line.strip()]
    bank = [x for x in problems if usable(x)]
    by_id = {}
    for x in bank:
        by_id.setdefault(x["id"], {})[x["language"]] = x
    print(f"{len(problems)} problems read, {len(bank)} usable (text only, five clean options), {len(by_id)} ids")

    db = sqlite3.connect(args.output)
    db.executescript("""
        CREATE TABLE IF NOT EXISTS problems (id TEXT, lang TEXT, level INTEGER, year INTEGER, topic TEXT,
            text TEXT, options TEXT, label TEXT, PRIMARY KEY (id, lang));
        CREATE TABLE IF NOT EXISTS neighbours (id TEXT, nb_id TEXT, score REAL);
        CREATE TABLE IF NOT EXISTS hints (id TEXT, lang TEXT, hint TEXT, PRIMARY KEY (id, lang));
        CREATE TABLE IF NOT EXISTS hint_log (id TEXT, lang TEXT, hint TEXT, rejected TEXT, seconds REAL,
            PRIMARY KEY (id, lang));
        CREATE TABLE IF NOT EXISTS explanations (id TEXT, lang TEXT, text TEXT, PRIMARY KEY (id, lang));
        CREATE INDEX IF NOT EXISTS neighbours_id ON neighbours (id);
        DELETE FROM problems; DELETE FROM neighbours; DELETE FROM explanations;
    """)

    # the topic belongs to the id: take it from the data, or tag the English text
    for pid, versions in by_id.items():
        ref = versions.get("eng") or next(iter(versions.values()))
        topic = ref.get("topic") or topics.tag(ref["text"])[0]
        for lang, x in versions.items():
            db.execute("INSERT INTO problems VALUES (?,?,?,?,?,?,?,?)",
                       (pid, lang, x["level"], x["year"], topic, x["text"],
                        json.dumps(x["options"], ensure_ascii=False), x["label"]))

    # the map: one text per id, the English one when there is one
    docs = [versions.get("eng") or next(iter(versions.values())) for versions in by_id.values()]
    start = time.perf_counter()
    index = TwinIndex(docs)
    for doc in docs:
        hits = index.search(doc, k=NEIGHBOURS + 1, exclude=lambda d, me=doc["id"]: d["id"] == me)
        db.executemany("INSERT INTO neighbours VALUES (?,?,?)",
                       [(doc["id"], other["id"], round(score, 4)) for score, other in hits[:NEIGHBOURS]])
    db.commit()
    print(f"similarity map: {len(docs)} problems, {NEIGHBOURS} neighbours each, {time.perf_counter() - start:.1f} s")

    if args.explanations:
        kept = 0
        with open(args.explanations, encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                if row.get("correct") and row.get("finished") and (row["id"] in by_id):
                    text = row["text"].rsplit("Answer:", 1)[0].strip()
                    db.execute("INSERT OR REPLACE INTO explanations VALUES (?,?,?)", (row["id"], row["language"], text))
                    kept += 1
        db.commit()
        print(f"explanations: {kept} kept (teacher reached the right letter)")

    if args.hints:
        done = {(pid, lang) for pid, lang in db.execute("SELECT id, lang FROM hint_log")}
        todo = [x for x in bank if (x["id"], x["language"]) not in done]
        if args.hint_limit:
            todo = todo[:args.hint_limit]
        print(f"hints: {len(done)} already tried, {len(todo)} to go")
        for n, x in enumerate(todo, 1):
            try:
                hint, rejected, seconds = tutor.generate_hint(x, args.llm, timeout=120)
            except Exception as error:
                print(f"  {x['id']} {x['language']}: {type(error).__name__}: {error}")
                continue
            db.execute("INSERT OR REPLACE INTO hint_log VALUES (?,?,?,?,?)",
                       (x["id"], x["language"], hint, rejected, round(seconds, 2)))
            if not rejected:
                db.execute("INSERT OR REPLACE INTO hints VALUES (?,?,?)", (x["id"], x["language"], hint))
            if n % 50 == 0 or n == len(todo):
                db.commit()
                print(f"  [{n}/{len(todo)}] {x['id']} {x['language']}: {rejected or 'ok'} | {hint[:90]}", flush=True)
        db.commit()

    # hints for problems that left the bank are dropped; the log keeps what was tried
    db.execute("DELETE FROM hints WHERE NOT EXISTS (SELECT 1 FROM problems p WHERE p.id = hints.id AND p.lang = hints.lang)")
    db.commit()
    tried, kept = db.execute("SELECT COUNT(*) FROM hint_log").fetchone()[0], db.execute("SELECT COUNT(*) FROM hints").fetchone()[0]
    print(f"Wrote {args.output}: {len(bank)} problems, hints kept {kept} of {tried} tried")
    db.close()


if __name__ == "__main__":
    main()
