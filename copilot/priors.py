"""No-model answer priors for Kangaroo problems (stdlib only, runs on the board).

Why it works: Kangaroo distractors are built around the correct answer, so the
correct value is rarely the smallest or the largest option.

  numeric options  -> the rank of each value among the sorted options decides
  text options     -> the "medoid" string, the one most similar to the others
  no options       -> the letter distribution of problems without options

The three tables are fitted on the training split only by
laptop/fit_priors.py, which writes results/priors.json. Without that file the
values below are used: they are the same fit, on data/v2/train.jsonl.
"""
import json
import os
import re
from difflib import SequenceMatcher

from .config import ROOT

LETTERS = "ABCDE"
NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?")
PRIORS_PATH = os.path.join(ROOT, "results", "priors.json")

# P(correct | rank among the sorted values), smallest value first
RANK_PRIOR = [0.0987, 0.2174, 0.2861, 0.2726, 0.1251]
# P(letter) for problems without options, and for all problems
EMPTY_PRIOR = {"A": 0.1772, "B": 0.169, "C": 0.2057, "D": 0.277, "E": 0.1711}
LETTER_PRIOR = {"A": 0.1535, "B": 0.1944, "C": 0.2244, "D": 0.2503, "E": 0.1774}


def load_tables(path=PRIORS_PATH):
    """Replace the three tables with the ones in a priors.json. True if it was read."""
    global RANK_PRIOR, EMPTY_PRIOR, LETTER_PRIOR
    try:
        with open(path, encoding="utf-8") as f:
            tables = json.load(f)
        rank = [float(x) for x in tables["rank_prior"]]
        empty = {k: float(tables["empty_prior"][k]) for k in LETTERS}
        letter = {k: float(tables["letter_prior"][k]) for k in LETTERS}
        if len(rank) != 5:
            raise ValueError("rank_prior needs five values")
    except (OSError, ValueError, KeyError, TypeError):
        return False
    RANK_PRIOR, EMPTY_PRIOR, LETTER_PRIOR = rank, empty, letter
    return True


def _num(t):
    t = t.replace(",", ".").replace(" ", "")
    if "/" in t:
        a, b = t.split("/")
        return float(a) / float(b) if float(b) else None
    return float(t)


def _median(xs):
    s = sorted(xs)
    return s[len(s) // 2]


def option_values(options):
    """One number per option, or None. Robust to stray OCR glyphs ("1 17", "场 17")."""
    if len(options) != 5:
        return None
    cands = []
    for k in LETTERS:
        found = [_num(x) for x in NUM.findall(options[k].replace("−", "-"))]
        found = [x for x in found if x is not None]
        if not found:
            return None
        cands.append(found)
    singles = [c[0] for c in cands if len(c) == 1]
    ref = _median(singles) if singles else None
    vals = []
    for c in cands:
        if len(c) == 1 or ref is None:
            vals.append(c[-1])
        else:  # keep the number whose size looks like the other options
            vals.append(min(c, key=lambda x: abs(len(str(int(abs(x)))) - len(str(int(abs(ref)))))))
    return vals if len(set(vals)) == 5 else None


def prior(problem):
    """Return ({letter: probability}, kind). kind is empty, numeric or text."""
    options = problem.get("options") or {}
    if len(options) != 5 or any(k not in options for k in LETTERS):
        return dict(EMPTY_PRIOR), "empty"
    vals = option_values(options)
    if vals:
        order = sorted(range(5), key=lambda i: vals[i])
        p = {LETTERS[i]: RANK_PRIOR[r] for r, i in enumerate(order)}
        return p, "numeric"
    strs = [options[k] for k in LETTERS]
    dist = [sum(1 - SequenceMatcher(None, a, b).ratio() for b in strs) for a in strs]
    best = min(range(5), key=lambda i: (dist[i], i))
    p = {k: 0.8 * LETTER_PRIOR[k] for k in LETTERS}
    p[LETTERS[best]] += 0.2
    return p, "text"


def best_letter(problem):
    p, kind = prior(problem)
    return max(p, key=p.get), kind


load_tables()
