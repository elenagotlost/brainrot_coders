"""No-LLM answer priors for Kangaroo problems (stdlib only, runs on the board).

Why it works: Kangaroo distractors are built around the correct answer, so the
correct value is rarely the smallest or the largest option. Measured on M3Kang
eng (all 1,747 problems):
  numeric options -> pick the median value .......... 28.8 %  (947 problems)
  non-numeric options -> pick the "medoid" string .... 22.9 %  (131)
  empty options -> answer D .......................... 27.4 %  (669)
  overall ............................................ 27.8 %  (always-D: 24.7 %, baseline LLM: ~22.5 %)
Rank of the correct value among the sorted numeric options:
  min 8.4 %, 2nd 22.1 %, median 31.1 %, 4th 27.7 %, max 10.8 %
"""
import re
from difflib import SequenceMatcher

LETTERS = "ABCDE"
NUM = re.compile(r"-?\d+(?:[.,]\d+)?(?:\s*/\s*\d+)?")
RANK_PRIOR = [0.084, 0.221, 0.311, 0.277, 0.108]       # P(correct | rank among sorted values)
EMPTY_PRIOR = {"A": 0.164, "B": 0.178, "C": 0.212, "D": 0.274, "E": 0.172}
LETTER_PRIOR = {"A": 0.160, "B": 0.199, "C": 0.218, "D": 0.247, "E": 0.176}


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
    """Return ({letter: probability}, route)."""
    options = problem.get("options") or {}
    if len(options) != 5:
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
    p, route = prior(problem)
    return max(p, key=p.get), route
