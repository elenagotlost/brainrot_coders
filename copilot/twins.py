"""Find a solved "twin" of a problem in M3Kang: the same problem in another
language, or reused at another level/year. Stdlib only (runs on the board).

Signal: character 3-5-gram TF-IDF (works across languages that share names,
symbols and numbers) + Jaccard overlap of the numbers in the text.
Measured cat -> eng on all 1,747 problems: top-1 correct 87 %;
with margin > 0.05: 70 % coverage at 99.3 % precision.

Answer transfer: Kangaroo shuffles the option order when a problem is reused
at another level (9 of 15 reused problems changed letter), so we transfer the
answer VALUE, and only fall back to the letter for the same level.
"""
import math, re
from collections import Counter, defaultdict

def _clean(t):
    return re.sub(r"^\s*\d+\.\s*", "", t).lower()

def _grams(t, lo=3, hi=5):
    out = Counter()
    for w in re.findall(r"\w+|[^\w\s]", t):
        w = f" {w} "
        for n in range(lo, hi + 1):
            for i in range(max(1, len(w) - n + 1)):
                out[w[i:i + n]] += 1
    return out

def _nums(t):
    return set(re.findall(r"\d+", t))

class TwinIndex:
    def __init__(self, problems):
        self.docs = problems
        grams = [_grams(_clean(p["text"])) for p in problems]
        df = Counter(g for d in grams for g in d)
        n = len(problems)
        self.idf = {g: math.log((n + 1) / (c + 1)) + 1 for g, c in df.items() if c >= 2}
        self.vecs, self.post = [], defaultdict(list)
        for j, d in enumerate(grams):
            v = {g: (1 + math.log(c)) * self.idf[g] for g, c in d.items() if g in self.idf}
            norm = math.sqrt(sum(x * x for x in v.values())) or 1.0
            v = {g: x / norm for g, x in v.items()}
            self.vecs.append(v)
            for g, x in v.items():
                self.post[g].append((j, x))
        self.nums = [_nums(_clean(p["text"])) for p in problems]

    def search(self, problem, k=3, exclude=lambda p: False, w_num=0.5):
        q = _grams(_clean(problem["text"]))
        qv = {g: (1 + math.log(c)) * self.idf[g] for g, c in q.items() if g in self.idf}
        norm = math.sqrt(sum(x * x for x in qv.values())) or 1.0
        scores = defaultdict(float)
        for g, x in qv.items():
            for j, y in self.post[g]:
                scores[j] += x / norm * y
        qn = _nums(_clean(problem["text"]))
        out = []
        for j, s in scores.items():
            if exclude(self.docs[j]):
                continue
            a, b = qn, self.nums[j]
            jac = len(a & b) / len(a | b) if (a or b) else 0.5
            out.append((s + w_num * jac, self.docs[j]))
        out.sort(key=lambda t: -t[0])
        return out[:k]

def _norm_opt(s):
    return re.sub(r"\s+", "", s.lower().replace(",", "."))

def transfer(problem, twin):
    """Map the twin's answer onto this problem's options. Returns a letter or None."""
    t_opts, opts = twin.get("options") or {}, problem.get("options") or {}
    if t_opts and opts and twin["label"] in t_opts:
        value = _norm_opt(t_opts[twin["label"]])
        hits = [k for k, v in opts.items() if _norm_opt(v) == value]
        if len(hits) == 1:
            return hits[0]
        tn = re.findall(r"\d+(?:\.\d+)?", value)
        hits = [k for k, v in opts.items() if re.findall(r"\d+(?:\.\d+)?", _norm_opt(v)) == tn and tn]
        if len(hits) == 1:
            return hits[0]
    if problem.get("level") is not None and problem.get("level") == twin.get("level"):
        return twin["label"]   # same exam: translations keep the option order
    return None

def answer(index, problem, same_lang_min=1.1, cross_margin=0.05, exclude=lambda p: False):
    """Return (letter or None, info). Two trust rules, both measured on M3Kang:
    - near-duplicate (score >= same_lang_min): 16/16 correct on val vs train (cross-level reuse)
    - translation (best hit in another language, margin >= cross_margin over the next
      *different id*): 202/202 correct, 72 % coverage (cat queries vs eng index)."""
    hits = index.search(problem, k=6, exclude=exclude)
    if not hits:
        return None, {}
    s, twin = hits[0]
    nxt = next((h for h, d in hits[1:] if d["id"] != twin["id"]), 0.0)
    info = {"twin": twin["id"], "score": round(s, 3), "margin": round(s - nxt, 3)}
    cross = twin.get("language") != problem.get("language")
    if s >= same_lang_min or (cross and s - nxt >= cross_margin):
        return transfer(problem, twin), info
    return None, info
