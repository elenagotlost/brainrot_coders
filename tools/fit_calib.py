#!/usr/bin/env python3
"""Fit the calibration file of one model from its probability dump.

    python tools/fit_calib.py --probs results/probs-<model>-direct-train.csv \\
           --data data/v2/train.jsonl --out results/calib-<model>.json

Two things are fitted, on training problems only:

  bias    the model's mean P(letter) over the problems that have options.
          Dividing by it removes the model's taste for some letters.
  alpha   how much to trust the model against the no-model prior, one value
          for numeric options and one for text options. The answer is
          argmax of alpha * log p_model + (1 - alpha) * log p_prior.
          alpha is chosen from 0, 0.1, ..., 1 by 5-fold cross-validation
          grouped by problem id (both languages of an id are in the same
          fold). alpha = 0 ignores the model, so "the prior alone" is always
          one of the candidates.

Inside each fold the bias and the prior tables are fitted on the other four
folds, so no problem is scored with numbers that saw it.

It also writes two helper files for ablations, next to --out:
  calib-<model>-raw.json        alpha 1, no bias: the model alone
  calib-<model>-debiased.json   alpha 1, with the bias: the debiased model alone

Standard library only.
"""
import argparse
import csv
import hashlib
import json
import os
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "laptop"))

from copilot import priors  # noqa: E402
from copilot.pipeline import debias, fuse  # noqa: E402
from fit_priors import fit as fit_prior_tables, use as use_prior_tables  # noqa: E402

LETTERS = "ABCDE"
GRID = [round(0.1 * i, 1) for i in range(11)]
FOLDS = 5
KINDS = ("numeric", "text")


def fold_of(problem_id):
    return int(hashlib.md5(problem_id.encode()).hexdigest(), 16) % FOLDS


def load_data(path):
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def load_probs(path):
    """{(id, language): {letter: p}} for the rows where the model was really read."""
    table, skipped = {}, 0
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                skipped += 1
                continue
            table[r["id"], r["language"]] = {k: float(r["p" + k]) for k in LETTERS}
    return table, skipped


def mean_bias(rows):
    return {k: sum(r["p_llm"][k] for r in rows) / len(rows) for k in LETTERS}


def with_prior(rows):
    """Attach the prior of the currently loaded tables to each row."""
    out = []
    for r in rows:
        p_prior, kind = priors.prior(r["problem"])
        out.append({**r, "p_prior": p_prior, "kind": kind})
    return out


def pct(hits):
    return f"{100 * sum(hits) / len(hits):5.1f}%" if hits else "   n/a"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--probs", required=True, help="the model's dump on the training split")
    p.add_argument("--data", default="data/v2/train.jsonl", help="the training problems of that dump")
    p.add_argument("--out", required=True, help="results/calib-<model>.json")
    p.add_argument("--model", default="", help="name stored in the file (default: taken from --probs)")
    args = p.parse_args()

    problems = load_data(args.data)
    probs, skipped = load_probs(args.probs)
    model = args.model or os.path.basename(args.probs).removeprefix("probs-").split("-direct-")[0]

    rows, no_probs = [], 0
    for problem in problems:
        if len(problem.get("options") or {}) != 5:
            continue  # no options: the cascade never calls the model there
        key = (problem["id"], problem["language"])
        if key not in probs:
            no_probs += 1
            continue
        rows.append({"problem": problem, "label": problem["label"], "p_llm": probs[key],
                     "fold": fold_of(problem["id"])})
    if not rows:
        sys.exit("No training problem with options has probabilities in the dump.")
    print(f"Model {model}: {len(rows)} training rows with options and probabilities "
          f"({no_probs} without probabilities, {skipped} dump rows skipped for their status)")

    # Out-of-fold answers for every candidate, pooled over the five folds
    hits = defaultdict(lambda: defaultdict(list))  # hits[kind][candidate] = [bool, ...]
    for fold in range(FOLDS):
        fit_ids = {q["id"] for q in problems if fold_of(q["id"]) != fold}
        use_prior_tables(fit_prior_tables([q for q in problems if q["id"] in fit_ids]))
        fit_rows = [r for r in rows if r["fold"] != fold]
        bias = mean_bias(fit_rows)
        for r in with_prior([r for r in rows if r["fold"] == fold]):
            flat = {k: 0.2 for k in LETTERS}
            fair = debias(r["p_llm"], bias)
            kind = r["kind"]
            hits[kind]["raw"].append(fuse(r["p_llm"], flat, 1.0) == r["label"])
            hits[kind]["debiased"].append(fuse(fair, flat, 1.0) == r["label"])
            for alpha in GRID:
                hits[kind][alpha].append(fuse(fair, r["p_prior"], alpha) == r["label"])

    alpha, cv_acc = {}, {}
    print(f"\n5-fold cross-validation on train, grouped by id (accuracy by alpha)")
    print(f"  {'alpha':8}" + "".join(f"{a:>7}" for a in GRID))
    for kind in KINDS:
        if not hits[kind]["raw"]:
            alpha[kind] = 0.0
            continue
        score = {a: sum(hits[kind][a]) for a in GRID}
        best = max(score.values())
        alpha[kind] = min(a for a in GRID if score[a] == best)  # ties: lean on the prior
        n = len(hits[kind]["raw"])
        print(f"  {kind:8}" + "".join(f"{100 * round(score[a] / n, 4):>6.1f}%" for a in GRID) + f"   n = {n}")
        cv_acc[kind] = {"n": n,
                        "prior": round(score[0.0] / n, 4),
                        "raw": round(sum(hits[kind]["raw"]) / n, 4),
                        "debiased": round(sum(hits[kind]["debiased"]) / n, 4),
                        "fused": round(score[alpha[kind]] / n, 4)}
    pooled = {name: [h for kind in KINDS for h in hits[kind][name]] for name in ("raw", "debiased", 0.0)}
    fused_hits = [h for kind in KINDS for h in hits[kind][alpha[kind]]]
    cv_acc["all"] = {"n": len(fused_hits),
                     "prior": round(sum(pooled[0.0]) / len(fused_hits), 4),
                     "raw": round(sum(pooled["raw"]) / len(fused_hits), 4),
                     "debiased": round(sum(pooled["debiased"]) / len(fused_hits), 4),
                     "fused": round(sum(fused_hits) / len(fused_hits), 4)}

    print(f"\n  {'':8} {'n':>5} {'prior only':>11} {'raw model':>10} {'debiased':>9} {'fused':>7}   chosen alpha")
    for kind in KINDS + ("all",):
        if kind not in cv_acc:
            continue
        c = cv_acc[kind]
        chosen = f"{alpha[kind]}" if kind in alpha else ""
        print(f"  {kind:8} {c['n']:>5} {100 * c['prior']:>10.1f}% {100 * c['raw']:>9.1f}% "
              f"{100 * c['debiased']:>8.1f}% {100 * c['fused']:>6.1f}%   {chosen}")
    print("  (the fused column is a little optimistic: alpha was picked on these same folds)")

    # The final file: bias from all training rows, prior tables back to the full fit
    use_prior_tables(fit_prior_tables(problems))
    bias = {k: round(v, 4) for k, v in mean_bias(rows).items()}
    calib = {"model": model, "bias": bias, "alpha": alpha, "cv_acc": cv_acc, "n_train": len(rows),
             "fitted_on": [args.probs.replace("\\", "/"), args.data.replace("\\", "/")]}
    print("\nbias (mean P of each letter on train): " + "  ".join(f"{k} {bias[k]:.3f}" for k in LETTERS))
    stem = args.out[:-5] if args.out.endswith(".json") else args.out
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    variants = {args.out: calib,
                stem + "-raw.json": {"model": model, "alpha": {k: 1.0 for k in KINDS}},
                stem + "-debiased.json": {"model": model, "bias": bias, "alpha": {k: 1.0 for k in KINDS}}}
    for path, content in variants.items():
        with open(path, "w", encoding="utf-8", newline="\n") as f:
            json.dump(content, f, indent=1)
            f.write("\n")
    print("Written: " + ", ".join(variants))


if __name__ == "__main__":
    main()
