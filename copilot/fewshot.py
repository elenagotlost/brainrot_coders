"""R6: a fixed block of solved examples for the system prompt.

The block is the same for every problem, so llama-server reads it once and
reuses it from its cache. It is built at run time from the training split and
is never written to a tracked file: the examples are M3Kang problems.
"""
import random

from .llm import format_problem

LETTERS = "ABCDE"
MAX_TEXT_CHARS = 300  # short problems only: every example costs reading time once


def pick_examples(bank, n, seed=42):
    """n solved problems from the bank, the same ones every run.

    Only English problems whose own five options were read cleanly, with a
    short text and no picture. The answers cycle through A-E, so the examples
    don't teach the model a favourite letter.
    """
    candidates = [p for p in bank
                  if p.get("language") == "eng"
                  and len(p.get("options") or {}) == 5
                  and p.get("options_quality", "ok") == "ok"
                  and p.get("options_source", "own") == "own"
                  and p.get("text_only", True)
                  and len(p.get("text", "")) <= MAX_TEXT_CHARS]
    candidates.sort(key=lambda p: p["id"])
    rng = random.Random(seed)
    rng.shuffle(candidates)
    picked = []
    for i in range(n):
        wanted = LETTERS[i % 5]
        match = next((p for p in candidates if p["label"] == wanted and p not in picked), None)
        if match is None:
            break
        picked.append(match)
    rng.shuffle(picked)  # so the answers are not in A, B, C, D, E order
    return picked


def build_prefix(bank, n, seed=42):
    """Return (prefix text, ids of the examples). Empty text when n is 0."""
    examples = pick_examples(bank, n, seed) if n > 0 else []
    if not examples:
        return "", []
    parts = ["Solved examples:"]
    for example in examples:
        parts.append(format_problem(example) + f"\nAnswer: {example['label']}")
    return "\n\n".join(parts), [example["id"] for example in examples]
