#!/usr/bin/env python3
"""E0: does llama-server reuse a cached prompt prefix on this model?

    python tools/e0_cache_check.py [--out results/e0.txt]

Four calls, all with the payload of copilot.llm.answer_letter:
  0. an unrelated prompt, so nothing useful is cached (the script can be re-run)
  1. long system prompt (about 500 tokens) + sample problem 1      cold
  2. the same system prompt + sample problem 2                      prefix reuse?
  3. the same system prompt + problem 2, options shifted by one     permutation reuse?

The verdict compares timings.prompt_n (tokens really processed) with the
tokens that changed. It uses only data/sample_problems.jsonl (made-up
problems), so nothing from M3Kang is read or written. Standard library only.
"""
import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from copilot.client import post  # noqa: E402
from copilot.config import CONFIG, ROOT  # noqa: E402
from copilot.llm import SYSTEM, format_problem  # noqa: E402

BASE = f"http://127.0.0.1:{CONFIG['LLM_PORT']}"
LETTERS = "ABCDE"
TARGET_SYSTEM_TOKENS = 500
SLACK = 16  # chat-template tokens after the user text, plus tokenizer edge effects

FILLER = [
    "Read the whole problem before you look at the options.",
    "Work with exact numbers and keep fractions as fractions when you can.",
    "Check the units of every quantity in the problem.",
    "If the problem asks for a count, make sure you count each case once.",
    "If the problem is about a figure, think about its symmetry.",
    "Estimate the size of the answer and drop options that are far from it.",
    "When two options look alike, check the small difference between them.",
    "Try small cases first when the problem has a general rule.",
    "Check that your answer fits every condition in the problem.",
    "For ages, dates and clocks, write down what is known at each moment.",
    "For digits, remember that the first digit of a number is never zero.",
    "For areas and perimeters, split the figure into simple parts.",
    "For averages, work with the total instead of the single values.",
    "For remainders, look at the pattern of the first few cases.",
    "For speed problems, write distance, time and speed for each part.",
    "For logic problems, test each statement as true and as false.",
    "Never answer with a number: answer with the letter of the option.",
    "Reply with one letter and nothing else.",
]


def tokenize(text):
    return post(f"{BASE}/tokenize", {"content": text}, timeout=30)["tokens"]


def shifted(problem):
    """The same problem with its options moved one place: A<-B, ..., E<-A."""
    options = problem["options"]
    moved = {LETTERS[i]: options[LETTERS[(i + 1) % 5]] for i in range(5)}
    return {**problem, "options": moved}


def build_system(samples):
    parts = [SYSTEM, "", "Solved examples:"]
    for ex in samples:
        parts.append(format_problem(ex) + f"\nAnswer: {ex['label']}")
        parts.append("")
    parts.append("Advice:")
    text = "\n".join(parts)
    for line in FILLER:
        if len(tokenize(text)) >= TARGET_SYSTEM_TOKENS:
            break
        text += "\n- " + line
    return text


def call(system, user):
    start = time.perf_counter()
    response = post(f"{BASE}/v1/chat/completions", {
        "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
        "grammar": "root ::= [A-E]",
        "max_tokens": 1,
        "temperature": 0,
        "seed": int(CONFIG["SEED"]),
        "cache_prompt": True,
    }, timeout=600)
    timings = response.get("timings", {})
    return {
        "total": response.get("usage", {}).get("prompt_tokens"),
        "prompt_n": timings.get("prompt_n"),
        "cache_n": timings.get("cache_n"),
        "prompt_ms": timings.get("prompt_ms"),
        "wall_s": round(time.perf_counter() - start, 2),
        "letter": response["choices"][0]["message"]["content"].strip(),
    }


def common_prefix(a, b):
    n = 0
    while n < min(len(a), len(b)) and a[n] == b[n]:
        n += 1
    return n


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--samples", default=os.path.join(ROOT, "data", "sample_problems.jsonl"))
    p.add_argument("--out", default=os.path.join(ROOT, "results", "e0.txt"))
    args = p.parse_args()

    with open(args.samples, encoding="utf-8") as f:
        samples = [json.loads(line) for line in f if line.strip()]
    system = build_system(samples)
    user1 = format_problem(samples[0]) + "\nAnswer:"
    user2 = format_problem(samples[1]) + "\nAnswer:"
    user3 = format_problem(shifted(samples[1])) + "\nAnswer:"

    n_system = len(tokenize(system))
    tok2, tok3 = tokenize(user2), tokenize(user3)
    changed3 = len(tok3) - common_prefix(tok2, tok3)  # tokens from the first changed position

    runs = [
        ("0 unrelated prompt (clears the cache)", call("Say one letter.", "Pick a letter.\nAnswer:")),
        ("1 system + problem 1 (cold)", call(system, user1)),
        ("2 system + problem 2", call(system, user2)),
        ("3 system + problem 2, options shifted", call(system, user3)),
    ]
    _, r1 = runs[1]
    _, r2 = runs[2]
    _, r3 = runs[3]

    prefix_ok = r2["prompt_n"] is not None and r2["prompt_n"] <= len(tok2) + SLACK
    perm_ok = r3["prompt_n"] is not None and r3["prompt_n"] <= changed3 + SLACK
    if perm_ok:
        perm_verdict = "YES"
    elif r3["prompt_n"] is not None and r3["prompt_n"] <= len(tok3) + SLACK:
        perm_verdict = "NO, only the system prefix is reused: every permutation re-reads the whole problem"
    else:
        perm_verdict = "NO, nothing is reused"

    props = {}
    try:
        from copilot.client import get
        props = get(f"{BASE}/props")
    except Exception:  # the verdict doesn't need it
        pass

    lines = [
        "E0 caching check",
        f"time: {time.strftime('%Y-%m-%d %H:%M:%S')}",
        f"server: {BASE}  model: {os.path.basename(str(props.get('model_path', '?')))}"
        f"  build: {props.get('build_info', '?')}",
        f"system prompt: {n_system} tokens;  problem 2: {len(tok2)} tokens;"
        f"  problem 2 shifted: {len(tok3)} tokens, {changed3} from the first changed position",
        "",
        f"{'call':40} {'total':>6} {'prompt_n':>9} {'cache_n':>8} {'prompt_ms':>10} {'wall_s':>7}",
    ]
    for name, r in runs:
        ms = f"{r['prompt_ms']:.0f}" if r["prompt_ms"] is not None else "?"
        lines.append(f"{name:40} {str(r['total']):>6} {str(r['prompt_n']):>9} {str(r['cache_n']):>8} "
                     f"{ms:>10} {r['wall_s']:>7}")
    lines += [
        "",
        f"prefix reuse: {'YES' if prefix_ok else 'NO'}"
        f"  (call 2 processed {r2['prompt_n']} tokens; reuse means at most {len(tok2)} + {SLACK};"
        f" no reuse means about {r2['total']})",
        f"permutation reuse: {perm_verdict}"
        f"  (call 3 processed {r3['prompt_n']} tokens; reuse means at most {changed3} + {SLACK};"
        f" prefix-only reuse means about {len(tok3)}; no reuse means about {r3['total']})",
        f"cold call for scale: call 1 processed {r1['prompt_n']} tokens in "
        + (f"{r1['prompt_ms']:.0f} ms" if r1["prompt_ms"] is not None else "? ms"),
    ]
    report = "\n".join(lines)
    print(report)
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    print(f"\nWritten to {args.out}")


if __name__ == "__main__":
    main()
