#!/usr/bin/env python3
"""Measure a model's speed on the board: how fast it reads, how fast it writes.

One file, standard library only, no imports from the repository: copy it to the board
next to a problems file and run it against any llama-server.

    python3 board_latency.py --input val.jsonl --url http://127.0.0.1:8081 --name q08-q8
    python3 board_latency.py --input val.jsonl --url http://127.0.0.1:8091 --name q2b-q4

For each of N problems that have options it makes the two calls the reasoning mode makes:
  1. write up to --tokens tokens of working      -> reading speed and writing speed
  2. add "Answer:" and read one letter           -> cost of the final letter call
and one plain call, the baseline's direct letter -> seconds for a no-thinking answer.

Prints the speeds and a table: thinking budget -> estimated seconds per problem, against
the 60 s limit. Writes latency-<name>.csv with one row per problem.
"""
import argparse
import csv
import json
import statistics
import time
import urllib.request

DIRECT_SYSTEM = ("You solve multiple-choice math competition problems for school students. "
                 "Read the problem and the options, then reply with the letter of the correct option.")
REASON_SYSTEM = ("You solve multiple-choice math competition problems for school students. "
                 "Reason step by step, very briefly, then finish with a line 'Answer: X' "
                 "where X is the letter of the correct option.")


def post(url, payload, timeout):
    request = urllib.request.Request(url + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    start = time.perf_counter()
    with urllib.request.urlopen(request, timeout=timeout) as response:
        body = json.loads(response.read())
    return body, time.perf_counter() - start


def problem_text(problem):
    lines = [problem["text"].strip()]
    lines += [f"{k}) {problem['options'][k]}" for k in "ABCDE" if k in problem["options"]]
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--input", required=True)
    ap.add_argument("--url", default="http://127.0.0.1:8081")
    ap.add_argument("--name", required=True, help="model name for the output file, e.g. q2b-q4")
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--tokens", type=int, default=64, help="tokens of working to write per problem")
    ap.add_argument("--timeout", type=float, default=300)
    args = ap.parse_args()

    with open(args.input, encoding="utf-8") as f:
        problems = [json.loads(line) for line in f if line.strip()]
    problems = [p for p in problems if p.get("options")][:args.n]
    common = {"temperature": 0, "seed": 42, "cache_prompt": True}

    rows = []
    for i, p in enumerate(problems, 1):
        text = problem_text(p)
        # the baseline: one letter, no thinking
        body, direct_s = post(args.url, dict(common, max_tokens=1, grammar="root ::= [A-E]", messages=[
            {"role": "system", "content": DIRECT_SYSTEM},
            {"role": "user", "content": text + "\nAnswer:"}]), args.timeout)
        direct_read = body.get("timings", {})
        # reasoning call 1: write the working
        messages = [{"role": "system", "content": REASON_SYSTEM}, {"role": "user", "content": text}]
        body, gen_s = post(args.url, dict(common, max_tokens=args.tokens, stop=["Answer:"],
                                          messages=messages), args.timeout)
        t = body.get("timings", {})
        working = body["choices"][0]["message"]["content"].rstrip()
        # reasoning call 2: one letter after "Answer:"
        prefill = (working + "\n" if working else "") + "Answer:"
        body, letter_s = post(args.url, dict(common, max_tokens=1, messages=messages + [
            {"role": "assistant", "content": prefill}]), args.timeout)
        t2 = body.get("timings", {})
        rows.append({
            "id": p["id"], "language": p.get("language", ""),
            "direct_s": round(direct_s, 2), "direct_prompt_n": direct_read.get("prompt_n", 0),
            "read_n": t.get("prompt_n", 0), "read_ms": round(t.get("prompt_ms", 0)),
            "write_n": t.get("predicted_n", 0), "write_ms": round(t.get("predicted_ms", 0)),
            "gen_s": round(gen_s, 2), "letter_s": round(letter_s, 2),
            "letter_prompt_n": t2.get("prompt_n", 0), "letter_cache_n": t2.get("cache_n", 0),
        })
        r = rows[-1]
        print(f"[{i}/{len(problems)}] direct {r['direct_s']:.1f} s | read {r['read_n']} tok in "
              f"{r['read_ms'] / 1000:.1f} s, wrote {r['write_n']} tok in {r['write_ms'] / 1000:.1f} s, "
              f"letter {r['letter_s']:.1f} s (re-read {r['letter_prompt_n']} tok)", flush=True)

    out = f"latency-{args.name}.csv"
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)

    read_speed = sum(r["read_n"] for r in rows) / max(sum(r["read_ms"] for r in rows) / 1000, 1e-9)
    write_speed = sum(r["write_n"] for r in rows) / max(sum(r["write_ms"] for r in rows) / 1000, 1e-9)
    direct = [r["direct_s"] for r in rows]
    letter = [r["letter_s"] for r in rows]
    read_s = [r["read_ms"] / 1000 for r in rows]
    print(f"\n== {args.name}, {len(rows)} problems")
    print(f"Direct letter:  median {statistics.median(direct):.1f} s, worst {max(direct):.1f} s")
    print(f"Reading speed:  {read_speed:.1f} tokens/s   (problem read: median {statistics.median(read_s):.1f} s, "
          f"worst {max(read_s):.1f} s)")
    print(f"Writing speed:  {write_speed:.2f} tokens/s")
    print(f"Letter call:    median {statistics.median(letter):.1f} s, worst {max(letter):.1f} s "
          f"(re-read {statistics.median([r['letter_prompt_n'] for r in rows]):.0f} tokens; "
          f"a small number means the cache was reused)")
    print("\nThinking budget -> estimated seconds per problem (read + write + letter), limit 60 s")
    print("budget   median   worst")
    for budget in (0, 32, 64, 96, 128, 192, 256, 384):
        est = [rs + budget / write_speed + ls for rs, ls in zip(read_s, letter)]
        flag = "" if max(est) <= 50 else ("   over the limit" if max(est) > 60 else "   tight")
        print(f"{budget:6}   {statistics.median(est):6.1f}   {max(est):5.1f}{flag}")
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
