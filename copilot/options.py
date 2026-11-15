"""The cascade's command-line options, shared by predict.py and eval.py."""
import argparse
import csv
import json
import os
import re
import sys
import time

from .config import ROOT

DEFAULT_CALIB = os.path.join(ROOT, "results", "calib-default.json")


def mode_type(value):
    if not re.fullmatch(r"direct|reason:\d+", value):
        raise argparse.ArgumentTypeError("use direct or reason:<tokens>, e.g. reason:128")
    return value


def add_cascade_arguments(parser):
    parser.add_argument("--no-llm", action="store_true", help="never call the model: the prior answers")
    parser.add_argument("--twins", action="store_true",
                        help="look for the same problem, already solved, in the bank first (off by default)")
    parser.add_argument("--bank", default="", metavar="JSONL", help="solved problems for --twins")
    parser.add_argument("--calib", default="", metavar="PATH",
                        help="calibration file (default: results/calib-default.json if it exists)")
    parser.add_argument("--mode", type=mode_type, default="direct", help="direct or reason:<tokens>")
    parser.add_argument("--fewshot", type=int, default=0, metavar="N",
                        help="R6: add N solved training problems to the system prompt (read once, then cached)")
    parser.add_argument("--fewshot-bank", default="data/v2/train.jsonl", metavar="JSONL",
                        help="where the --fewshot examples come from: training problems only")
    parser.add_argument("--perms", type=int, default=1, metavar="N", choices=range(1, 6),
                        help="R5: ask N times with the options moved round and average (each costs a re-read)")
    parser.add_argument("--code", choices=["auto", "on", "off"], default="auto",
                        help="the code stage: the model writes a short program and the board runs it. "
                             "auto (default): on when the code model server answers")
    parser.add_argument("--probs", default=[], nargs="+", metavar="CSV",
                        help="replay the probabilities saved by laptop/dump_probs.py instead of calling "
                             "the model (for scoring a whole split fast; the times mean nothing). "
                             "Several files are joined")
    parser.add_argument("--priors", default="", metavar="PATH",
                        help="prior tables to use instead of results/priors.json (for another split)")


def load_probs(paths):
    """{(id, language): ({letter: p}, letter_mass)} from one or more dumps. Rows the
    dump tool could not read from the model (status other than ok) are left out."""
    table = {}
    for path in [paths] if isinstance(paths, str) else paths:
        with open(path, newline="", encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r.get("status", "ok") != "ok":
                    continue
                mass = float(r["letter_mass"]) if r.get("letter_mass") else None
                table[r["id"], r["language"]] = ({k: float(r["p" + k]) for k in "ABCDE"}, mass)
    return table


def build_ctx(args, default_bank):
    """The run options that solve() gets with every problem. Built once, before the loop."""
    ctx = {"twins": None, "calib": None, "no_llm": args.no_llm, "mode": args.mode, "probs": None,
           "prefix": None, "prefix_warm": False, "perms": args.perms}
    if args.probs:
        ctx["probs"] = load_probs(args.probs)
    ctx["code"] = False
    if args.code != "off" and not args.no_llm and not args.probs:
        from . import tir
        seconds = tir.warm_up()
        if seconds is None and args.code == "on":
            sys.exit(f"--code on, but no code model answers at {tir.URL}")
        ctx["code"] = seconds is not None
        ctx["code_speeds"] = tir.Speeds()
        print(f"code stage: {'on, warm-up ' + format(seconds, '.1f') + ' s' if ctx['code'] else 'off (no server)'}"
              f" ({tir.URL})", file=sys.stderr)
    if args.priors:
        from . import priors
        if not priors.load_tables(args.priors):
            sys.exit(f"{args.priors} could not be read as prior tables.")
    if args.fewshot and not args.no_llm and not args.probs:
        from .fewshot import build_prefix
        from .pipeline import warm_up
        if not os.path.exists(args.fewshot_bank):
            sys.exit(f"{args.fewshot_bank} not found. Pass the training problems with --fewshot-bank.")
        with open(args.fewshot_bank, encoding="utf-8") as f:
            ctx["prefix"], ids = build_prefix([json.loads(line) for line in f if line.strip()], args.fewshot)
        ctx["fewshot_ids"] = ids
        seconds = warm_up(ctx)
        print(f"few-shot prefix: {len(ids)} examples ({', '.join(ids)}), {len(ctx['prefix'])} characters; "
              + (f"warm-up call {seconds:.1f} s" if seconds is not None else "WARM-UP FAILED"),
              file=sys.stderr)
    calib = args.calib or (DEFAULT_CALIB if os.path.exists(DEFAULT_CALIB) else "")
    if calib:
        with open(calib, encoding="utf-8") as f:
            ctx["calib"] = json.load(f)
    if args.twins:
        from .twins import TwinIndex
        bank = args.bank or default_bank
        if not os.path.exists(bank):
            sys.exit(f"{bank} not found. Pass the solved problems for --twins with --bank.")
        start = time.perf_counter()
        with open(bank, encoding="utf-8") as f:
            ctx["twins"] = TwinIndex([json.loads(line) for line in f if line.strip()])
        print(f"twin bank: {len(ctx['twins'].docs)} problems from {bank}, "
              f"built in {time.perf_counter() - start:.1f} s", file=sys.stderr)
    return ctx
