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
    parser.add_argument("--probs", default="", metavar="CSV",
                        help="replay the probabilities saved by laptop/dump_probs.py instead of calling "
                             "the model (for scoring a whole split fast; the times mean nothing)")


def load_probs(path):
    """{(id, language): ({letter: p}, letter_mass)} from a dump. Rows the dump tool
    could not read from the model (status other than ok) are left out."""
    table = {}
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("status", "ok") != "ok":
                continue
            mass = float(r["letter_mass"]) if r.get("letter_mass") else None
            table[r["id"], r["language"]] = ({k: float(r["p" + k]) for k in "ABCDE"}, mass)
    return table


def build_ctx(args, default_bank):
    """The run options that solve() gets with every problem. Built once, before the loop."""
    ctx = {"twins": None, "calib": None, "no_llm": args.no_llm, "mode": args.mode, "probs": None}
    if args.probs:
        ctx["probs"] = load_probs(args.probs)
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
