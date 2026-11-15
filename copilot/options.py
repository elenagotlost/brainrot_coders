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
                        help="look for the same problem, already solved, in the bank first, and stop if "
                             "the bank is missing. predict.py does this by itself when its bank is there")
    parser.add_argument("--no-twins", action="store_true", help="never look problems up in the bank")
    parser.add_argument("--bank", default="", metavar="PATH",
                        help="solved problems for the lookup: a .jsonl file, or a folder of them")
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


def load_bank(path):
    """The solved problems of a bank: one .jsonl file, or every .jsonl file in a folder.

    Rows without a text or a label are skipped. In a folder, each id gets its file's name in
    front, because two sources can use the same id for different problems (M3Kang-style ids
    of another year), and the lookup tells problems apart by id."""
    folder = os.path.isdir(path)
    files = sorted(os.path.join(path, name) for name in os.listdir(path) if name.endswith(".jsonl")) if folder else [path]
    problems = []
    for file in files:
        source = os.path.splitext(os.path.basename(file))[0]
        with open(file, encoding="utf-8") as f:
            for line in f:
                try:
                    row = json.loads(line)
                except ValueError:
                    continue
                if not isinstance(row, dict) or not row.get("text") or row.get("label") not in list("ABCDE"):
                    continue
                if folder:
                    row["id"] = f"{source}:{row.get('id', '')}"
                problems.append(row)
    return problems


def build_ctx(args, default_bank, auto_twins=False):
    """The run options that solve() gets with every problem. Built once, before the loop.

    auto_twins (predict.py): the lookup is on whenever the bank is there. A missing or broken
    bank is reported and the run goes on without it: it must never cost the other answers."""
    ctx = {"twins": None, "calib": None, "no_llm": args.no_llm, "mode": args.mode, "probs": None,
           "prefix": None, "prefix_warm": False, "perms": args.perms}
    if args.probs:
        ctx["probs"] = load_probs(args.probs)
    ctx["code"] = False
    if args.code != "off" and not args.no_llm and not args.probs:
        from . import coder
        seconds = coder.warm_up()
        if seconds is None and args.code == "on":
            sys.exit(f"--code on, but no code model answers at {coder.URL}")
        ctx["code"] = seconds is not None
        print(f"code stage: {'on, warm-up ' + format(seconds, '.1f') + ' s' if ctx['code'] else 'off (no server)'}"
              f" ({coder.URL})", file=sys.stderr)
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
    bank = args.bank or default_bank
    if args.twins and not os.path.exists(bank):
        sys.exit(f"{bank} not found. Pass the solved problems for --twins with --bank.")
    if not args.no_twins and (args.twins or (auto_twins and os.path.exists(bank))):
        start = time.perf_counter()
        try:
            from .twins import TwinIndex
            problems = load_bank(bank)
            if not problems:
                raise ValueError("no solved problems in it")
            ctx["twins"] = TwinIndex(problems)
            print(f"lookup: on, {len(problems)} solved problems from {bank}, "
                  f"built in {time.perf_counter() - start:.1f} s", file=sys.stderr)
        except Exception as error:
            if args.twins:
                sys.exit(f"--twins, but the bank {bank} could not be read: {error}")
            ctx["twins"] = None
            print(f"lookup: off, the bank {bank} could not be read ({error})", file=sys.stderr)
    elif auto_twins:
        print(f"lookup: off ({'--no-twins' if args.no_twins else 'no bank at ' + bank})", file=sys.stderr)
    return ctx
