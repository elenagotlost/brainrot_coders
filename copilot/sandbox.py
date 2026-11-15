"""Run a short program written by the math model and return what it printed.

The program runs in a separate Python process with a time limit, a memory limit (Linux),
a reduced set of builtins and no imports (math, itertools and Fraction are already there).
A static check refuses anything that reaches outside: imports of other modules, dunder
attributes, open/exec/eval... This guards against a small model's mistakes, not against an
attacker: only run programs written by our own local models.

Standard library only: this runs on the board.
"""
import ast
import json
import re
import subprocess
import sys

FORBIDDEN_NAMES = {
    "open", "exec", "eval", "compile", "__import__", "globals", "locals", "vars", "getattr",
    "setattr", "delattr", "input", "breakpoint", "exit", "quit", "help", "memoryview", "type",
    "object", "super", "classmethod", "staticmethod", "property", "dir", "id", "hash",
}
FORBIDDEN_NODES = (ast.Import, ast.ImportFrom, ast.Global, ast.Nonlocal, ast.ClassDef,
                   ast.AsyncFunctionDef, ast.Await, ast.With, ast.AsyncWith, ast.AsyncFor)

# Imports of modules the sandbox already provides are removed instead of refused.
# [ \t] and not \s: \s would also match a newline and swallow the next line of code.
HARMLESS_IMPORT = re.compile(
    r"^[ \t]*(?:import[ \t]+(?:math|itertools|fractions)(?:[ \t]*,[ \t]*(?:math|itertools|fractions))*"
    r"|from[ \t]+(?:math|itertools|fractions)[ \t]+import[ \t]+[\w \t,*()]+)[ \t]*$", re.M)

RUNNER = r'''
import builtins, io, json, math, sys, itertools
from fractions import Fraction
try:
    import resource
    resource.setrlimit(resource.RLIMIT_AS, (768 * 1024 * 1024, 768 * 1024 * 1024))
except Exception:
    pass
SAFE = ("abs all any bool dict divmod enumerate filter float frozenset int isinstance len list map "
        "max min pow print range reversed round set sorted str sum tuple zip chr ord bin next iter "
        "Exception ValueError ZeroDivisionError StopIteration ArithmeticError").split()
g = {"__builtins__": {k: getattr(builtins, k) for k in SAFE}}
for k in dir(math):
    if not k.startswith("_"):
        g[k] = getattr(math, k)
for k in "permutations combinations combinations_with_replacement product accumulate".split():
    g[k] = getattr(itertools, k)
g.update(math=math, itertools=itertools, Fraction=Fraction)
real, printed = sys.stdout, io.StringIO()
sys.stdout = printed
try:
    exec(compile(sys.stdin.read(), "<program>", "exec"), g)
    out = {"printed": printed.getvalue()[-300:]}
except Exception as e:
    out = {"printed": printed.getvalue()[-300:], "error": type(e).__name__ + ": " + str(e)[:120]}
real.write(json.dumps(out))
'''


def clean(code):
    """Drop imports of math, itertools and fractions: everything in them is already available."""
    return HARMLESS_IMPORT.sub("", code)


def check(code):
    """'' when the program may run, else the reason it may not."""
    try:
        tree = ast.parse(code)
    except SyntaxError as e:
        return f"SyntaxError: {e.msg}"
    for node in ast.walk(tree):
        if isinstance(node, FORBIDDEN_NODES):
            return f"not allowed: {type(node).__name__}"
        if isinstance(node, ast.Name) and (node.id in FORBIDDEN_NAMES or node.id.startswith("__")):
            return f"not allowed: {node.id}"
        if isinstance(node, ast.Attribute) and node.attr.startswith("_"):
            return f"not allowed: .{node.attr}"
    return ""


def run(code, timeout=3.0):
    """Run a program; return (what it printed, error message or '')."""
    code = clean(code)
    why = check(code)
    if why:
        return "", why
    try:
        done = subprocess.run([sys.executable, "-I", "-S", "-c", RUNNER], input=code, text=True,
                              capture_output=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return "", "timeout: the program took too long"
    try:
        out = json.loads(done.stdout)
    except ValueError:
        return "", "crashed"
    return out.get("printed", "").strip(), out.get("error", "")
