#!/usr/bin/env python3
"""Check the math-model stage without a model: option reading, the sandbox, answer parsing.

    python3 tools/tir_check.py

Runs on the board and on the laptop in a few seconds. Standard library only.
"""
import os
import sys
from fractions import Fraction

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from copilot.sandbox import run                                    # noqa: E402
from copilot.tir import applies, boxed_answer, looped, numeric_options, option_value  # noqa: E402

PROBLEM = {"text": "How many two-digit numbers are divisible by 7?", "text_only": True,
           "options": {"A": "12", "B": "13", "C": "14", "D": "15", "E": "90"}, "label": "B"}


def main():
    # options as numbers
    assert option_value("12") == 12
    assert option_value("3,5") == Fraction(7, 2)
    assert option_value("3/4") == Fraction(3, 4)
    assert option_value("12 cm") == 12
    assert option_value("25%") == 25
    assert option_value("1 000") == 1000
    assert option_value("−2") == -2
    assert option_value("2 and 3") is None
    assert numeric_options(PROBLEM)["B"] == 13
    assert numeric_options({"options": {"A": "1", "B": "1", "C": "2", "D": "3", "E": "4"}}) is None
    assert applies(PROBLEM)
    assert not applies({**PROBLEM, "text_only": False})

    # the sandbox: what the program printed
    assert run("print(sum(1 for n in range(10, 100) if n % 7 == 0))") == ("13", "")
    assert run("from itertools import permutations\nprint(len(list(permutations(range(4)))))") == ("24", "")
    assert run("import math\nprint(math.comb(5, 2), sqrt(16))") == ("10 4.0", "")
    assert run("print(Fraction(1, 3) + Fraction(1, 6))") == ("1/2", "")
    assert run("print(1)\nprint(1/0)")[0] == "1"
    assert run("print(1)\nprint(1/0)")[1].startswith("ZeroDivisionError")
    assert run("import os")[1].startswith("not allowed")
    assert run("print(().__class__)")[1].startswith("not allowed")
    assert run("open('x')")[1].startswith("not allowed")
    assert run("while True:\n    pass", timeout=1)[1].startswith("timeout")

    # reading the model's answer
    values = numeric_options(PROBLEM)
    assert boxed_answer("so \\boxed{13}", values) == ("B", "13")
    assert boxed_answer("\\boxed{(C)}", values) == ("C", "(C)")
    assert boxed_answer("\\boxed{13 \\text{ numbers}}", values)[0] == "B"
    assert boxed_answer("\\boxed{11}", values) == ("", "11")
    assert boxed_answer("no box", values) == ("", None)
    assert looped("a long line here\nx\na long line here\na long line here")
    assert not looped("one line\nanother line")
    print("tir checks pass")


if __name__ == "__main__":
    main()
