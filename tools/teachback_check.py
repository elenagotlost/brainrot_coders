#!/usr/bin/env python3
"""Check the teach mode's code side, without a model: when Kanga is convinced, and what it may say.

    python3 tools/teachback_check.py

Uses a made-up problem, not M3Kang. Runs in a second on the board and on the laptop.
Standard library only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "math-matrix", "python"))
from teachback import analyse, conclusion, kanga_reply, question_grammar, reply_problem  # noqa: E402

PROBLEM = {"id": "made-up-1", "language": "eng", "label": "C",
           "text": "Anna has 3 bags with 4 apples in each bag. She eats 2 apples. How many apples are left?",
           "options": {"A": "8", "B": "9", "C": "10", "D": "12", "E": "14"}}
NOTES = "3 bags with 4 apples is 3 * 4 = 12 apples. She eats 2, so 12 - 2 = 10 are left."


def status(*explanations, notes=NOTES):
    return analyse(PROBLEM, list(explanations), notes)["status"]


def main():
    # which option an explanation ends on
    assert conclusion(PROBLEM, "so the answer is C") == ("C", "10")
    assert conclusion(PROBLEM, "12 minus 2 gives 10") == ("C", "10")
    assert conclusion(PROBLEM, "the answer is a number") == (None, None)
    assert conclusion(PROBLEM, "I think it's 11") == (None, None)
    assert conclusion(PROBLEM, "the answer is 7 because 2 + 5 = 7") == ("?", "7")
    assert conclusion(PROBLEM, "crec que surt 9") == ("B", "9")

    # when Kanga is convinced
    assert status("no idea") == "short"
    assert status("I multiply the bags by the apples in each bag and then I take away the ones she ate") == "no_answer"
    assert status("Three bags of four is twelve, and she eats two, so twelve plus two is 14") == "wrong"
    assert analyse(PROBLEM, ["she had 12 and ate 2 so 14"], NOTES)["said"] == "14"
    assert status("It is 10 because that is what I think the answer to this problem should be") == "gap"
    assert status("There are 3 bags with 4 in each, that makes 12, she eats 2 and 10 are left") == "convinced"
    # another path than the notes: accepted after a second explanation
    assert status("It is 10 because that is what I think the answer to this problem should be",
                  "I count all the apples first and then I take away what she ate, so 10") == "convinced"
    assert status("10") == "short"
    # no notes: a right answer that is explained is enough
    assert status("Three bags with four apples each, take away the two she ate: 10 apples", notes="") == "convinced"

    # what Kanga may say
    said = "3 bags of 4 is 12"
    assert reply_problem("I see you got 12. What do you do next?", PROBLEM, said) == ""
    assert reply_problem("So 12 - 2 = 10?", PROBLEM, said) == "uses a number the student did not say"
    assert reply_problem("Is the answer C?", PROBLEM, said) == "points at an option"
    assert reply_problem("I understand now.", PROBLEM, said) == "asks nothing"
    assert reply_problem("What do you do", PROBLEM, said) == "cut off"
    assert reply_problem("Do you get ten in the end?", PROBLEM, said) == "gives the answer in words"
    off = analyse(PROBLEM, ["I think the answer is 7"], NOTES)
    assert off["status"] == "wrong" and off["said"] == "7"

    # without a model server Kanga still answers, with its fixed question
    chat = [{"who": "tutor", "text": "intro"}, {"who": "student", "text": "she has 3 bags of 4 so 12, minus 2 is 14"}]
    analysis = analyse(PROBLEM, [chat[-1]["text"]], NOTES)
    reply, source = kanga_reply(PROBLEM, "eng", chat, analysis, "http://127.0.0.1:9", timeout=1)
    assert source.startswith("fixed") and "14" in reply, (reply, source)
    grammar = question_grammar(["17", "9", "17"])
    assert grammar.count('"17"') == 1 and '"9"' in grammar and grammar.startswith("root ::= first")
    print("teach mode checks pass")


if __name__ == "__main__":
    main()
