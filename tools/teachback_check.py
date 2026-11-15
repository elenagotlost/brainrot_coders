#!/usr/bin/env python3
"""Check the teach mode's code side, without a model: what the student's message ends on, when
Kanga is convinced, and what Kanga may say.

    python3 tools/teachback_check.py

Uses made-up problems, not M3Kang. Runs in a second on the board and on the laptop.
Standard library only.
"""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app", "math-matrix", "python"))
from teachback import (analyse, calculations, conclusion, kanga_reply, question_grammar,  # noqa: E402
                       read_result, reply_problem)

PROBLEM = {"id": "made-up-1", "language": "eng", "label": "C",
           "text": "Anna has 3 bags with 4 apples in each bag. She eats 2 apples. How many apples are left?",
           "options": {"A": "8", "B": "9", "C": "10", "D": "12", "E": "14"}}
NOTES = "3 bags with 4 apples is 3 * 4 = 12 apples. She eats 2, so 12 - 2 = 10 are left."
CANDY = {"id": "made-up-2", "language": "eng", "label": "C",
         "text": "Marc has 9 candies and Tom has 17. How many candies must Tom give to Marc so that they have the same?",
         "options": {"A": "2", "B": "3", "C": "4", "D": "6", "E": "8"}}
DOWN = "http://127.0.0.1:9"   # no model server: Kanga uses its fixed lines


def status(*explanations, notes=NOTES, problem=PROBLEM):
    return analyse(problem, list(explanations), notes)["status"]


def main():
    # calculations: worked out, wrong, unfinished
    sums = calculations("so (9 + 17) / 2 = 13, and 17 - 13 = 4")
    assert [(s["text"], s["value"], s["slip"], s["unfinished"]) for s in sums] == \
        [("(9 + 17) / 2 = 13", "13", None, False), ("17 - 13 = 4", "4", None, False)]
    assert calculations("9 + 17 = 25")[0]["slip"] == ("9 + 17", "25")
    assert calculations("26 / 3 = 8,67")[0]["slip"] is None          # rounded is fine
    assert calculations("he gives 17 - 13")[0]["unfinished"]
    assert calculations("17 - 13 =")[0]["unfinished"]
    assert calculations("there are 9 and 17") == []

    # what one message ends on
    assert conclusion(PROBLEM, "so the answer is C") == ("C", "10")
    assert conclusion(PROBLEM, "12 minus 2 gives 10") == ("C", "10")
    assert conclusion(PROBLEM, "the answer is a number") == (None, None)
    assert conclusion(PROBLEM, "I think it's 11") == ("?", "11")
    assert conclusion(PROBLEM, "the answer is 7 because 2 + 5 = 7") == ("?", "7")
    assert conclusion(PROBLEM, "crec que surt 9") == ("B", "9")
    # numbers in the middle of a calculation, or from the problem, are not results
    assert conclusion(CANDY, "add the two numbers together and divide by 3") == (None, None)
    assert conclusion(CANDY, "add the two numbers together and divide by 2") == (None, None)
    assert conclusion(CANDY, "Tom has 17 and Marc has 9") == (None, None)
    assert read_result(CANDY, "each needs 26 / 2 = 13, he gives 17 - 13")["unfinished"] == "17 - 13"

    # when Kanga is convinced
    assert status("no idea") == "short"
    assert status("I multiply the bags by the apples in each bag and then I take away the ones she ate") == "method"
    assert status("Three bags of four is twelve, and she eats two, so twelve plus two is 14") == "wrong"
    assert analyse(PROBLEM, ["she had 12 and ate 2 so 14"], NOTES)["said"] == "14"
    assert status("It is 10 because that is what I think the answer to this problem should be") == "gap"
    assert status("There are 3 bags with 4 in each, that makes 12, she eats 2 and 10 are left") == "convinced"
    assert status("It is 10 because that is what I think the answer to this problem should be",
                  "I count all the apples first and then I take away what she ate, so 10") == "convinced"
    assert status("10") == "short"
    assert status("Three bags with four apples each, take away the two she ate: 10 apples", notes="") == "convinced"
    off = analyse(PROBLEM, ["I think the answer is 7"], NOTES)
    assert off["status"] == "off" and off["said"] == "7"
    slip = analyse(CANDY, ["9 + 17 = 25, half is 12, so he gives 17 - 12 = 5"], "")
    assert (slip["status"], slip["expr"], slip["stated"]) == ("slip", "9 + 17", "25")
    assert status("together 9 + 17 = 26, each needs 26 / 2 = 13, he gives 17 - 13", problem=CANDY, notes="") == "compute"
    # the conversation that went wrong: a method with no number is not stuck on an old wrong result
    yours = ["add the two numbers together and divide by 3", "I count something and the answer is 7 because 2 + 5 = 7"]
    assert status(*yours[:1], problem=CANDY, notes="") == "method"
    assert status(*yours, problem=CANDY, notes="") == "off"
    assert status(*yours, yours[0], problem=CANDY, notes="") == "repeat"
    method = "add them, divide by 2, then he gives the difference 17 - that previous result"
    assert status(*yours, method, problem=CANDY, notes="") == "method"
    assert status(*yours, method, "so (9 + 17) / 2 = 13, and 17 - 13 = 4", problem=CANDY, notes="") == "convinced"
    # a right answer still stands when the next message only answers Kanga's question
    assert status("9 + 17 = 26, each gets 13, so Tom gives 17 - 13 = 4", "because then both have 13",
                  problem=CANDY, notes="") == "convinced"

    # what Kanga may say
    said = "3 bags of 4 is 12"
    assert reply_problem("I see you got 12. What do you do next?", PROBLEM, said) == ""
    assert reply_problem("So 12 - 2 = 10?", PROBLEM, said) == "uses a number the student did not say"
    assert reply_problem("Is the answer C?", PROBLEM, said) == "points at an option"
    assert reply_problem("I understand now.", PROBLEM, said) == "asks nothing"
    assert reply_problem("What do you do", PROBLEM, said) == "cut off"
    assert reply_problem("Do you get ten in the end?", PROBLEM, said) == "gives the answer in words"

    # without a model server Kanga still answers, never twice the same, and helps more when stuck
    chat = [{"who": "tutor", "text": "intro"}]
    replies = []
    for n, text in enumerate(["add the two numbers and divide by 3", "add the two numbers and divide by 3",
                              "add the two numbers and divide by 3", "add the two numbers and divide by 3"], 1):
        chat.append({"who": "student", "text": text})
        analysis = analyse(CANDY, [t["text"] for t in chat if t["who"] == "student"], "")
        reply, source = kanga_reply(CANDY, "eng", chat, analysis, DOWN, timeout=1, help_text="Try a smaller example.")
        assert source.startswith("fixed"), source
        replies.append(reply)
        chat.append({"who": "tutor", "text": reply})
    assert len(set(replies)) == len(replies), replies
    assert "Try a smaller example." in replies[2] and "Show me how" in replies[3], replies
    slip_chat = [{"who": "tutor", "text": "intro"}, {"who": "student", "text": "9 + 17 = 25"}]
    reply, _ = kanga_reply(CANDY, "eng", slip_chat, analyse(CANDY, ["9 + 17 = 25"], ""), DOWN, timeout=1)
    assert "9 + 17" in reply and "25" in reply, reply

    grammar = question_grammar(["17", "9", "17"])
    assert grammar.count('"17"') == 1 and '"9"' in grammar and grammar.startswith("root ::= first")
    print("teach mode checks pass")


if __name__ == "__main__":
    main()
