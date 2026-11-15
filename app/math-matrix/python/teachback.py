"""Learning by teaching: the student explains the problem, the tutor plays a classmate who can't solve it.

The student gets a problem and explains to "Kanga" how to solve it. Kanga does not know how,
and keeps asking until the explanation is right and complete. Only then can the student
answer. Explaining a solution to someone else is one of the best ways to learn it.

What is decided where, so that a small model on a slow board can do it:

  while the student reads  the problem is sent to the server once, so it is in the prompt cache
                           and every reply only pays for the student's new text. Kanga's notes
                           are the teacher's solution stored in the bank, when there is one
                           (laptop/teacher.py, then build_teach_bank.py --explanations).
  is the explanation good  decided in code, never by the model (analyse()): does it reach one of
                           the options, is that the right one, does it go through the key numbers
                           of the notes, is it long enough to follow?
  what Kanga says          a first sentence from code, about what code found ("You get 8, but
                           something doesn't add up for me."), then ONE question written by the
                           model. The model is never told the answer. A grammar makes it write a
                           single question, with no number but the ones in the problem or in the
                           student's words. The question is checked again in code
                           (reply_problem()). If it fails, or the model is slow or down, a fixed
                           question for that case is used: the mode works with no model at all.

Measured on a laptop with the board's 0.8B: told the answer, it writes it into its reply at once
("which leads to the answer C) 4"), and it could not write usable notes for 2 of 2 problems even
when told the answer. Hence the model only asks; code judges.

Standard library only: this runs in App Lab's container on the board and on a laptop.
"""
import ast
import json
import operator
import re
import time
import urllib.request
from fractions import Fraction

LETTERS = "ABCDE"
MIN_WORDS = 12          # fewer words than this, over all of the student's messages, can't be an explanation
MAX_CHARS = 1200        # what one message may hold
MAX_REPLY_WORDS = 45
NOTES_WAIT_S = 45       # how long a reply waits for the notes before it goes on without them

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
# "the answer is C", "option (B)", "la resposta és D": the letter must be a capital, so "is a number" is not a letter
LETTER_SAID = re.compile(r"(?i:\b(?:answer|option|choice|resposta|opci[oó]|soluci[oó]|solution|letter|lletra)\b)"
                         r"[^.\n]{0,14}?\(?\b([A-E])\b")


def _norm(number):
    number = number.replace(",", ".")
    return number.rstrip("0").rstrip(".") if "." in number else number  # 12,50 and 12.5 are the same number


def number_list(text):
    """The numbers of a text, in order, written the same way."""
    return [_norm(n) for n in NUMBER.findall(text or "")]


def number_set(text):
    return set(number_list(text))


def words(text):
    return len(re.findall(r"\w+", text or ""))


def option_numbers(problem):
    """{letter: its number} for the options that are one number ("12", "12 cm", "3,5 kg")."""
    out = {}
    for letter, value in (problem.get("options") or {}).items():
        found = number_list(value)
        if len(found) == 1:
            out[letter] = found[0]
    return out


# "the answer is 7", "so the result is 12,5", "la resposta és 4", "em surt 3"
ANSWER_IS = re.compile(r"(?i:\b(?:answer|result|resposta|resultat|solution|soluci[oó]|surt|surten)\b)[^0-9.\n]{0,14}?"
                       r"(\d+(?:[.,]\d+)?)")
OFF = "?"   # the letter for "a value that is none of the options"
# A number right after one of these is in the middle of a calculation ("divide by 3"), not its result
OPERAND_BEFORE = re.compile(r"(?i)(?:\b(?:by|times|plus|minus|over|of|from|entre|per|més|menys|de|amb|with)|[+\-*/×÷])"
                            r"\s*\(?\s*$")
OPERAND_AFTER = re.compile(r"^\s*\)?\s*[+\-*/×÷]\s*\(?\s*\d")
# A written calculation: numbers, + - * / × ÷, brackets and =
SUM = re.compile(r"[\d(][\d\s.,()+\-*/×÷=]*[\d)=]")
OPERATOR = re.compile(r"\d\s*\)?\s*[+\-*/×÷]\s*\(?\s*\d")
# Words that describe a calculation: the student says what they do, without the numbers
METHOD_WORDS = re.compile(
    r"(?i)\b(add\w*|sum|plus|subtract\w*|minus|take away|difference|divid\w*|half|double|twice|multipl\w*|times|"
    r"product|total|count\w*|sum[ao]|sumar|sumem|rest[ao]|restar|restem|diferència|divideix\w*|meitat|doble|"
    r"vegades|compt\w*)\b")
OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul, ast.Div: operator.truediv}


def exact(expression):
    """The exact value of a plain calculation ("9 + 17", "(9+17)/2", "3,5 × 2"), or None."""
    text = expression.replace("×", "*").replace("÷", "/")
    text = re.sub(r"(\d),(\d)", r"\1.\2", text)
    if not text.strip() or not re.fullmatch(r"[\d\s.()+\-*/]+", text):
        return None

    def value(node):
        if isinstance(node, ast.Expression):
            return value(node.body)
        if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)) \
                and not isinstance(node.value, bool):
            return Fraction(str(node.value))
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub):
            return -value(node.operand)
        if isinstance(node, ast.BinOp) and type(node.op) in OPS:
            return OPS[type(node.op)](value(node.left), value(node.right))
        raise ValueError("not a plain calculation")

    try:
        return value(ast.parse(text.strip(), mode="eval"))
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError, RecursionError):
        return None


def shown(value):
    """A Fraction as the student would write it: 4, 3.5, 8.67."""
    if value.denominator == 1:
        return str(value.numerator)
    return f"{float(value):.2f}".rstrip("0").rstrip(".")


def same(computed, written):
    """True when a written result is the computed value: exactly, or rounded when it has decimals."""
    stated = exact(written)
    if stated is None:
        return False
    if stated == computed:
        return True
    return bool(re.search(r"[.,]\d", written)) and abs(float(stated) - float(computed)) <= 0.006 * max(1, abs(float(computed)))


def calculations(text):
    """The calculations written in a text, in order: [{"start", "end", "text", "value", "slip", "unfinished"}].

    "9 + 17 = 25" is a slip: slip is (the calculation, what the student wrote for it).
    "17 - 13" (or "17 - 13 =") is unfinished: the student has not said what it gives.
    The value is that of the last part, so the result of a chain "9 + 17 = 26 / 2 = 13" is 13."""
    found = []
    for m in SUM.finditer(text or ""):
        chunk = m.group(0)
        if not OPERATOR.search(chunk):
            continue
        parts = [part.strip() for part in chunk.split("=")]
        unfinished = parts[-1] == ""
        parts = [part for part in parts if part]
        values = [exact(part) for part in parts]
        if not parts or values[0] is None:
            continue
        slip = None
        for left, right, part in zip(values, values[1:], parts[1:]):
            if right is None or not same(left, part):
                slip = (parts[parts.index(part) - 1], part)
                break
        last = values[-1] if values[-1] is not None else values[0]
        found.append({"start": m.start(), "end": m.end(), "text": chunk.strip().rstrip("=").strip(),
                      "value": shown(last), "slip": slip, "unfinished": unfinished or len(parts) == 1})
    return found


def letter_of(problem, value):
    """The option whose number is this value, or OFF."""
    hits = [letter for letter, number in option_numbers(problem).items() if number == _norm(value)]
    return hits[0] if len(hits) == 1 else OFF


def read_result(problem, text):
    """What one message of the student ends on. {} when it ends on nothing, else a dict with:

      letter      the option it ends on, OFF for a value that is none of the options, None if unfinished
      value       the value, as the student wrote it (or as their own calculation gives it)
      how         "named" (said to be the answer), "sum" (ends a calculation) or "number" (stands alone)
      unfinished  a calculation the student wrote but did not work out ("17 - 13"), or None
      slip        (calculation, what they wrote) for a calculation that is wrong ("9 + 17", "25"), or None

    A number counts as a result when it is named as the answer ("the answer is 7"), when it ends a
    calculation ("= 13"), or when it stands alone and is neither a number of the problem nor a
    number inside a calculation ("divide by 3" ends on nothing). The last result in the text wins."""
    text = text or ""
    sums = calculations(text)
    slips = [s["slip"] for s in sums if s["slip"]]
    inside = [(s["start"], s["end"]) for s in sums]
    given = number_set(problem.get("text", ""))
    candidates = []  # (position, letter, value, unfinished, how)
    for m in LETTER_SAID.finditer(text):
        letter = m.group(1)
        candidates.append((m.start(1), letter, str((problem.get("options") or {}).get(letter, letter)), None, "named"))
    for m in ANSWER_IS.finditer(text):
        candidates.append((m.start(1), letter_of(problem, m.group(1)), m.group(1), None, "named"))
    for s in sums:
        if s["unfinished"]:
            candidates.append((s["end"], None, s["value"], s["text"], "sum"))
        elif not s["slip"]:
            candidates.append((s["end"], letter_of(problem, s["value"]), s["value"], None, "sum"))
        else:  # a wrong calculation: what the student wrote is what they believe
            written = s["slip"][1]
            candidates.append((s["end"], letter_of(problem, written), written, None, "sum"))
    for m in NUMBER.finditer(text):
        if any(a <= m.start() < b for a, b in inside) or _norm(m.group(0)) in given:
            continue
        if OPERAND_BEFORE.search(text[max(0, m.start() - 12):m.start()]) or OPERAND_AFTER.match(text[m.end():]):
            continue
        candidates.append((m.start(), letter_of(problem, m.group(0)), m.group(0), None, "number"))
    if not candidates:  # options that are words: the one named last
        lower, best, where = text.lower(), None, -1
        for letter, value in (problem.get("options") or {}).items():
            value = str(value).strip().lower()
            if len(value) >= 3 and not number_set(value):
                at = lower.rfind(value)
                if at > where:
                    best, where = letter, at
        if best:
            candidates.append((where, best, str(problem["options"][best]), None, "named"))
    if not candidates:
        return {"slip": slips[-1]} if slips else {}
    _at, letter, value, unfinished, how = max(candidates, key=lambda c: c[0])
    return {"letter": letter, "value": value, "unfinished": unfinished, "how": how,
            "slip": slips[-1] if slips else None}


def conclusion(problem, text):
    """(the option a message ends on, its value), or (None, None). OFF: a value that is no option."""
    found = read_result(problem, text)
    if not found.get("letter"):
        return None, None
    return found["letter"], found["value"]


def key_numbers(problem, notes):
    """The numbers a solution goes through: in the notes, but not in the problem and not the answer."""
    if not notes:
        return set()
    answer = set(number_list(str((problem.get("options") or {}).get(problem.get("label"), ""))))
    return number_set(notes) - number_set(problem.get("text", "")) - answer


def verifiable(problem):
    """True when the right option's value can only come from the student: it is not in the problem text."""
    value = str((problem.get("options") or {}).get(problem.get("label"), "")).strip()
    numbers = number_set(value)
    if numbers:
        return bool(numbers - number_set(problem.get("text", "")))
    return bool(value) and value.lower() not in problem.get("text", "").lower()


def _plain(text):
    return re.sub(r"\W+", " ", (text or "").lower()).strip()


def analyse(problem, explanations, notes=""):
    """What the student's explanation so far still lacks. Returns a dict with "status":

      repeat     the latest message is one the student already sent
      slip       a calculation in the latest message is wrong ("expr" and "stated")
      compute    the latest message ends on a calculation it does not work out ("expr")
      off        it ends on a value that is none of the options ("said")
      wrong      it ends on a wrong option ("said")
      method     it says what to do, but not what number that gives
      short      too short to follow, and no answer yet
      no_answer  it does not reach one of the options yet
      gap        right answer, but it skips the key steps of the notes (first explanation only)
      check      right answer, but it can't be checked from the text alone (first explanation only)
      convinced  right answer, and explained: the student may answer now

    Only the latest message decides what it ends on: an earlier result counts only when it was
    the right one (the latest message then answers a question about it). Never asks the model.
    After the student has explained twice, a right answer that is long enough is accepted even if
    it took another path than the notes: there is more than one way to solve a problem."""
    label = problem.get("label")
    latest = explanations[-1] if explanations else ""
    everything = "\n".join(explanations)
    keys = key_numbers(problem, notes)
    covered = keys & number_set(everything)
    out = {"words": words(everything), "keys": len(keys), "covered": len(covered),
           "explanations": len(explanations), "conclusion": None}
    if _plain(latest) and _plain(latest) in [_plain(text) for text in explanations[:-1]]:
        out["status"] = "repeat"
        return out
    found = read_result(problem, latest)
    if found.get("slip"):
        out["status"], out["expr"], out["stated"] = "slip", found["slip"][0], found["slip"][1]
        return out
    if found.get("unfinished"):
        out["status"], out["expr"] = "compute", found["unfinished"]
        return out
    earlier = None  # the result of the most recent earlier message that had one
    for text in reversed(explanations[:-1]):
        earlier = read_result(problem, text).get("letter")
        if earlier:
            break
    letter = found.get("letter")
    if earlier == label and (not letter or (letter == OFF and found.get("how") != "named")):
        letter = label  # the right answer stands: this message justifies a step ("then both have 13")
    elif not letter:
        letter = None   # an earlier wrong answer does not stand: the student may be changing course
    out["conclusion"] = letter
    if letter == OFF:
        out["status"], out["said"] = "off", found["value"]
    elif letter and letter != label:
        out["status"], out["said"] = "wrong", found["value"]
    elif not letter:
        if METHOD_WORDS.search(latest) and words(latest) >= 4:
            out["status"] = "method"
        else:
            out["status"] = "short" if out["words"] < MIN_WORDS else "no_answer"
    elif out["words"] < MIN_WORDS:
        out["status"] = "short"
    elif len(explanations) < 2 and keys and not covered:
        out["status"] = "gap"
    elif len(explanations) < 2 and not verifiable(problem):
        out["status"] = "check"
    else:
        out["status"] = "convinced"
    return out


# ---------------------------------------------------------------- what Kanga says without the model

# The first sentence of Kanga's reply, from code, when the model adds the question.
OPEN = {
    "eng": {"short": "I'm lost already.", "no_answer": "I'm following you so far.",
            "wrong": "Hmm, you get {said}, but something doesn't add up for me.",
            "gap": "You say the result, but I don't see how you get there.", "check": "That sounds possible."},
    "cat": {"short": "Ja m'he perdut.", "no_answer": "Fins aquí et segueixo.",
            "wrong": "Mmm, et surt {said}, però hi ha alguna cosa que no em quadra.",
            "gap": "Dius el resultat, però no veig com hi arribes.", "check": "Sembla que pot ser."},
}

# Whole lines. A list holds variants: Kanga takes the next one rather than say the same thing twice.
SAY = {
    "eng": {
        "intro": "I have no idea how to solve this one. Can you explain it to me, step by step?",
        "short": ["I don't follow yet. Where do you start, and what do you do first?",
                  "Can you say it in small steps? What is the very first thing you do?"],
        "no_answer": ["I think I follow so far. But what comes out at the end, and which option is it?",
                      "Okay, and then? What number do you get in the end?"],
        "method": ["I think I see what you want to do. If you really do that calculation, what number comes out?",
                   "Can you do that with the numbers of the problem and tell me what you get?",
                   "Let's go one step at a time: what is the first number you get when you do that?"],
        "wrong": ["You end with {said}, but I lost you on the way there. Can you go through that last step again?",
                  "Hmm, {said}? Something doesn't fit for me. Can you check your steps once more?"],
        "off": ["You get {said}, but {said} isn't one of the options. Which step could be off?",
                "{said} isn't among the answers, so something must be wrong somewhere. Can you check each step?"],
        "slip": ["Wait, is {expr} really {stated}? I get something different. Can you check that one?",
                 "Hold on: {expr} = {stated}? Can you work that out again, slowly?"],
        "compute": ["And how much is {expr}?", "Nice, and what does {expr} give?"],
        "repeat": ["You said that already, and I'm still lost. Can you explain it another way, with the numbers?",
                   "That's the same as before. Can you show me one step with the actual numbers?"],
        "gap": "How do you get from the numbers in the problem to your result? Can you show me the middle step?",
        "check": "Okay, but why is that true? Can you show me how you checked it?",
        "convinced": "Now I get it, thanks! Go ahead and pick the answer.",
        "stuck": "I'm still lost. Maybe try a smaller example first?",
        "teacher": "My teacher once told me: \"{text}\" Does that help?",
        "showhow": "If you're stuck, you can press \"Show me how\".",
    },
    "cat": {
        "intro": "No tinc ni idea de com resoldre aquest. M'ho pots explicar pas a pas?",
        "short": ["Encara no t'entenc. Per on comences, i què fas primer?",
                  "M'ho pots dir a passos petits? Què és el primer que fas?"],
        "no_answer": ["Fins aquí crec que t'entenc. Però què surt al final, i quina opció és?",
                      "D'acord, i després? Quin nombre et surt al final?"],
        "method": ["Crec que veig què vols fer. Si fas aquest càlcul de veritat, quin nombre et surt?",
                   "Ho pots fer amb els nombres del problema i dir-me què et surt?",
                   "Anem pas a pas: quin és el primer nombre que et surt quan ho fas?"],
        "wrong": ["Acabes amb {said}, però m'he perdut pel camí. Pots tornar a fer l'últim pas?",
                  "Mmm, {said}? Hi ha alguna cosa que no em quadra. Pots revisar els passos?"],
        "off": ["Et surt {said}, però {said} no és cap de les opcions. Quin pas pot estar malament?",
                "{said} no hi és, a les respostes, així que alguna cosa falla. Pots revisar cada pas?"],
        "slip": ["Espera, {expr} fa {stated}? A mi em surt una altra cosa. Ho pots comprovar?",
                 "Un moment: {expr} = {stated}? Ho pots tornar a calcular a poc a poc?"],
        "compute": ["I quant fa {expr}?", "Molt bé, i què dona {expr}?"],
        "repeat": ["Això ja ho has dit, i encara estic perdut. M'ho pots explicar d'una altra manera, amb els nombres?",
                   "És el mateix que abans. M'ensenyes un pas amb els nombres de veritat?"],
        "gap": "Com passes dels nombres del problema al teu resultat? Em pots ensenyar el pas del mig?",
        "check": "D'acord, però per què és així? M'ensenyes com ho has comprovat?",
        "convinced": "Ara ho entenc, gràcies! Ja pots triar la resposta.",
        "stuck": "Encara estic perdut. Potser prova primer amb un exemple més petit?",
        "teacher": "La meva mestra un cop em va dir: \"{text}\" T'ajuda?",
        "showhow": "Si t'has encallat, pots prémer \"Ensenya-m'ho\".",
    },
}


def say(lang, key, avoid=(), **values):
    """Kanga's line for this case, in this language: the first variant not in `avoid`."""
    lines = (SAY.get(lang) or SAY["eng"]).get(key) or SAY["eng"][key]
    lines = [lines] if isinstance(lines, str) else lines
    texts = [line.format(**values) for line in lines]
    return next((text for text in texts if not any(text in line for line in avoid)), texts[0])


# ---------------------------------------------------------------- the model

KANGA_SYSTEM = ("You are Kanga, a curious school student who does NOT know how to solve the problem below. A "
                "classmate explains it to you. Ask ONE short question about the step of their explanation that is "
                "missing, unclear or looks wrong to you. Talk to your classmate directly, as 'you'. Do not solve the "
                "problem, do not calculate and do not give any answer.")
LANGUAGE = {"cat": " Ask your question in Catalan."}
FOCUS = {
    "short": "You can't follow yet. Ask where they start.",
    "no_answer": "They have not said what the result is. Ask what comes out at the end.",
    "wrong": "Their result does not look right to you. Ask about the step that gives their result.",
    "gap": "They skip a step. Ask how they get from the numbers of the problem to the result.",
    "check": "Ask them why one of their steps is true.",
}
# Number words, so that "four" can't give away an answer of 4 either
NUMBER_WORDS = {
    "eng": "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
           "sixteen seventeen eighteen nineteen twenty".split(),
    "cat": "zero u dos tres quatre cinc sis set vuit nou deu onze dotze tretze catorze quinze setze disset "
           "divuit dinou vint".split(),
}


def clean_problem_text(problem):
    return re.sub(r"^\s*\d+\.\s*", "", problem.get("text", "")).strip()


def problem_block(problem):
    lines = [clean_problem_text(problem)]
    for letter in LETTERS:
        if letter in (problem.get("options") or {}):
            lines.append(f"{letter}) {problem['options'][letter]}")
    return "\n".join(lines)


def ask(llm, messages, max_tokens, timeout, stop=None, grammar=None):
    """One chat call to the small model. Returns (text, seconds)."""
    start = time.perf_counter()
    payload = {"messages": messages, "max_tokens": max_tokens, "temperature": 0, "seed": 42, "cache_prompt": True}
    if stop:
        payload["stop"] = stop
    if grammar:
        payload["grammar"] = grammar
    request = urllib.request.Request(llm.rstrip("/") + "/v1/chat/completions", data=json.dumps(payload).encode(),
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = json.loads(response.read())["choices"][0]["message"]["content"]
    return (text or "").strip(), time.perf_counter() - start


def kanga_messages(problem, lang):
    """The fixed start of the conversation: the same for every turn, so the server caches it.
    It holds no answer and no notes: the model can't give away what it does not know."""
    return [{"role": "system", "content": KANGA_SYSTEM + LANGUAGE.get(lang, "") + f"\n\nProblem:\n{problem_block(problem)}"}]


def warm(problem, lang, llm, timeout=90):
    """Send the start of the conversation once, so the server has read it before the student writes."""
    ask(llm, kanga_messages(problem, lang) + [{"role": "user", "content": "Hi!"}], max_tokens=1, timeout=timeout)


def question_grammar(allowed):
    """One question, at most about 140 characters, whose only numbers are in `allowed`."""
    numbers = sorted({n for n in allowed if n}, key=len, reverse=True)
    piece = r'[^0-9\n?.!<>]' + "".join(f' | "{n}"' for n in numbers)
    return 'root ::= first piece{8,140} "?"\nfirst ::= [^0-9\\n?.!<> ]\npiece ::= ' + piece.replace("\n", "\\n")


def clean_reply(text):
    text = re.sub(r"^\s*(kanga|tutor)\s*:\s*", "", (text or "").strip(), flags=re.I).strip().strip('"').strip()
    return re.sub(r"\s{2,}", " ", text)


def reply_problem(reply, problem, said, lang="eng"):
    """"" if Kanga's question can be shown, else the reason it is rejected. Checked in code:
    Kanga must not teach, so it may only use numbers from the problem or from the student."""
    if not reply or len(reply) < 8:
        return "empty"
    if words(reply) > MAX_REPLY_WORDS:
        return "too long"
    if reply[-1] not in ".?!":
        return "cut off"
    if "?" not in reply:
        return "asks nothing"
    if LETTER_SAID.search(reply):
        return "points at an option"
    if number_set(reply) - number_set(problem.get("text", "")) - number_set(said):
        return "uses a number the student did not say"
    value = str(problem["options"].get(problem["label"], "")).strip().lower()
    if not number_set(value) and len(value) >= 3 and value in reply.lower() and value not in said.lower():
        return "gives the answer"
    known = (problem.get("text", "") + " " + said).lower()
    for number in number_set(value):
        if number.isdigit() and int(number) <= 20:
            for names in NUMBER_WORDS.values():
                word = names[int(number)]
                if len(word) > 2 and re.search(rf"\b{word}\b", reply.lower()) and not re.search(rf"\b{word}\b", known):
                    return "gives the answer in words"
    return ""


# Cases where code knows exactly what to say: the model adds nothing there
FIXED = ("convinced", "repeat", "slip", "compute", "off", "method")


def kanga_reply(problem, lang, chat, analysis, llm, timeout=55, help_text=""):
    """Kanga's reply for this turn. chat is [{"who": "student" | "tutor", "text"}], the student's
    latest message last. help_text is a method for this kind of problem: when the student is
    stuck, Kanga passes it on ("My teacher once told me: ..."). Returns (reply, source): source
    is "model", "fixed" or "fixed: <why>"."""
    reply, source = _kanga_line(problem, lang, chat, analysis, llm, timeout)
    status, n = analysis["status"], analysis.get("explanations", 0)
    if status not in ("convinced", "slip", "compute"):  # stuck: more help, one step at a time
        if n == 3 and help_text:
            reply += " " + say(lang, "teacher", text=help_text)
        elif n >= 4 and n % 2 == 0:
            reply += " " + say(lang, "showhow")
    return reply, source


def _kanga_line(problem, lang, chat, analysis, llm, timeout):
    status = analysis["status"]
    asked = [turn["text"] for turn in chat if turn["who"] == "tutor"]
    values = {"said": analysis.get("said") or "", "expr": analysis.get("expr") or "",
              "stated": analysis.get("stated") or ""}
    fixed = say(lang, status, avoid=asked, **values)
    if status in FIXED:
        return fixed, "fixed"
    said_value = values["said"]
    opener = (OPEN.get(lang) or OPEN["eng"])[status].format(said=said_value)
    said = "\n".join(turn["text"] for turn in chat if turn["who"] == "student")
    messages = kanga_messages(problem, lang)
    for turn in chat[1:-1]:  # the intro is not part of what the model reads
        messages.append({"role": "user" if turn["who"] == "student" else "assistant", "content": turn["text"]})
    messages.append({"role": "user", "content": chat[-1]["text"] + f"\n\n({FOCUS[status]})"})
    allowed = NUMBER.findall(problem.get("text", "")) + NUMBER.findall(said)
    try:
        payload_grammar = question_grammar(allowed)
        text, _seconds = ask(llm, messages, max_tokens=50, timeout=timeout, grammar=payload_grammar)
    except Exception as error:
        return fixed, f"fixed: model {type(error).__name__}"
    question = clean_reply(text)
    why = reply_problem(question, problem, said, lang)
    reply = f"{opener} {question}"
    if not why and any(reply in line for line in asked):
        why = "asked that already"
    if why:
        if any(fixed in line for line in asked):  # never the same line twice: the student needs something new
            fixed = say(lang, "stuck")
        return fixed, "fixed: " + why
    return reply, "model"
