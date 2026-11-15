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
import json
import re
import time
import urllib.request

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


def conclusion(problem, text):
    """(the option an explanation ends on, the value it ends on), or (None, None).

    In this order: a letter named as the answer ("so the answer is C"); else a number named as the
    answer ("the answer is 7": OFF when it is none of the options); else the last number in the
    text that is the value of exactly one option; else the option text that appears last."""
    named = list(LETTER_SAID.finditer(text or ""))
    if named:
        letter = named[-1].group(1)
        return letter, str((problem.get("options") or {}).get(letter, letter))
    values = option_numbers(problem)
    stated = list(ANSWER_IS.finditer(text or ""))
    if stated:
        raw = stated[-1].group(1)
        hits = [letter for letter, value in values.items() if value == _norm(raw)]
        return (hits[0], str(problem["options"][hits[0]])) if len(hits) == 1 else (OFF, raw)
    for number in reversed(number_list(text)):
        hits = [letter for letter, value in values.items() if value == number]
        if len(hits) == 1:
            return hits[0], str(problem["options"][hits[0]])
    lower, best, where = (text or "").lower(), None, -1
    for letter, value in (problem.get("options") or {}).items():
        value = str(value).strip().lower()
        if len(value) >= 3 and letter not in values:
            at = lower.rfind(value)
            if at > where:
                best, where = letter, at
    return (best, str(problem["options"][best])) if best else (None, None)


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


def analyse(problem, explanations, notes=""):
    """What the student's explanation so far still lacks. Returns a dict with "status":

      short      too short to follow, and no answer yet
      no_answer  it does not reach one of the options yet
      wrong      it ends on a wrong option ("said" is the value it ends on)
      gap        right answer, but it skips the key steps of the notes (first explanation only)
      check      right answer, but it can't be checked from the text alone (first explanation only)
      convinced  right answer, and explained: the student may answer now

    Never asks the model. After the student has explained twice, a right answer that is long
    enough is accepted even if it took another path than the notes: there is more than one
    way to solve a problem."""
    label = problem.get("label")
    everything = "\n".join(explanations)
    ends_on, value = None, None
    for text in reversed(explanations):  # the latest message first: it may answer an earlier question
        ends_on, value = conclusion(problem, text)
        if ends_on:
            break
    keys = key_numbers(problem, notes)
    covered = keys & number_set(everything)
    out = {"conclusion": ends_on, "words": words(everything), "keys": len(keys), "covered": len(covered),
           "explanations": len(explanations)}
    if ends_on is None:
        out["status"] = "short" if out["words"] < MIN_WORDS else "no_answer"
    elif ends_on != label:
        out["status"], out["said"] = "wrong", value
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

# The first sentence of Kanga's reply, from code. The model adds the question.
OPEN = {
    "eng": {"short": "I'm lost already.", "no_answer": "I'm following you so far.",
            "wrong": "Hmm, you get {said}, but something doesn't add up for me.",
            "gap": "You say the result, but I don't see how you get there.", "check": "That sounds possible."},
    "cat": {"short": "Ja m'he perdut.", "no_answer": "Fins aquí et segueixo.",
            "wrong": "Mmm, et surt {said}, però hi ha alguna cosa que no em quadra.",
            "gap": "Dius el resultat, però no veig com hi arribes.", "check": "Sembla que pot ser."},
}

SAY = {
    "eng": {
        "intro": "I have no idea how to solve this one. Can you explain it to me, step by step?",
        "short": "I don't follow yet. Where do you start, and what do you do first?",
        "no_answer": "I think I follow so far. But what comes out at the end, and which option is it?",
        "wrong": "You end with {said}, but I lost you on the way there. Can you go through that last step again?",
        "gap": "How do you get from the numbers in the problem to your result? Can you show me the middle step?",
        "check": "Okay, but why is that true? Can you show me how you checked it?",
        "convinced": "Now I get it, thanks! Go ahead and pick the answer.",
        "stuck": "I'm still lost. Maybe try a smaller example first?",
    },
    "cat": {
        "intro": "No tinc ni idea de com resoldre aquest. M'ho pots explicar pas a pas?",
        "short": "Encara no t'entenc. Per on comences, i què fas primer?",
        "no_answer": "Fins aquí crec que t'entenc. Però què surt al final, i quina opció és?",
        "wrong": "Acabes amb {said}, però m'he perdut pel camí. Pots tornar a fer l'últim pas?",
        "gap": "Com passes dels nombres del problema al teu resultat? Em pots ensenyar el pas del mig?",
        "check": "D'acord, però per què és així? M'ensenyes com ho has comprovat?",
        "convinced": "Ara ho entenc, gràcies! Ja pots triar la resposta.",
        "stuck": "Encara estic perdut. Potser prova primer amb un exemple més petit?",
    },
}


def say(lang, key, **values):
    text = (SAY.get(lang) or SAY["eng"]).get(key) or SAY["eng"][key]
    return text.format(**values)


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


def kanga_reply(problem, lang, chat, analysis, llm, timeout=55):
    """Kanga's reply for this turn. chat is [{"who": "student" | "tutor", "text"}], the student's
    latest message last. Returns (reply, source): source is "model", "fixed" or "fixed: <why>"."""
    status = analysis["status"]
    if status == "convinced":
        return say(lang, "convinced"), "fixed"
    said_value = analysis.get("said") or ""
    fixed = say(lang, status, said=said_value)
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
    asked = [turn["text"] for turn in chat if turn["who"] == "tutor"]
    if not why and reply in asked:
        why = "asked that already"
    if why:
        if fixed in asked:  # never the same line twice in a row: the student needs something new
            fixed = say(lang, "stuck")
        return fixed, "fixed: " + why
    return reply, "model"
