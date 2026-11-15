"""The teaching half of the copilot: a hint ladder and an adaptive trainer.

Standard library only, so the same file runs in App Lab's container on the
board (main.py), in the laptop dev server (dev_server.py) and in the
measurement tools.

The student gets a Kangaroo problem and answers A to E. Each time they ask for
help the copilot climbs ONE rung, and never jumps to the answer:

  1. a similar, easier problem from M3Kang, already solved   (lookup, instant)
  2. the topic and a method for it                           (lookup, instant)
  3. a one-sentence hint from the small language model, checked in code so it
     cannot contain the answer                    (stored, or generated live)
  4. the solution, only when the student asks for it on purpose

The next problem is chosen from the M3Kang map (similarity, topic, level):
right answer -> a neighbour one level up; needed help -> same level;
gave up -> same topic, one level down.

It is also a game, with rules that reward thinking over guessing (see score()):
a hint costs fewer points than a wrong guess. Four modes: "train" (endless
practice), "round" (five problems, then stars), "climb" (three lives: how
high can you get?) and "teach" (learning by teaching: the student explains the
problem to Kanga, a classmate who can't solve it, and can answer only once
Kanga understands; see teachback.py).

The bank (teach.db) is built by laptop/build_teach_bank.py. The student's
history stays in a second SQLite file on the board: nothing leaves the device.
"""
import base64
import json
import os
import random
import re
import sqlite3
import threading
import time
import urllib.request

import teachback

LETTERS = "ABCDE"
HERE = os.path.dirname(os.path.abspath(__file__))
BANK_PATH = os.environ.get("TUTOR_BANK", os.path.join(HERE, "teach.db"))
STATE_PATH = os.environ.get("TUTOR_STATE", os.path.join(HERE, "students.db"))
TWIN_CAP = 0.9      # above this similarity two problems are the same problem: showing one gives the other away
MIN_SIM = 0.12      # below this a "similar" problem is not similar
MAX_HINT_WORDS = 40
ROWS, COLS = 8, 13
MODES = ("train", "round", "climb", "teach")
ROUND_SIZE = 5      # problems in one round
LIVES = 3           # wrong answers allowed in a climb
RANKS = (0, 300, 800, 1500, 3000)   # total points for each rank; the page has the names


# ---------------------------------------------------------------- the game: points and stars

def score(hints, wrong, level, streak):
    """Points for a solved problem, as parts that add up to "total".

    100 for solving it, minus 15 per hint and 25 per wrong answer (never below 10),
    plus 10 per level and 10 per problem already solved in a row (at most 50).
    A hint is cheaper than a wrong guess on purpose: asking beats guessing.
    A problem only counts for the row when it took at most one wrong answer,
    so clicking through the options breaks the row (see Tutor._finish).
    """
    hint_cost = 15 * min(hints, 3)
    wrong_cost = min(25 * wrong, 90 - hint_cost)
    parts = {"solved": 100, "hints": -hint_cost, "wrong": -wrong_cost,
             "level": 10 * level, "streak": 10 * min(max(streak, 0), 5)}
    parts["total"] = sum(parts.values())
    return parts


def stars(outcome, wrong):
    """3: first try, at most the first hint. 2: with help. 1: after several wrong answers. 0: not yet."""
    if outcome == "not_yet":
        return 0
    return 3 if outcome == "alone" else 2 if wrong <= 1 else 1


def rank(total):
    """(the rank, 0 to 4; the points where the next rank starts, or None at the top)."""
    index = max(i for i, need in enumerate(RANKS) if total >= need)
    return index, RANKS[index + 1] if index + 1 < len(RANKS) else None

# ---------------------------------------------------------------- rung 2: methods

METHODS = {
    "arithmetic": {
        "eng": ("Arithmetic", "Write the calculation step by step. Estimate the size of the result first: which options can you rule out?"),
        "cat": ("Aritmètica", "Escriu el càlcul pas a pas. Primer estima la mida del resultat: quines opcions pots descartar?")},
    "number_theory": {
        "eng": ("Numbers and digits", "Try small cases and look for a pattern. Think about divisibility, remainders, odd and even, and the last digit."),
        "cat": ("Nombres i xifres", "Prova casos petits i busca un patró. Pensa en la divisibilitat, els residus, parells i senars, i l'última xifra.")},
    "measurement_time": {
        "eng": ("Measures, time and money", "Put everything in the same unit first. Then draw a timeline or a small table with what you know."),
        "cat": ("Mesures, temps i diners", "Primer posa-ho tot en la mateixa unitat. Després dibuixa una línia del temps o una taula amb el que saps.")},
    "plane_geometry": {
        "eng": ("Plane geometry", "Draw the figure and mark everything you know. Look for equal lengths, equal angles, and shapes you can cut or move."),
        "cat": ("Geometria plana", "Dibuixa la figura i marca-hi tot el que saps. Busca longituds iguals, angles iguals i formes que puguis tallar o moure.")},
    "solid_geometry": {
        "eng": ("Solids", "Count faces, edges and corners in order, layer by layer. A quick sketch of each layer helps."),
        "cat": ("Cossos geomètrics", "Compta cares, arestes i vèrtexs amb ordre, capa per capa. Un esbós de cada capa ajuda.")},
    "combinatorics": {
        "eng": ("Counting", "List the cases in order so you don't miss or repeat any. Start with a smaller version of the problem."),
        "cat": ("Comptar casos", "Fes la llista dels casos amb ordre per no deixar-te'n cap ni repetir-ne. Comença amb una versió més petita.")},
    "logic": {
        "eng": ("Logic", "Suppose one statement is true and see what follows. If you reach a contradiction, it was false."),
        "cat": ("Lògica", "Suposa que una afirmació és certa i mira què en resulta. Si arribes a una contradicció, era falsa.")},
    "grids_tables": {
        "eng": ("Grids and tables", "Start with the row or column that has the most information, and fill in what is forced."),
        "cat": ("Quadrícules i taules", "Comença per la fila o columna amb més informació i omple el que és obligat.")},
    "algebra": {
        "eng": ("Algebra", "Give the unknown a name and write what the problem says as an equation. Or test the options one by one."),
        "cat": ("Àlgebra", "Posa nom a la incògnita i escriu el que diu el problema com una equació. O prova les opcions una per una.")},
    "sequences_patterns": {
        "eng": ("Patterns", "Write the first terms and look at how each one changes to the next. Does something repeat?"),
        "cat": ("Patrons", "Escriu els primers termes i mira com canvia cadascun al següent. Hi ha alguna cosa que es repeteix?")},
    "spatial_visual": {
        "eng": ("Shapes in space", "Imagine turning or folding the shape one step at a time, and follow one corner or mark."),
        "cat": ("Formes a l'espai", "Imagina que gires o plegues la forma pas a pas, i segueix un sol vèrtex o marca.")},
    "graphs_paths": {
        "eng": ("Paths", "Mark each point with the number of ways to reach it, starting from the beginning."),
        "cat": ("Camins", "Marca a cada punt de quantes maneres s'hi pot arribar, començant pel principi.")},
    "probability": {
        "eng": ("Chance", "Count all the equally likely outcomes, then count the ones you want."),
        "cat": ("Probabilitat", "Compta tots els resultats igual de probables, i després els que t'interessen.")},
    "other": {
        "eng": ("Problem solving", "What is asked, and what do you know? Try a smaller example, or work backwards from the options."),
        "cat": ("Resolució de problemes", "Què et demanen i què saps? Prova un exemple més petit, o treballa enrere des de les opcions.")},
}


def method(topic, lang):
    entry = METHODS.get(topic) or METHODS["other"]
    return entry.get(lang) or entry["eng"]


# ---------------------------------------------------------------- the leak check

NUMBER = re.compile(r"\d+(?:[.,]\d+)?")
POINTS_AT_OPTION = re.compile(
    r"\b(answer|option|choice|resposta|opci[oó]|soluci[oó]|solution)\b[^.]{0,15}?\b(is|és|es|:)?\s*\(?[A-E]\)?(?![a-z])",
    re.I)


def _numbers(text):
    found = set()
    for n in NUMBER.findall(text or ""):
        n = n.replace(",", ".")
        found.add(n.rstrip("0").rstrip(".") if "." in n else n)  # 12,50 and 12.5 are the same number
    return found


def leak(hint, problem):
    """Why this hint gives the answer away, or "" if it does not.

    The rule: the value of the correct option must not appear in the hint,
    unless the problem text itself already contains it. A hint must not point
    at a letter either. Checked in code, so no model is trusted with it.
    """
    options = problem.get("options") or {}
    value = (options.get(problem.get("label", "")) or "").strip()
    if POINTS_AT_OPTION.search(hint):
        return "points at an option"
    if not value:
        return ""
    text = problem.get("text", "")
    value_numbers = _numbers(value)
    if value_numbers:
        new = value_numbers - _numbers(text)
        if new and new <= _numbers(hint):
            return "contains the answer value"
        return ""
    norm = re.sub(r"\s+", " ", value.lower())
    if len(norm) >= 3 and norm in hint.lower() and norm not in text.lower():
        return "contains the answer text"
    return ""


def verifiable(problem):
    """True when the leak check can tell: the answer value is not already in the problem text."""
    options = problem.get("options") or {}
    value = (options.get(problem.get("label", "")) or "").strip()
    numbers = _numbers(value)
    if numbers:
        return bool(numbers - _numbers(problem.get("text", "")))
    return bool(value) and value.lower() not in problem.get("text", "").lower()


def clean_hint(text):
    text = (text or "").strip().split("\n")[0].strip().strip('"').strip()
    return re.sub(r"^(hint|pista)\s*:\s*", "", text, flags=re.I)


def hint_problem(hint, problem):
    """"" if the hint can be shown, else the reason it is rejected."""
    if not hint or len(hint) < 12:
        return "empty"
    if len(hint.split()) > MAX_HINT_WORDS:
        return "too long"
    if hint[-1] not in ".?!":
        return "cut off"
    reason = leak(hint, problem)
    if reason:
        return reason
    # A hint guides, it does not calculate: the small model is often wrong when it
    # computes, and a wrong worked step misleads. Every number must come from the problem.
    if "=" in hint or _numbers(hint) - _numbers(problem.get("text", "")):
        return "calculates"
    return ""


# ---------------------------------------------------------------- rung 3: the model

HINT_SYSTEM = ("You are a patient math tutor for school students. Write ONE short hint, at most 20 words: "
               "a guiding question or the first thing to look at. Do not calculate anything, do not solve the "
               "problem, never give the answer and never mention the options.")
HINT_LANGUAGE = {"cat": " Write the hint in Catalan."}


def hint_messages(problem):
    lang = problem.get("language", "eng")
    return [{"role": "system", "content": HINT_SYSTEM + HINT_LANGUAGE.get(lang, "")},
            {"role": "user", "content": "Problem: " + re.sub(r"^\s*\d+\.\s*", "", problem["text"]).strip() + "\nHint:"}]


def generate_hint(problem, llm, timeout=55):
    """Ask the small model for one sentence. Returns (hint, reason rejected or "", seconds)."""
    start = time.perf_counter()
    body = json.dumps({"messages": hint_messages(problem), "max_tokens": 48, "temperature": 0, "seed": 42,
                       "cache_prompt": True, "stop": ["\n"]}).encode()
    request = urllib.request.Request(llm.rstrip("/") + "/v1/chat/completions", data=body,
                                     headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(request, timeout=timeout) as response:
        text = json.loads(response.read())["choices"][0]["message"]["content"]
    hint = clean_hint(text)
    return hint, hint_problem(hint, problem), time.perf_counter() - start


# ---------------------------------------------------------------- the bank

class Bank:
    """M3Kang as a map: problems by language, their neighbours, stored hints and explanations."""

    def __init__(self, path=BANK_PATH):
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} not found: build it with laptop/build_teach_bank.py")
        db = sqlite3.connect(path)
        self.problems = {}   # (id, lang) -> problem
        for pid, lang, level, year, topic, text, options, label in db.execute(
                "SELECT id, lang, level, year, topic, text, options, label FROM problems"):
            self.problems[pid, lang] = {"id": pid, "language": lang, "level": level, "year": year, "topic": topic,
                                        "text": text, "options": json.loads(options), "label": label}
        self.neighbours = {}  # id -> [(score, other id)], best first
        for pid, other, score in db.execute("SELECT id, nb_id, score FROM neighbours ORDER BY id, score DESC"):
            self.neighbours.setdefault(pid, []).append((score, other))
        self.hints = {(pid, lang): hint for pid, lang, hint in db.execute("SELECT id, lang, hint FROM hints")}
        self.explanations = {(pid, lang): text for pid, lang, text in
                             db.execute("SELECT id, lang, text FROM explanations")}
        db.close()
        self.languages = sorted({lang for _pid, lang in self.problems})

    def get(self, pid, lang):
        return self.problems.get((pid, lang))

    def of(self, lang):
        return [p for (_pid, l), p in self.problems.items() if l == lang]

    def easier_example(self, problem):
        """Rung 1: the most similar solved problem at a lower level (the same level if there is none).

        Returns (example, info) or (None, {}). Never a twin of the problem itself.
        """
        lang, level, topic = problem["language"], problem["level"], problem["topic"]
        found = []
        for score, other in self.neighbours.get(problem["id"], []):
            example = self.get(other, lang)
            if example is None or score >= TWIN_CAP or score < MIN_SIM:
                continue
            if example["level"] > level:
                continue
            found.append(((example["level"] < level, example["topic"] == topic, score), score, example))
        if not found:
            return None, {}
        _key, score, example = max(found, key=lambda item: item[0])
        return example, {"score": round(score, 3), "easier": example["level"] < level,
                         "same_topic": example["topic"] == topic}

    def pick(self, lang, level, topic=None, near=None, seen=(), stay_on_topic=False, rng=random):
        """The next problem: at `level`, close to problem id `near` on the map, on `topic` if it can be."""
        level = max(0, min(7, level))
        seen = set(seen)
        pool = [p for p in self.of(lang) if p["id"] not in seen]
        if not pool:
            pool = self.of(lang)
        if near:
            for _score, other in self.neighbours.get(near, []):
                p = self.get(other, lang)
                if p and p["id"] not in seen and p["level"] == level and _score < TWIN_CAP \
                        and (p["topic"] == topic or not stay_on_topic):
                    return p, "neighbour"
        on_topic = [p for p in pool if p["topic"] == topic]
        at_level = [p for p in on_topic if p["level"] == level]
        if at_level:
            return rng.choice(at_level), "topic+level"
        close = [p for p in on_topic if abs(p["level"] - level) == 1]
        if stay_on_topic and close:  # the topic matters, but never at the price of the level
            return rng.choice(close), "topic"
        at_level = [p for p in pool if p["level"] == level]
        if at_level:
            return rng.choice(at_level), "level"
        best = min(abs(p["level"] - level) for p in pool)
        return rng.choice([p for p in pool if abs(p["level"] - level) == best]), "nearest"


# ---------------------------------------------------------------- the student's history

class History:
    """Attempts per student, in SQLite on the board. Falls back to memory if the file can't be written."""

    def __init__(self, path=STATE_PATH):
        self.lock = threading.Lock()
        try:
            self.db = sqlite3.connect(path, check_same_thread=False)
            self._create()
        except sqlite3.Error:
            self.db = sqlite3.connect(":memory:", check_same_thread=False)
            self._create()

    def _create(self):
        self.db.execute("CREATE TABLE IF NOT EXISTS attempts (student TEXT, id TEXT, lang TEXT, level INTEGER, "
                        "topic TEXT, outcome TEXT, tries INTEGER, rungs INTEGER, seconds REAL, ts REAL)")
        for column in ("points INTEGER DEFAULT 0", "mode TEXT DEFAULT 'train'"):
            try:  # a history file from before the game has no such columns
                self.db.execute("ALTER TABLE attempts ADD COLUMN " + column)
            except sqlite3.OperationalError:
                pass
        self.db.commit()

    def add(self, student, problem, outcome, tries, rungs, seconds, points=0, mode="train"):
        with self.lock:
            self.db.execute("INSERT INTO attempts (student, id, lang, level, topic, outcome, tries, rungs, seconds, "
                            "ts, points, mode) VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                            (student, problem["id"], problem["language"], problem["level"], problem["topic"],
                             outcome, tries, rungs, round(seconds, 1), time.time(), points, mode))
            self.db.commit()

    def total(self, student):
        with self.lock:
            return self.db.execute("SELECT COALESCE(SUM(points), 0) FROM attempts WHERE student = ?",
                                   (student,)).fetchone()[0]

    def streaks(self, student):
        """(problems solved in a row right now, the longest such run ever). At most one wrong answer each."""
        now = best = 0
        for row in self.rows(student):
            now = now + 1 if row[3] != "not_yet" and row[4] <= 2 else 0
            best = max(best, now)
        return now, best

    def board(self, limit=5):
        """The students on this board with the most points: [(name, points, problems solved)]."""
        with self.lock:
            return self.db.execute("SELECT student, SUM(points), SUM(outcome != 'not_yet') FROM attempts "
                                   "GROUP BY student ORDER BY 2 DESC, 3 DESC LIMIT ?", (limit,)).fetchall()

    def rows(self, student):
        with self.lock:
            return self.db.execute("SELECT id, level, topic, outcome, tries, rungs FROM attempts "
                                   "WHERE student = ? ORDER BY ts", (student,)).fetchall()

    def profile(self, student):
        """What the student and the teacher see: by topic, solved alone / with help / not yet."""
        topics, rows = {}, self.rows(student)
        for _pid, _level, topic, outcome, _tries, _rungs in rows:
            entry = topics.setdefault(topic, {"alone": 0, "helped": 0, "not_yet": 0})
            entry[outcome] += 1
        return {"attempts": len(rows), "topics": topics, "last_level": rows[-1][1] if rows else None}


# ---------------------------------------------------------------- the LED matrix

GLYPHS = {
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.", "1": "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####", "3": "##### ...#. ..#.. ...#. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.", "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.", "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..", "L": "#.... #.... #.... #.... #.... #.... #####",
    "tick": "..... ....# ...#. ...#. #.#.. .#... .....", "cross": "..... #...# .#.#. ..#.. .#.#. #...# .....",
    "star": "..#.. ..#.. ##### .###. .###. #...# .....",
    "dots": "..... ..... ..... #.#.# ..... ..... .....",
}


def matrix(left, level, rung=0):
    """An 8x13 grid, brightness 0 to 7: a symbol on the left, the level digit on the right,
    and on the bottom row one bar per hint rung used."""
    grid = [[0] * COLS for _ in range(ROWS)]
    for glyph, x0 in ((left, 1), (str(max(0, min(7, level))), 7)):
        for y, row in enumerate(GLYPHS[glyph].split()):
            for x, pixel in enumerate(row):
                if pixel == "#":
                    grid[y][x0 + x] = 7
    for r in range(min(rung, 4)):
        grid[7][1 + r * 3] = grid[7][2 + r * 3] = 3  # two dim pixels per rung, one dark between
    return grid


# ---------------------------------------------------------------- one student's session

class Tutor:
    """One student at a time: the state the web page shows, and the matrix frame that goes with it.

    Every public method returns state(); none of them blocks on the model. The live
    hint (rung 3 without a stored hint) runs in a thread and the page polls state().
    """

    def __init__(self, bank=None, history=None, llm="http://127.0.0.1:8081", on_matrix=None, seed=42):
        self.bank = bank or Bank()
        self.history = history or History()
        self.llm = llm
        self.on_matrix = on_matrix or (lambda grid: None)
        self.rng = random.Random(seed)
        self.lock = threading.RLock()
        self.student, self.lang, self.level = "", self.bank.languages[0], 0
        self.problem, self.hints, self.tries = None, [], []
        self.status, self.busy, self.seen, self.started = "idle", False, [], 0.0
        self.picked_by, self.last, self.grid = "", "", matrix("?", 0)
        # the game
        self.mode, self.points, self.streak, self.best_streak = "train", 0, 0, 0
        self.lives, self.results, self.over, self.gained, self.top_level = LIVES, [], False, None, 0
        self.teach = None   # the conversation with Kanga, in the "teach" mode

    # -- helpers
    def _draw(self, left):
        self.grid = matrix(left, self.level, len(self.hints))
        try:
            self.on_matrix(self.grid)
        except Exception:
            pass  # the matrix must never break the lesson

    def _helped(self):
        """How much help this problem took, counted like hints for the points: in the teach mode,
        every explanation after the first one."""
        if self.mode == "teach" and self.teach:
            return min(3, max(0, len(self.teach["explanations"]) - 1))
        return len([h for h in self.hints if h["rung"] <= 3])

    def _public(self, problem):
        return {"id": problem["id"], "level": problem["level"], "topic": problem["topic"], "year": problem["year"],
                "text": re.sub(r"^\s*\d+\.\s*", "", problem["text"]).strip(), "options": problem["options"]}

    def state(self):
        with self.lock:
            out = {"student": self.student, "lang": self.lang, "level": self.level, "status": self.status,
                   "busy": self.busy, "tries": list(self.tries), "hints": list(self.hints),
                   "rung": len(self.hints), "matrix": self.grid, "last": self.last,
                   "picked_by": self.picked_by, "languages": self.bank.languages,
                   "problem": self._public(self.problem) if self.problem else None}
            if self.problem and self.status in ("solved", "revealed"):
                out["label"] = self.problem["label"]
            if self.mode == "teach" and self.teach:
                t = self.teach
                out["teach"] = {"chat": [dict(turn) for turn in t["chat"]], "convinced": t["convinced"],
                                "status": t["status"], "explained": len(t["explanations"]),
                                "prepared": t["ready"].is_set(), "notes_source": t["notes_source"]}
            if self.student:
                out["profile"] = self.history.profile(self.student)
                total = self.history.total(self.student)
                index, need = rank(total)
                out["game"] = {"mode": self.mode, "points": self.points, "total": total, "rank": index,
                               "rank_from": RANKS[index], "rank_next": need, "streak": self.streak,
                               "best_streak": self.best_streak, "lives": self.lives, "max_lives": LIVES,
                               "results": list(self.results), "round_size": ROUND_SIZE, "over": self.over,
                               "gained": self.gained, "top_level": self.top_level,
                               "board": [list(row) for row in self.history.board()]}
            return out

    def _set_problem(self, problem, picked_by):
        self.problem, self.picked_by = problem, picked_by
        self.hints, self.tries, self.status, self.busy, self.gained = [], [], "open", False, None
        self.seen.append(problem["id"])
        self.started = time.perf_counter()
        self.teach = None
        if self.mode == "teach":
            self.teach = {"chat": [{"who": "tutor", "text": teachback.say(self.lang, "intro"), "source": "fixed"}],
                          "explanations": [], "status": "start", "convinced": False, "notes": "",
                          "notes_source": "", "ready": threading.Event()}
            threading.Thread(target=self._prepare, args=(problem, self.teach), daemon=True).start()
        self._draw("L")

    # -- the teach mode: Kanga's notes are prepared while the student reads
    def _prepare(self, problem, teach):
        notes = self.bank.explanations.get((problem["id"], self.lang), "")
        source = "bank" if notes else "none"
        try:  # the problem, in the server's cache before the student writes
            teachback.warm(problem, self.lang, self.llm)
        except Exception:
            pass
        with self.lock:
            teach["notes"], teach["notes_source"] = notes, source
        teach["ready"].set()

    def explain(self, text):
        """The student's explanation, or their answer to Kanga's last question."""
        with self.lock:
            text = (text or "").strip()[:teachback.MAX_CHARS]
            teach = self.teach
            if self.mode != "teach" or not teach or self.status != "open" or self.busy or teach["convinced"] \
                    or len(text) < 2:
                return self.state()
            teach["chat"].append({"who": "student", "text": text})
            teach["explanations"].append(text)
            self.busy = True
            self._draw("dots")
            threading.Thread(target=self._reply, args=(self.problem, teach), daemon=True).start()
            return self.state()

    def _reply(self, problem, teach):
        begin = time.perf_counter()
        teach["ready"].wait(teachback.NOTES_WAIT_S)
        with self.lock:
            chat, notes, lang = [dict(turn) for turn in teach["chat"]], teach["notes"], self.lang
        analysis = teachback.analyse(problem, teach["explanations"], notes)
        try:
            reply, source = teachback.kanga_reply(problem, lang, chat, analysis, self.llm)
        except Exception as error:  # Kanga must never break the lesson
            reply, source = teachback.say(lang, "stuck"), f"fixed: {type(error).__name__}"
        with self.lock:
            if self.teach is not teach:
                return  # the student moved on
            teach["chat"].append({"who": "tutor", "text": reply, "source": source, "status": analysis["status"],
                                  "ms": round(1000 * (time.perf_counter() - begin))})
            teach["status"] = analysis["status"]
            teach["convinced"] = analysis["status"] == "convinced"
            self.busy = False
            self._draw("tick" if teach["convinced"] else "L")

    # -- actions
    def start(self, student, lang, level, mode="train"):
        with self.lock:
            self.student = re.sub(r"[^\w-]", "", student)[:24] or "student"
            self.lang = lang if lang in self.bank.languages else self.bank.languages[0]
            self.level = max(0, min(7, int(level)))
            self.seen = [row[0] for row in self.history.rows(self.student)]
            self.last = ""
            self.mode = mode if mode in MODES else "train"
            self.points, self.lives, self.results, self.over, self.top_level = 0, LIVES, [], False, self.level
            self.streak, self.best_streak = self.history.streaks(self.student)
            problem, how = self.bank.pick(self.lang, self.level, seen=self.seen, rng=self.rng)
            self._set_problem(problem, how)
            return self.state()

    def answer(self, letter):
        with self.lock:
            letter = (letter or "").upper()
            if self.status != "open" or letter not in LETTERS or len(letter) != 1 or letter in self.tries:
                return self.state()
            if self.mode == "teach" and not (self.teach and self.teach["convinced"]):
                return self.state()  # first explain it to Kanga
            self.tries.append(letter)
            if letter == self.problem["label"]:
                self.status = "solved"
                alone = len(self.tries) == 1 and self._helped() <= 1
                self._finish("alone" if alone else "helped")
                self._draw("star" if alone else "tick")
            else:
                if self.mode == "climb":
                    self.lives -= 1
                    if self.lives <= 0:     # out of lives: the climb ends here, with the answer shown
                        self.status = "revealed"
                        self._finish("not_yet")
                self._draw("cross")
            return self.state()

    def _finish(self, outcome):
        """Close the problem: points, streak, stars, the history, and whether the game is over."""
        self.last = outcome
        solved = outcome != "not_yet"
        wrong = len(self.tries) - solved
        rungs = self._helped()
        self.gained = score(rungs, wrong, self.problem["level"], self.streak) if solved else None
        points = self.gained["total"] if solved else 0
        self.points += points
        self.streak = self.streak + 1 if solved and wrong <= 1 else 0   # guessed through: the row breaks
        self.best_streak = max(self.best_streak, self.streak)
        if solved:
            self.top_level = max(self.top_level, self.problem["level"])
        self.results.append(stars(outcome, wrong))
        self.over = (self.mode == "round" and len(self.results) >= ROUND_SIZE) or \
                    (self.mode == "climb" and self.lives <= 0)
        used = len(self.teach["explanations"]) if self.mode == "teach" and self.teach else len(self.hints)
        self.history.add(self.student, self.problem, outcome, len(self.tries), used,
                         time.perf_counter() - self.started, points, self.mode)

    def hint(self):
        """Climb one rung. Rung 4, the solution, has its own action: solution()."""
        with self.lock:
            if self.status != "open" or self.busy or len(self.hints) >= 3 or self.mode == "teach":
                return self.state()
            rung, problem = len(self.hints) + 1, self.problem
            begin = time.perf_counter()
            if rung == 1:
                example, info = self.bank.easier_example(problem)
                if example:
                    self.seen.append(example["id"])  # its answer is on screen: never ask it later
                    entry = {"rung": 1, "kind": "example", "problem": self._public(example),
                             "label": example["label"], "answer": example["options"][example["label"]],
                             "explanation": self.bank.explanations.get((example["id"], self.lang), ""), **info}
                else:
                    entry = {"rung": 1, "kind": "example", "problem": None}
            elif rung == 2:
                name, text = method(problem["topic"], self.lang)
                entry = {"rung": 2, "kind": "method", "topic": name, "text": text}
            else:
                stored = self.bank.hints.get((problem["id"], self.lang))
                if stored and not hint_problem(stored, problem):
                    entry = {"rung": 3, "kind": "hint", "text": stored, "source": "stored"}
                else:
                    self.busy = True
                    self.hints.append({"rung": 3, "kind": "hint", "text": "", "source": "pending"})
                    self._draw("dots")
                    threading.Thread(target=self._live_hint, args=(problem,), daemon=True).start()
                    return self.state()
            entry["ms"] = round(1000 * (time.perf_counter() - begin), 1)
            self.hints.append(entry)
            self._draw("L")
            return self.state()

    def _live_hint(self, problem):
        begin = time.perf_counter()
        try:
            text, rejected, _seconds = generate_hint(problem, self.llm)
            source = "rejected: " + rejected if rejected else "live"
        except Exception as error:
            text, source = "", f"unavailable: {type(error).__name__}"
        with self.lock:
            if self.problem is not problem:
                return  # the student moved on
            if source != "live":  # never show a hint that failed the check: fall back to a safe prompt
                text = ""
            self.hints[-1] = {"rung": 3, "kind": "hint", "text": text, "source": source,
                              "ms": round(1000 * (time.perf_counter() - begin), 1)}
            self.busy = False
            self._draw("L")

    def solution(self):
        """Rung 4: only when the student asks for it on purpose."""
        with self.lock:
            teaching = self.mode == "teach" and self.teach
            if self.status != "open" or self.busy:
                return self.state()
            if teaching and (len(self.teach["explanations"]) < 2 or self.teach["convinced"]):
                return self.state()  # explain it twice first; once Kanga understands, just answer
            if not teaching and len(self.hints) < 3:
                return self.state()  # the ladder is climbed one rung at a time
            problem = self.problem
            self.hints.append({"rung": 4, "kind": "solution", "label": problem["label"],
                               "answer": problem["options"][problem["label"]],
                               "explanation": self.bank.explanations.get((problem["id"], self.lang), "")})
            self.status = "revealed"
            if self.mode == "climb":
                self.lives -= 1
            self._finish("not_yet")
            self._draw("cross")
            return self.state()

    def next(self):
        with self.lock:
            if self.problem is None or self.over:
                return self.state()        # a finished game starts again with start()
            if self.status == "open":      # skipped: counts as not yet
                if self.mode == "climb":
                    self.lives -= 1
                self._finish("not_yet")
                if self.over:              # that skip ended the game: stay here, show the answer
                    self.status = "revealed"
                    self._draw("cross")
                    return self.state()
            step = {"alone": 1, "helped": 0, "not_yet": -1}[self.last]
            self.level = max(0, min(7, self.level + step))
            problem, how = self.bank.pick(self.lang, self.level, topic=self.problem["topic"],
                                          near=self.problem["id"], seen=self.seen,
                                          stay_on_topic=self.last == "not_yet", rng=self.rng)
            self._set_problem(problem, how)
            return self.state()

    # -- one entry point for both servers
    def handle(self, parts):
        """parts is the path after /api/, split on "/"."""
        action, args = (parts[0] if parts else "state"), parts[1:]
        if action == "start" and len(args) in (3, 4):   # the fourth is the game mode
            return self.start(*args)
        if action == "answer" and len(args) == 1:
            return self.answer(args[0])
        if action == "explain" and len(args) == 1:   # the text as base64url, so it holds no "/"
            try:
                raw = args[0][:4 * teachback.MAX_CHARS]
                text = base64.urlsafe_b64decode(raw + "=" * (-len(raw) % 4)).decode("utf-8", "replace")
            except (ValueError, TypeError):
                return {"error": "unreadable explanation"}
            return self.explain(text)
        if action in ("hint", "solution", "next", "state") and not args:
            return getattr(self, action)()
        return {"error": "unknown request"}
