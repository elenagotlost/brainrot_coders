import json
import socket
import struct
import threading
import time
import urllib.request

import numpy as np
from arduino.app_bricks.web_ui import WebUI
from arduino.app_utils import App, Bridge, Frame


def default_gateway():
    """The board, as seen from inside App Lab's container: the container's default gateway."""
    try:
        with open("/proc/net/route") as routes:
            for line in routes.readlines()[1:]:
                fields = line.split()
                if fields[1] == "00000000":
                    return socket.inet_ntoa(struct.pack("<L", int(fields[2], 16)))
    except OSError:
        pass
    return "127.0.0.1"


LLM = f"http://{default_gateway()}:8081"
ROWS, COLS = 8, 13
SCROLL_STEP = 0.12  # seconds per column when text scrolls

# 5x7 font: seven rows per character, top to bottom
FONT_SRC = {
    "A": ".###. #...# #...# ##### #...# #...# #...#",
    "B": "####. #...# #...# ####. #...# #...# ####.",
    "C": ".###. #...# #.... #.... #.... #...# .###.",
    "D": "####. #...# #...# #...# #...# #...# ####.",
    "E": "##### #.... #.... ####. #.... #.... #####",
    "F": "##### #.... #.... ####. #.... #.... #....",
    "G": ".###. #...# #.... #.### #...# #...# .###.",
    "H": "#...# #...# #...# ##### #...# #...# #...#",
    "I": ".###. ..#.. ..#.. ..#.. ..#.. ..#.. .###.",
    "J": "..### ...#. ...#. ...#. ...#. #..#. .##..",
    "K": "#...# #..#. #.#.. ##... #.#.. #..#. #...#",
    "L": "#.... #.... #.... #.... #.... #.... #####",
    "M": "#...# ##.## #.#.# #.#.# #...# #...# #...#",
    "N": "#...# #...# ##..# #.#.# #..## #...# #...#",
    "O": ".###. #...# #...# #...# #...# #...# .###.",
    "P": "####. #...# #...# ####. #.... #.... #....",
    "Q": ".###. #...# #...# #...# #.#.# #..#. .##.#",
    "R": "####. #...# #...# ####. #.#.. #..#. #...#",
    "S": ".#### #.... #.... .###. ....# ....# ####.",
    "T": "##### ..#.. ..#.. ..#.. ..#.. ..#.. ..#..",
    "U": "#...# #...# #...# #...# #...# #...# .###.",
    "V": "#...# #...# #...# #...# #...# .#.#. ..#..",
    "W": "#...# #...# #...# #.#.# #.#.# #.#.# .#.#.",
    "X": "#...# #...# .#.#. ..#.. .#.#. #...# #...#",
    "Y": "#...# #...# .#.#. ..#.. ..#.. ..#.. ..#..",
    "Z": "##### ....# ...#. ..#.. .#... #.... #####",
    "0": ".###. #...# #..## #.#.# ##..# #...# .###.",
    "1": "..#.. .##.. ..#.. ..#.. ..#.. ..#.. .###.",
    "2": ".###. #...# ....# ...#. ..#.. .#... #####",
    "3": "##### ...#. ..#.. ...#. ....# #...# .###.",
    "4": "...#. ..##. .#.#. #..#. ##### ...#. ...#.",
    "5": "##### #.... ####. ....# ....# #...# .###.",
    "6": "..##. .#... #.... ####. #...# #...# .###.",
    "7": "##### ....# ...#. ..#.. .#... .#... .#...",
    "8": ".###. #...# #...# .###. #...# #...# .###.",
    "9": ".###. #...# #...# .#### ....# ...#. .##..",
    " ": "..... ..... ..... ..... ..... ..... .....",
    "!": "..#.. ..#.. ..#.. ..#.. ..#.. ..... ..#..",
    "?": ".###. #...# ....# ...#. ..#.. ..... ..#..",
    ".": "..... ..... ..... ..... ..... ..... ..#..",
    ",": "..... ..... ..... ..... ..... ..#.. .#...",
    "-": "..... ..... ..... ##### ..... ..... .....",
    "+": "..... ..#.. ..#.. ##### ..#.. ..#.. .....",
    "=": "..... ..... ##### ..... ##### ..... .....",
    ":": "..... ..#.. ..... ..... ..... ..#.. .....",
    "*": "..... #.#.# .###. ##### .###. #.#.# .....",
}
FONT = {char: rows.split() for char, rows in FONT_SRC.items()}

# --- drawing ------------------------------------------------------------

bridge_lock = threading.Lock()
generation = [0]  # bumped on every new command, so an old scroll stops


def send(grid):
    with bridge_lock:
        Bridge.call("draw", Frame(grid).to_board_bytes())


def text_columns(text):
    """Turn text into a list of 8-pixel columns, with one blank column between characters."""
    columns = []
    for char in text.upper():
        rows = FONT.get(char, FONT["?"])
        for col in range(5):
            columns.append([1 if row < 7 and rows[row][col] == "#" else 0 for row in range(ROWS)])
        columns.append([0] * ROWS)
    return columns[:-1] if columns else columns


def grid_from_columns(columns, start=0):
    grid = np.zeros((ROWS, COLS), dtype=np.uint8)
    for x in range(COLS):
        i = start + x
        if 0 <= i < len(columns):
            for y in range(ROWS):
                if columns[i][y]:
                    grid[y, x] = 7
    return grid


def show_text(text):
    """Static and centred if it fits, otherwise scrolls in a loop until the next command."""
    generation[0] += 1
    my_gen = generation[0]
    columns = text_columns(text)
    if len(columns) <= COLS:
        pad = (COLS - len(columns)) // 2
        send(grid_from_columns([[0] * ROWS] * pad + columns))
        return

    def scroll():
        padded = [[0] * ROWS] * COLS + columns + [[0] * ROWS] * COLS
        while generation[0] == my_gen:
            for start in range(len(padded) - COLS + 1):
                if generation[0] != my_gen:
                    return
                send(grid_from_columns(padded, start))
                time.sleep(SCROLL_STEP)

    threading.Thread(target=scroll, daemon=True).start()


def show(char):
    show_text(char)


# --- web API ------------------------------------------------------------

state = {"busy": False, "message": f"Ready. (Model server: {LLM})"}


def write_text(text: str):
    text = text.strip()[:60]
    if not text:
        return {"error": "type something"}
    show_text(text)
    state["message"] = f"Showing: {text.upper()}"
    return {"ok": True}


def draw_pixels(bits: str):
    if len(bits) != ROWS * COLS or set(bits) - {"0", "1"}:
        return {"error": f"need {ROWS * COLS} characters of 0 or 1"}
    generation[0] += 1
    grid = np.array([7 if b == "1" else 0 for b in bits], dtype=np.uint8).reshape(ROWS, COLS)
    send(grid)
    state["message"] = "Showing your drawing."
    return {"ok": True}


def work(letter):
    """Runs in a thread: put your copilot here, such as checking the answer or asking for a hint."""
    show("?")
    try:
        with urllib.request.urlopen(LLM + "/health", timeout=5) as response:
            state["message"] = f"You chose {letter}. Model server {LLM}: {json.loads(response.read())}"
    except OSError as error:
        state["message"] = f"You chose {letter}. Model server {LLM} unreachable: {error}"
    finally:
        show(letter)
        state["busy"] = False


def answer(letter: str):
    letter = letter.upper()
    if letter not in "ABCDE" or len(letter) != 1:
        return {"error": "choose A to E"}
    if state["busy"]:
        return {"error": "still working"}
    state["busy"] = True
    threading.Thread(target=work, args=(letter,), daemon=True).start()
    return {"ok": True}


ui = WebUI()
ui.expose_api("GET", "/answer/{letter}", answer)
ui.expose_api("GET", "/text/{text}", write_text)
ui.expose_api("GET", "/pixels/{bits}", draw_pixels)
ui.expose_api("GET", "/status", lambda: state)

show("?")
App.run()
