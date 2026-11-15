# math-matrix: the App Lab app

The teaching half of the copilot: a hint ladder and an adaptive trainer on a web page served by the
board, with the board's 8x13 LED matrix.
It is separate from `predict.py`, which the jury runs from the terminal.

## What it does

**The tutor** (`/`, `assets/index.html`): the student answers a Kangaroo problem A to E. Each time they
ask for help, the copilot climbs one rung of a hint ladder, and never jumps to the answer:

| Rung | What the student gets | Where it comes from | Cost |
| --- | --- | --- | --- |
| 1 | A similar, easier problem, already solved | M3Kang as a map: the nearest neighbour at a lower level | lookup |
| 2 | The topic, and a method for it | The problem's topic tag | lookup |
| 3 | One sentence from the small model | Stored in the bank, or generated live. A code check rejects any hint that contains the answer value, points at an option, or calculates | lookup, or one model call |
| 4 | The solution | Only after the three hints, and only when the student asks for it | lookup |

The next problem comes from the same map: solved alone, a neighbour one level up; solved with help, the
same level; not solved, the same topic one level down. The LED matrix shows `L` and the level, a tick or
a cross, a row of dots while the model writes, and one small bar per hint used. The student's history
(by topic: alone, with help, not yet) stays in `python/students.db` on the board.

**The game.** Four games, chosen on the start page. The first three play the same ladder:

| Game | Rules |
| --- | --- |
| Free practice | No end. The level follows the student. |
| Round of 5 | Five problems, then a summary: up to three stars per problem. |
| The climb | Three lives. A wrong answer, a skip or asking for the solution costs one. A hint costs none. |
| Teach Kanga | Learning by teaching. The student explains the problem to Kanga, a classmate who can't solve it, and the answer buttons open only when Kanga understands. No hint ladder. |

**Teach Kanga** (`python/teachback.py`). Explaining a solution to someone else is one of the best ways
to learn it. The student writes how they would solve the problem. Kanga answers with a sentence about
what it noticed and one question about the step that is missing, unclear or wrong, and the student
answers it, adds a step or fixes one, until Kanga says "Now I get it". Then they pick the answer.

| Part | Who does it | Cost on the board |
| --- | --- | --- |
| Read the problem while the student thinks, so it is in the prompt cache | the 0.8B, one call of one token | in the background |
| Is the explanation good enough? It must name one of the options, the right one, be long enough to follow, and on the first try go through the key numbers of the teacher's solution in the bank, when there is one | code, never the model | none |
| Kanga's first sentence ("You get 8, but something doesn't add up for me.") | code | none |
| Kanga's question | the 0.8B, about 20 tokens, under a grammar: one question, no numbers but those of the problem or the student | a few seconds |
| Check the question: no answer, in figures or in words, and no letter | code; if it fails, a fixed question for that case | none |

The model is never told the answer: told it, the 0.8B wrote it into its reply at once. It can't judge an
explanation either, so code does. With no model at all the mode still works, with fixed questions. After
two explanations the student can ask "Show me how", which reveals the solution and counts as not solved.
Each explanation after the first counts like a hint for the points, so explaining it well the first time
earns three stars. `python3 tools/teachback_check.py` checks the code side without a model.

The points reward thinking over guessing (`score()` in `tutor.py`): 100 for a solved problem, minus 15
per hint and 25 per wrong answer (never below 10), plus 10 per level and up to 50 for problems solved in
a row. A hint is cheaper than a wrong guess on purpose. A problem that took more than one wrong answer
breaks the row, so clicking through the options does not pay. Three stars: right at the first try with
at most the first hint; two: with help; one: after several wrong answers. Points add up to five ranks
(Joey to Super Kangaroo) and a table of the players on this board. The matrix shows a star for a
three-star answer. Points and the game played are stored per attempt in `students.db`.

**The matrix page** (`/draw.html`): write text or draw pixels on the 8x13 matrix.

## Layout

```text
app.yaml                 name and bricks (arduino:web_ui)
python/main.py           runs on Linux in App Lab's container: web API, matrix frames
python/tutor.py          the hint ladder, the leak check and the adaptive trainer (standard library only)
python/teachback.py      Teach Kanga: when the explanation is good enough, and what Kanga asks
python/dev_server.py     the same page and API on a laptop, without App Lab
python/teach.db          the bank, built by laptop/build_teach_bank.py (M3Kang text: not in Git)
python/setup_bridge.sh   one-time board setup so the app can reach the model (see below)
assets/index.html        the tutor page (port 7000)
assets/draw.html         text and pixels on the matrix
sketch/sketch.ino        runs on the microcontroller: draws frames received over the Bridge
sketch/sketch.yaml       board platform
```

## Build the bank

From the repository root, with `data/train.jsonl` and `data/val.jsonl` prepared (SETUP.md step 9, then `laptop/canonicalize.py`)
and the language model server running:

```bash
python3 laptop/build_teach_bank.py --hints
```

It keeps the text-only problems with five clean options, builds the similarity map, and asks the 0.8B
model for one hint per problem. Hints that fail the code check are not stored. It is resumable, and it
runs on the board too (about 20 seconds per hint there), so the bank can be built with nothing but the
board. Then copy `python/teach.db` to the board with the rest of the app.

Try the page on a laptop:

```bash
python3 app/math-matrix/python/dev_server.py --port 7000
```

Measure it (leak rate, rung 1 quality, time per rung, simulated students):

```bash
python3 tools/teach_report.py --sample 20
```

## Install on a board

1. Copy this folder to `/home/arduino/ArduinoApps/math-matrix/` on the board, or create an app
   called `math-matrix` in App Lab, add the **WebUI - HTML** brick, and paste in the files.
2. Run it from App Lab (or `arduino-app-cli app start user:math-matrix`).
3. Open `http://<board-ip>:7000`, or over USB: `adb forward tcp:7000 tcp:7000` then `http://127.0.0.1:7000`.

## Reaching the model from the app

App Lab runs `main.py` inside a Docker container, where `127.0.0.1` is the container, not the board,
so the app can't reach the model servers (`math-llm` on 127.0.0.1:8081) directly.

- `main.py` finds the board's address as the container's default gateway (for example `172.19.0.1`).
- `python/setup_bridge.sh` installs the `llm-bridge` service, which forwards port 8081 on each Docker
  network gateway to `127.0.0.1:8081`. It listens only on the Docker networks, not on Wi-Fi, and
  starts at boot. Run it once per board:

  ```bash
  sudo bash ~/ArduinoApps/math-matrix/python/setup_bridge.sh
  ```

Check with `systemctl is-active llm-bridge` and `ss -ltn | grep 8081`.
