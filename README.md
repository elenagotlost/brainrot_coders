# Math Matrix: a Kangaroo math copilot on the Arduino UNO Q

Team **brainrot_coders**, Datathon FME 2026, Qualcomm challenge "Math on the Edge".

🥉 **3rd place overall** at the [Datathon FME 2026](https://www.datathon.cat/) (Barcelona, 14–15 November
2026), with **47% accuracy** on the organizers' hidden test set of unseen problems.

A math copilot that runs entirely on the Arduino UNO Q (4 GB), offline. It does both halves of the
challenge:

- **Teach.** A game where the student solves Kangaroo problems and climbs a ladder of hints that never
  gives the answer away. The board picks the next problem from the student's results and shows the
  level, hints and result on its LED matrix.
- **Reason.** `predict.py` answers a Kangaroo problem with one letter, A to E, in under 60 seconds. A
  small model writes a short Python program, the board runs it, and the answer counts only when the
  output equals exactly one option.

<!-- TODO: photo or short video of the board and the tutor page -->
<!-- TODO: link to the Arduino Project Hub write-up -->

| | Result |
| --- | --- |
| **Final ranking** | **3rd place overall** |
| **Reasoning, organizers' hidden test set (unseen problems)** | **47%** |
| Reasoning, 100 practice rows (new problems, English and Catalan) | **37%**, against 21% for the baseline model and 20% for guessing |
| Reasoning, 20-problem mock exam on the board, Wi-Fi off | 11 of 20, mean 32.1 s, worst 52.0 s |
| Teaching, hints shown that leak the answer | 0 of 873 |
| Teaching, time to show a stored hint | under 1 ms |

The organizers' guide to setting up a board (steps 0 to 15) is now in [SETUP.md](SETUP.md).

## Contents

- [Teach: the tutor](#teach-the-tutor)
- [Reason: the solver](#reason-the-solver)
- [Architecture](#architecture)
- [How we use M3Kang](#how-we-use-m3kang)
- [Results](#results)
- [Error analysis](#error-analysis)
- [What didn't work](#what-didnt-work)
- [Replicate it](#replicate-it)
- [Repository layout](#repository-layout)
- [Team](#team)
- [Credits and licenses](#credits-and-licenses)

## Teach: the tutor

The student opens a web page served by the board, picks a level and a game, and answers Kangaroo
problems A to E. Every time they ask for help, the copilot climbs one rung:

| Rung | What the student gets | Where it comes from | Cost on the board |
| --- | --- | --- | --- |
| 1 | A similar, easier problem, already solved | M3Kang as a map: the nearest neighbour at a lower level | lookup |
| 2 | The topic, and a method for it | The problem's topic tag | lookup |
| 3 | One sentence from the small model | Written in advance and stored, or generated live | lookup, or one model call |
| 4 | The solution | Only after the three hints, and only when the student asks twice | lookup |

**The model never decides what is safe to show.** A code check rejects any hint that contains the
answer value, points at an option, or does the calculation. Only hints that pass are stored or shown.

**The next problem follows the student.** Solved alone: a neighbour one level up. Solved with help: the
same level. Not solved: the same topic, one level down.

**It is a game.** Four ways to play:

| Game | Rules |
| --- | --- |
| Free practice | No end. The level follows the student. |
| Round of 5 | Five problems, then a summary with up to three stars per problem. |
| The climb | Three lives. A wrong answer, a skip or asking for the solution costs one. A hint costs none. |
| Teach Kanga | Learning by teaching. The student explains the problem to Kanga, a classmate who can't solve it. The answer buttons open only when Kanga understands. |

**Teach Kanga** turns the roles round. The student writes how they would solve the problem, and Kanga
asks one question about the step that is missing, unclear or wrong. Code, not the model, decides whether
the explanation is good enough: it reads the student's result, works out every calculation they wrote
("Wait, is 9 + 17 really 25?"), and checks the key steps. The 0.8B model only words Kanga's question,
under a grammar that allows no numbers but those of the problem or the student, and it is never told the
answer. For problems with a hand-written script (the question to ask for each wrong option and each
step), the model is not called at all. Details in [app/math-matrix/README.md](app/math-matrix/README.md).

The points reward thinking over guessing: 100 for a solved problem, minus 15 per hint and 25 per wrong
answer, plus a bonus for the level and for problems solved in a row. A hint is cheaper than a wrong
guess on purpose, and clicking through the options breaks the row. Points add up to five ranks, from
Joey to Super Kangaroo, and a table of the players on the board.

**The LED matrix** shows `L` and the level, a tick, a cross or a star, a row of dots while the model
writes, and one bar per hint used.

The page works in English and Catalan. The student's history stays in a file on the board. The rules
are in [app/math-matrix/README.md](app/math-matrix/README.md).

## Reason: the solver

The 0.8B model can't reason through a Kangaroo problem in the tokens that fit in 60 seconds. Asked for
a letter, it guesses (see [Results](#results)). But many Kangaroo problems are computations: count,
enumerate, last digit, divisibility. A few lines of brute force solve them, and Python does the
arithmetic the model gets wrong.

For each problem, `copilot/pipeline.py` runs the cheapest stage first:

0. **A lookup in the bank of solved problems** (`copilot/twins.py`), with no model. If the same problem
   is in the bank, in either language, its answer value is carried over to the matching option. It
   answers only on a near-exact match, so a new problem goes on to the next stages. The bank is every
   `.jsonl` file in `data/bank/` on the board, 3,798 solved rows: M3Kang (Kangaroo papers from 2007 to
   2024), the organizers' practice file, and 102 of the 227 questions of the Cangur 2026 papers (a
   random 45%: the team removed the other 125). None of it is in this repository.

   The code model then gives a second opinion on the looked-up answer (`coder.check()`). If it says
   no, the problem goes on to the next stages. This is not a proof: asked about the 100 practice
   problems, the model said yes to 52 right answers and to 51 wrong ones. `--no-check-lookup` switches
   the check off, and `--no-twins` switches the whole lookup off. See [The lookup](#the-lookup) for
   what it does to the scores.
1. **A prior over the five options**, with no model: the median value for numbers, the most central
   string for text. It is always ready, so there is always an answer.
2. **The code stage** (`copilot/coder.py`). Qwen3.5 2B writes a Python program of at most 10 lines. The
   board runs it in a separate process with a 5-second limit, a short list of allowed imports and no
   file or network access. The answer is accepted only when the output equals exactly one option.
3. **Otherwise the prior answers.**

The time is budgeted, not hoped for. On the board the 2B model reads 4.6 tokens per second and writes
2.6. Before each call, `budget()` works out how many tokens the model may write so that reading,
writing and running fit in the time left, with at most 96. The system prompt is read once at start-up
and cached. The run is deterministic: temperature 0 and a fixed seed.

## Architecture

```text
 Laptop or phone browser
        |  http, port 7000 (Wi-Fi, or USB with adb forward)
        v
+----------------------------- Arduino UNO Q, Linux (QRB2210) ----------------------------+
|                                                                                         |
|  App Lab container: app/math-matrix                Terminal: predict.py                 |
|    python/main.py   web API, matrix frames           copilot/pipeline.py                |
|    python/tutor.py  hint ladder, leak check,           prior -> code stage -> prior     |
|                     trainer, game                        |  asks for a program, then    |
|    teach.db         problems, neighbours, hints          |  runs it: python3 -I, 5 s    |
|    students.db      history and points                   |                              |
|        |                                                 |                              |
|        | llm-bridge (Docker network only)                |                              |
|        v                                                 v                              |
|  math-llm    Qwen3.5 0.8B, port 8081               math-code   Qwen3.5 2B, port 8091    |
|  (live hints)                                      (writes the program)                 |
|  math-embed  bge-small, port 8082 (retrieval experiments only)                          |
+----------------------|------------------------------------------------------------------+
                       |  Bridge
                       v
+---------------------------- Microcontroller (STM32) ------------------------------------+
|  sketch/sketch.ino: draws the frames it receives on the 8x13 LED matrix                 |
+-----------------------------------------------------------------------------------------+
```

- **On Linux:** three `llama.cpp` servers started at boot by systemd, `predict.py`, and the App Lab app.
  With all of them running, about 2.7 GB of the 4 GB is still available.
- **On the microcontroller:** only the sketch that draws frames. The Linux side decides what to show.
- **Nothing leaves the board.** The page has no web fonts, images or libraries from the internet.

## How we use M3Kang

- **A map for teaching.** Each problem is linked to its 40 most similar problems (character n-gram
  TF-IDF on the English text, computed in 1.3 seconds with no model). Rung 1 walks
  the map down one level to find an easier example, and the trainer walks it up or down to choose the
  next problem. A problem's own translation is never shown as its example.
- **A topic map.** M3Kang has no topic field, so `copilot/topics.py` tags each problem with one of 14
  topics from keywords. The tag gives rung 2 its method and keeps the trainer on topic after a miss.
- **A hint bank.** The 0.8B model wrote one hint for each of 1,151 problems in advance, and the code
  check kept 873. Showing one costs a lookup instead of 20 seconds of generation.
- **A multilingual bridge.** A problem has the same id in every language, so the topic, the neighbours
  and the level found in English also serve the Catalan student.
- **A benchmark.** All the numbers below.

No M3Kang data is in this repository: its license is for research use only. See the
[dataset card](https://huggingface.co/datasets/qualcomm/M3Kang).

## Results

### Reasoning

On the organizers' 100 practice rows (50 new problems, each in English and Catalan, levels 0 to 7,
20 rows with a figure). These problems are not in M3Kang.

| Approach | Accuracy |
| --- | --- |
| Random guess | 20% |
| Prior only, no model | 19% |
| Baseline: Qwen3.5 0.8B picks a letter | 21% |
| The same, asked 3 times with the options moved round | 18% |
| The same, asked 5 times | 21% |
| **Code stage with Qwen3.5 2B, then the prior** | **37%** |

The accuracy was measured on a laptop, with the same model file, prompt and 96-token limit as the
board. The board run below checks the timing.

On the board, Wi-Fi off, 20 of those rows chosen as a mock exam (10 English, 10 Catalan, 4 with a
figure):

| Measure | Value |
| --- | --- |
| Correct | 11 of 20 |
| Mean time per problem | 32.1 s |
| Worst time | 52.0 s |
| Over 60 s | 0 |
| Start-up, once (the server reads the system prompt) | 18.0 s |
| Start-up, once (the lookup builds its index of 4,048 solved problems; the bank now has 3,798) | 11.4 s |
| A lookup, per problem | 0.1 to 0.2 s |

Twenty problems is a small sample: the 100-row number is the one to trust.

#### The lookup

The 37% above is the solver on problems that are **not** in the bank: the practice file was taken out
of the bank for that run, and the lookup answered none of the 100. That is the number to compare with
the baseline. When a problem *is* in the bank, the lookup answers it, so these scores measure the
bank, not the reasoning:

| Test, on a laptop | Right | Answered by the lookup |
| --- | --- | --- |
| Practice rows, practice file in the bank, check off | 100 of 100 | 100 |
| Practice rows, practice file in the bank, check on | 92 of 100 | 91 |
| Cangur 2026, all 454 rows, 45% of its questions in the bank, check on | 315 of 454 (69%) | 255 |

On the Cangur rows, questions kept in the bank score 195 of 204, and questions taken out score 120 of
250: Kangaroo reuses questions between neighbouring levels, so 51 of those rows were still found
through a copy. The check costs points: it rejected 9 right answers on the practice rows, and the code
stage recovered 1.

On M3Kang validation (562 rows, English and Catalan), where we tried the letter-picking ideas:

| Approach | Accuracy |
| --- | --- |
| Organizers' baseline (40 problems) | 22.5% |
| 0.8B picks a letter, fused with the prior | 25.3% |
| Prior only, no model | 27.8% |
| 3 option orders averaged | 28.1% |
| The same problem found in another language, plus 3 option orders | 30.8% |

### Teaching

From `tools/teach_report.py` on the bank of 1,151 problems (583 ids, English and Catalan):

| Measure | Value |
| --- | --- |
| Problems with a similar example for rung 1 | 99.1% |
| Example at a lower level (the rest are at the same level) | 96.1% |
| Example on the same topic | 82.1% |
| A problem's translation shown as its example | 0 |
| Hints the model wrote that leak the answer | 21 of 1,151 (1.8%) |
| Hints the student sees that leak the answer | 0 of 873 |
| Hints rejected because they do the calculation | 201 of 1,151 |
| Time for rungs 1 and 2, and a stored hint | under 1 ms |
| Time for a hint generated on the board | about 20 s |

Simulated students who solve alone up to their level and nothing above it, 40 problems each: whatever
level they start at, the trainer settles between their level and one above it (for a level-3 student, a
mean of 3.3 to 3.5 over the last 15 problems). After a miss, the next problem stays on the same topic
in 70% to 100% of cases.

## Error analysis

All on the 100 practice rows, code stage with the 2B model.

**By route.** The code stage is right when it answers, and it answers less than half the time.

| Route | Rows | Correct | Accuracy |
| --- | --- | --- | --- |
| Program accepted | 42 | 21 | 50% |
| No accepted program: the prior answers | 58 | 16 | 28% |

Why a program was not accepted: 35 failed with a Python error, 11 printed nothing, 11 printed a value
that is not among the options, and 1 used a banned name. With 96 tokens the usual error is a program
cut off before its last line.

**By topic** (keyword tags from `copilot/topics.py`):

| Topic | Rows | Correct | Accuracy |
| --- | --- | --- | --- |
| Number theory | 32 | 15 | 47% |
| Arithmetic | 12 | 7 | 58% |
| Measurement and time | 10 | 2 | 20% |
| Plane geometry | 8 | 2 | 25% |
| Combinatorics | 6 | 3 | 50% |
| Solid geometry | 4 | 0 | 0% |
| Other topics (8 topics, 2 to 6 rows each) | 28 | 8 | 29% |

The code stage works where the problem is a computation: number theory and arithmetic give 16 of its
21 correct answers. It fails where the problem must first be modelled: in measurement and solid
geometry it ran 8 programs that matched an option, and all 8 were wrong.

**By language.** No difference: Catalan 19 of 50, English 18 of 50. The model reads the Catalan
problem and writes the same Python.

**By difficulty.** No clear trend, and 10 to 16 rows per level is too few to read one:

| Level | 0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Rows | 16 | 16 | 14 | 12 | 12 | 10 | 10 | 10 |
| Correct | 5 | 5 | 5 | 2 | 6 | 5 | 2 | 7 |

A harder Kangaroo problem is not harder to brute-force. Many level 6 and 7 problems are number theory
with a small search space.

**By figure.** Text-only rows: 32 of 80 (40%). Rows with a figure: 5 of 20 (25%). The solver never
sees the figure.

**By letter.** When the code stage has no answer, the prior picks the median option, which is usually
C: the copilot answered C on 58 of 100 rows, when each letter was right 20 times.

## What didn't work

- **Asking a small model for the letter.** Through the pipeline, every variant stayed between 18% and
  21% on the practice rows: one call, 3 option orders, 5 option orders. The 2B model asked for a letter
  got 21% too.
- **Reasoning in words before the letter.** It works, but not in the time we have. On the practice
  rows the 2B model went from 21% with no reasoning to 29% with 64 tokens, 32% with 256 and 46% with
  512. At 2.6 tokens per second, 512 tokens take over 3 minutes on the board. In the roughly 100
  tokens that fit, words give 20% to 29% and code gives 37%. The 0.8B model did not improve with
  reasoning at any length (21% to 27%).
- **Solved examples in the prompt.** 23.0% on M3Kang validation, below the 25.3% without them.
- **Fine-tuning the 0.8B to pick letters.** 25.1% against 25.3%.
- **Looking the problem up as a way to score.** The lookup answers a problem that is already in the
  bank, and nothing else: on the 100 practice rows, with the practice file out of the bank, it answered
  none. It is on by default as a safety net, not as a source of points.
- **Letting the model check the lookup.** The 2B cannot tell a right proposed answer from a wrong one
  (52 "yes" against 51). The check is in the pipeline as a second opinion, and it only costs points.
- **Letting the model guard the hints.** It wrote the answer value into 21 hints and did the
  calculation in 201. The check had to be code.

What we would do next: let a program that fails have one more try with the error message, and read
the figure's options, which the solver never sees.

## Replicate it

Start from a board set up with steps 0 to 12 of [SETUP.md](SETUP.md): `llama.cpp` built, the `math-llm`
and `math-embed` services running, this repository cloned on the board, and `data/train.jsonl` and
`data/val.jsonl` prepared on your laptop.

### 1. The solver

On the board, download the code model (1.4 GB) and check it:

```bash
wget -O ~/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF/resolve/main/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

```bash
sha256sum ~/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf
```

The sum must be `57a1085840f497d764a7fc5d346922dbde961efb54cc792ea81d694fd846a1d8`, as in `config.env`.

Install its service, from the repository folder on the board:

```bash
sudo cp board/systemd/math-code.service /etc/systemd/system/ && sudo systemctl daemon-reload && sudo systemctl enable --now math-code
```

Run it:

```bash
python3 predict.py --input data/sample_problems.jsonl --output results/smoke.csv
```

**Check:** the first line says `code stage: on`, and each problem prints its answer, its time and its
route (`code` or `code-prior`). If it says `code stage: off (no server)`, check
`systemctl status math-code`.

### 2. The tutor

On your laptop, start the language model server on port 8081:

```bash
bash laptop/serve.sh ~/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf
```

In a second terminal, build the bank from `data/train.jsonl` and `data/val.jsonl`:

```bash
python3 laptop/build_teach_bank.py --hints
```

It writes `app/math-matrix/python/teach.db`. Try the page on the laptop first:

```bash
python3 app/math-matrix/python/dev_server.py --port 7000
```

Copy the app to the board, over USB:

```bash
adb push app/math-matrix /home/arduino/ArduinoApps/
```

On the board, let the app reach the model (once per board), and start it:

```bash
sudo bash ~/ArduinoApps/math-matrix/python/setup_bridge.sh
```

```bash
arduino-app-cli app start user:math-matrix
```

Open `http://<board-ip>:7000`. Over USB, run `adb forward tcp:7000 tcp:7000` on the laptop and open
`http://127.0.0.1:7000`.

**Check:** the start page shows four games, and the LED matrix shows `L` and the level when a problem
starts.

### 3. Measure it

```bash
python3 eval.py --input data/val.jsonl --name my-run
```

```bash
python3 tools/error_report.py results/eval-<time>-my-run.csv --data data/val.jsonl
```

```bash
python3 tools/teach_report.py
```

## Repository layout

```text
predict.py             the entry point the jury runs: one letter per problem
eval.py                the same pipeline with labels: accuracy, time, one row per problem
config.env             models (repository, file, SHA256), ports, time limit, seed
copilot/pipeline.py    the cascade: prior, code stage, fallback
copilot/coder.py       the code stage: prompt, token budget, sandboxed run, match to an option
copilot/priors.py      the answer with no model
copilot/topics.py      the topic tagger
app/math-matrix/       the App Lab app: tutor page, game, LED matrix
board/                 board setup scripts and the three systemd services
laptop/                data preparation, the hint bank, experiments
tools/                 reports: errors, teaching, latency
SETUP.md               the organizers' guide: from a fresh board to the baseline
```
## Images
<img width="1459" height="847" alt="image" src="https://github.com/user-attachments/assets/4ecc0590-247c-4e0e-96ee-d00ec57e002e" />



## Team

brainrot_coders: Elena Andreev, Jan Balayla, Andreu Clotas and Josep Cardona. 3rd place overall at
the Datathon FME 2026.

## Credits and licenses

Our code is under the [MIT license](LICENSE). The organizers' starting code and guide keep their own terms.

- **M3Kang**: Torres-Camps et al., 2026, [arXiv:2601.16218](https://arxiv.org/abs/2601.16218). Qualcomm.
  [Dataset card and license](https://huggingface.co/datasets/qualcomm/M3Kang): research use only.
  Problems from the Kangaroo math competition.
- **The starting repository**: [mpous/datathon-UPC-maths-qualcomm-arduino-unoq](https://github.com/mpous/datathon-UPC-maths-qualcomm-arduino-unoq), by the organizers.
- **llama.cpp**: [ggml-org/llama.cpp](https://github.com/ggml-org/llama.cpp), MIT license.
- **Qwen3.5 0.8B and 2B**: by the Qwen team, GGUF by bartowski. See the model cards for
  [0.8B](https://huggingface.co/bartowski/Qwen_Qwen3.5-0.8B-GGUF) and
  [2B](https://huggingface.co/bartowski/Qwen_Qwen3.5-2B-GGUF).
- **bge-small-en-v1.5**: by BAAI, GGUF by CompendiumLabs.
  [Model card](https://huggingface.co/CompendiumLabs/bge-small-en-v1.5-gguf).
