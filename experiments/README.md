# Things we tried that didn't work

This folder holds the scripts of experiments that are **not** part of the final copilot, and this page says what each one was and why we dropped it. The final copilot works like this: a small code model (Qwen2.5-Coder 1.5B) writes a short brute-force program, the board runs it, and the answer counts only when the program's output equals exactly one option. When that fails, a prior fitted on M3Kang answers.

## How to read the numbers

- **Practice set**: the 100 rows of the datathon's practice file (50 problems, each in English and Catalan). The final test is written in the same style.
- **Held-out set**: 178 M3Kang validation rows that are text-only and have all five options read by OCR. No decision was tuned on them. They are harder than the practice set.
- **M3Kang validation**: the 562 rows of our validation split (281 problems in two languages). 223 of them have no options, because the options are pictures.
- Unless a row says "on the board", a number was measured on a laptop with the same model file and the board's token budget. We ran the final model on both: 40 of 100 on the laptop and 40 of 100 on the board, with 10 individual answers different.
- 50 problems is a small sample. A difference of 2 or 3 rows is noise.

The reference point: **the final pipeline scores 46 of 100 on the practice set, and always answering with the prior scores 24.**

## 1. Asking the model for the letter

The starting point of the challenge: Qwen3.5 0.8B reads the problem and its options, and a grammar forces one letter.

| What we tried | Result |
| --- | --- |
| Model alone, M3Kang validation problems with options (339 rows) | 24.2%. Always D scores 21.2% there. The model answered D 256 times. |
| Dividing out the model's letter bias | 19.5% to 20.6%: chance. The model's apparent skill was its taste for D. |
| Averaging the English and Catalan versions of a problem | 64 right against 60, of 316 rows: noise. |
| The prior alone, no model, all 562 rows | 27.8%: better than the model. |
| Calibrated fusion of model and prior (a teammate's run) | 28.1%, against 27.8% for the prior alone. |
| Few-shot examples, and asking with the options in a different order (a teammate's run, 50 rows) | 16% and 30%, against 18% plain. The 30% matched always answering A on that sample. |

**Why it failed:** an 0.8B model reading a competition problem once cannot solve it. Nothing we did to the letter probabilities found signal that the prior did not already have.

Scripts: `dump_probs.py`, `fit_calib.py`, `e0_cache_check.py`, `split_years.py`, `analysis/`.

## 2. Looking up the same problem in M3Kang

Kangaroo papers reuse problems across grade levels, so a test problem might already be solved in the training split.

- Only 7 of 562 validation rows have a copy in training.
- The copies list their options in a different order, so the stored letter was right for 3 of the 7.
- The final test is newly written, so there is nothing to look up.

## 3. Reasoning in words

| What we tried | Result |
| --- | --- |
| Qwen2.5-Math 1.5B, letter only, practice set | 24 of 100: chance |
| Qwen2.5-Math 1.5B, 96 tokens of working, then the letter | 17 of 100. It did not finish its working on a single problem. |
| Qwen2.5-Math 1.5B writing code, with the final prompt | 26 of 100. Its real computations were right 17 times of 19, but 34 programs crashed. |

**Why it failed:** the maths model writes long, formal working. On the board, 96 tokens already cost about half of the 60 seconds, and it needs several times that.

Scripts: `reason_lab.py`, `pot.py`.

## 4. Other code models

All on the practice set, same prompt and harness (`laptop/code_lab.py`), choices hidden.

| Model | Right of 100 | Real computations right | Outcome |
| --- | --- | --- | --- |
| Qwen3.5 2B | 15 | 14 of 17 | Dropped: writes a working computation too rarely |
| Qwen2.5-Math 1.5B | 26 | 17 of 19 | Dropped: see section 3 |
| **Qwen2.5-Coder 1.5B** | **30** | **26 of 33** | **Kept** |
| Qwen2.5-Coder 3B | 42 | 35 of 40 | Best on a laptop, too slow on the board |

### The 3B on the board

| 3B set-up, measured on the board | Reads (tokens/s) | Writes (tokens/s) |
| --- | --- | --- |
| Q4_K_M | 2.4 | 1.8 |
| Q4_0 | 2.6 | 2.0 |
| Q3_K_M | 2.3 | 1.6 |
| Q4_K_M with a 0.5B draft model (speculative decoding) | 2.1 | 1.1 |
| For comparison: Coder 1.5B, Q4_K_M | 4.4 | 2.8 |

- Reading a typical problem takes the 3B 20 to 30 seconds. That leaves room for 25 to 40 tokens of program, and its working programs need about 34 or more.
- On 20 practice rows on the board it got 0 right: 8 programs were cut off and crashed, 6 problems were skipped for lack of time.
- The draft model made it slower. The drafts were good (156 of 238 accepted), but on four slow cores running a second model costs more than the accepted tokens save.
- The 3B is also under a research licence. The 1.5B is Apache 2.0.

Script: `code_probe.py` (the first version of the code experiment).

## 5. Changes to the code stage that did not help

All with Qwen2.5-Coder 1.5B unless a row says otherwise.

| What we tried | Result | Why we dropped it |
| --- | --- | --- |
| Showing the answer choices to the model | Qwen3.5 2B: 50 of 100 replies only printed a choice, right 15 times | The model guesses instead of computing. The choices are now hidden. |
| When the output matches no option, take the nearest option | Right 4 times of 38 | Worse than chance. The prior got 10 of the same 38, so the prior answers now (40 to 46 of 100). |
| A second try, telling the model what went wrong (Qwen3.5 2B) | 2 more right of 100 | It made the server read the whole prompt again: about 170 s on the board. |
| No worked examples in the prompt | 30 of 100 on practice (46 with them), 49 of 178 on held-out (53 with them) | The examples help on both sets. Kept. |
| Three of the examples in Catalan, plus repairing programs that do not parse | 48 of 100 and 54 of 178, against 46 and 53 | Inside the noise. Not adopted: not enough evidence. |

## 6. Fine-tuning

We built the chain to fine-tune a small model on M3Kang (a larger local model writes solutions with `laptop/teacher.py`, then LoRA, then a GGUF file for the board), and proved it end to end on toy data. No fine-tuned model is part of the final copilot: the code approach gave more for the time we had.

Scripts: `make_ft_data.py`, `finetune.sh`, `ft_check.sh`, `mlx_to_hf.py`, `mac_setup.sh`.

## 7. Topic labels

`sample_for_labels.py` and `topic_agreement.py` were for hand-labelling problems by topic and scoring a rule-based tagger against the labels. We did not finish the labelling, so the final copilot has no error analysis by topic.

## What limits the final copilot

- **Problems with a figure.** The model never sees the image. These stay near chance: 6 of 20 practice rows in the board run.
- **Low-level word problems.** Levels 0 and 1 scored worse than levels 5 to 7. The high levels are number problems that turn straight into a loop. The low levels need careful reading ("for the first time", "greater than 1"), which a 1.5B model does badly.
- **Real Kangaroo problems.** On the held-out M3Kang set the code stage writes a matching computation for only 13 of 178 rows. Almost all of the 30% there comes from the prior.
- **Overfitting.** The prompt's worked examples were written after reading the practice set, and several follow the shape of practice problems. The practice score is optimistic.

## Running these scripts

They were moved here from `laptop/` and `tools/` and are kept as a record. To run one, start it from the repository root with `PYTHONPATH=.`, for example:

```bash
PYTHONPATH=. python3 experiments/reason_lab.py --name test --url http://127.0.0.1:8091/v1/chat/completions --tokens 96
```
