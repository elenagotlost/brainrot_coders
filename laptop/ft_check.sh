#!/usr/bin/env bash
# M0 tooling check: can we fine-tune Qwen3.5 on this Mac AND get it back onto the board?
# Runs the whole chain on toy data in ~10 minutes, BEFORE we spend hours making training data:
#
#   1. LoRA, 10 steps, with mlx-lm             (Apple Silicon)
#   2. fuse the adapter into the model
#   3. convert to GGUF with convert_hf_to_gguf.py from the PINNED llama.cpp commit
#   4. quantize to Q8_0
#   5. load it in llama-server (same flags as the board) and ask one question
#
# Usage, from the repository root, after laptop/mac_setup.sh:
#   bash laptop/ft_check.sh                 # 0.8B
#   bash laptop/ft_check.sh Qwen/Qwen3.5-2B # same check for the 2B
# Uses the 5 made-up problems in data/sample_problems.jsonl only (no M3Kang data).
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env
BASE="${1:-Qwen/Qwen3.5-0.8B}"
WORK="${FT_WORK:-$HOME/ftcheck/$(basename "$BASE")}"
VENV="${FT_VENV:-$HOME/venvs/ft}"        # mlx-lm (needs transformers 5.x)
GGUF_VENV="${GGUF_VENV:-$HOME/venvs/gguf}" # convert_hf_to_gguf.py at the pinned commit (pins transformers 4.x)
# Both must be native arm64 Pythons: Anaconda's x86 python3 cannot install mlx.
PY="${FT_PYTHON:-/opt/homebrew/bin/python3.12}"
mkdir -p "$WORK/data"

step() { printf '\n\033[1m== STEP %s\033[0m\n' "$*"; }
fail() { printf '\n\033[31mFAILED at step %s\033[0m -- paste the error above into the chat.\n' "$1"; exit 1; }

step "0  python envs ($VENV, $GGUF_VENV)"
if [ ! -d "$VENV" ]; then "$PY" -m venv "$VENV"; "$VENV/bin/pip" install -q -U pip mlx-lm; fi
if [ ! -d "$GGUF_VENV" ]; then
  "$PY" -m venv "$GGUF_VENV"
  "$GGUF_VENV/bin/pip" install -q -U pip
  "$GGUF_VENV/bin/pip" install -q -r "$LLAMA_CPP_DIR/requirements/requirements-convert_hf_to_gguf.txt"
fi
source "$VENV/bin/activate"
python -c "import platform, mlx_lm, importlib.metadata as m; print(platform.machine(), 'mlx-lm', m.version('mlx-lm'), 'transformers', m.version('transformers'))"
"$GGUF_VENV/bin/python" -c "import importlib.metadata as m; print('convert env: torch', m.version('torch'), 'transformers', m.version('transformers'))"

step "1a toy data in chat format"
python3 - "$WORK/data" <<'EOF'
import json, sys, os
sys.path.insert(0, os.getcwd())
from copilot.llm import build_messages
out = sys.argv[1]
rows = [json.loads(l) for l in open("data/sample_problems.jsonl")]
ex = [{"messages": build_messages(p) + [{"role": "assistant", "content": p["label"]}]} for p in rows]
for name, items in (("train", ex * 4), ("valid", ex)):
    with open(os.path.join(out, name + ".jsonl"), "w") as f:
        for e in items:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")
print("wrote", len(ex) * 4, "train /", len(ex), "valid examples")
EOF

step "1  LoRA, 10 steps on $BASE"
mlx_lm.lora --model "$BASE" --train --data "$WORK/data" --iters 10 --batch-size 2 \
  --num-layers 4 --adapter-path "$WORK/adapters" || fail 1

step "2  fuse"
mlx_lm.fuse --model "$BASE" --adapter-path "$WORK/adapters" --save-path "$WORK/fused" --dequantize \
  || mlx_lm.fuse --model "$BASE" --adapter-path "$WORK/adapters" --save-path "$WORK/fused" || fail 2
ls "$WORK/fused"
python3 -c "import json;c=json.load(open('$WORK/fused/config.json'));print('architectures:',c.get('architectures'),'model_type:',c.get('model_type'))"

step "2b restore the original tokenizer files and the Hugging Face weight layout"
# mlx-lm (transformers 5.x) saves tokenizer_class "Qwen3_5Tokenizer", which the pinned converter
# (transformers 4.x) cannot load. LoRA never changes the tokenizer, so put the originals back.
SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$BASE'))")
for name in tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja; do
  [ -f "$SNAP/$name" ] && cp -L "$SNAP/$name" "$WORK/fused/$name"
done
grep -o '"tokenizer_class": *"[^"]*"' "$WORK/fused/tokenizer_config.json"
# MLX stores norm weights as w + 1 and the converter adds 1 again: without this the GGUF
# loads and answers but is silently wrong (see laptop/mlx_to_hf.py).
"$GGUF_VENV/bin/python" laptop/mlx_to_hf.py "$WORK/fused" --base "$SNAP" || fail 2b

step "3  convert to GGUF (pinned llama.cpp ${LLAMA_CPP_REF:0:7})"
"$GGUF_VENV/bin/python" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$WORK/fused" --outfile "$WORK/fused-f16.gguf" --outtype f16 \
  || fail "3 (convert). Try again with the original config: cp \$(python -c \"from huggingface_hub import snapshot_download as s;print(s('$BASE'))\")/config.json $WORK/fused/ and rerun only this step"

step "4  quantize Q8_0"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$WORK/fused-f16.gguf" "$WORK/fused-Q8_0.gguf" Q8_0 >/dev/null || fail 4
ls -lh "$WORK/fused-Q8_0.gguf"

step "5  load in llama-server and answer one problem"
"$LLAMA_CPP_DIR/build/bin/llama-server" -m "$WORK/fused-Q8_0.gguf" --host 127.0.0.1 --port 8099 \
  --ctx-size 4096 --parallel 1 --cache-ram 0 --ctx-checkpoints 2 --jinja --reasoning off --reasoning-budget 0 \
  >"$WORK/server.log" 2>&1 &
SERVER=$!
trap 'kill $SERVER 2>/dev/null || true' EXIT
for _ in $(seq 60); do curl -sf http://127.0.0.1:8099/health >/dev/null && break; sleep 1; done
curl -sf http://127.0.0.1:8099/health >/dev/null || { tail -20 "$WORK/server.log"; fail "5 (server did not start)"; }
python3 - <<'EOF' || fail 5
import json, os, sys
sys.path.insert(0, os.getcwd())
from copilot.llm import answer_letter
p = json.loads(open("data/sample_problems.jsonl").readline())
letter, t = answer_letter(p, url="http://127.0.0.1:8099/v1/chat/completions")
print("answer:", letter, "(label", p["label"] + ")  prompt tokens:", t.get("prompt_n"))
EOF

printf '\n\033[32mALL STEPS PASSED for %s\033[0m: the fine-tune -> GGUF -> llama-server chain works.\n' "$BASE"
echo "Model file to copy to the board for a smoke test: $WORK/fused-Q8_0.gguf"
