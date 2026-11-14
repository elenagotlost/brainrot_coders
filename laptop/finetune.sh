#!/usr/bin/env bash
# M4: LoRA fine-tune of a Qwen3.5 model on this Mac, then all the way to a GGUF the board can run.
# The same chain as laptop/ft_check.sh (which proved it on toy data), with real data and settings.
#
#   bash laptop/finetune.sh <base> <name> <data folder> <iterations> [quantization] [layers] [lr]
#   bash laptop/finetune.sh Qwen/Qwen3.5-0.8B q08-reason ~/ft/data-reason 600 Q8_0
#   bash laptop/finetune.sh Qwen/Qwen3.5-2B   q2b-letter ~/ft/data-letter 400 Q4_K_M
#
# Result: ~/models/<name>-<quantization>.gguf and its SHA256. Work files in ~/ft/<name>.
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env
BASE="${1:?base model, e.g. Qwen/Qwen3.5-0.8B}"
NAME="${2:?short name, e.g. q08-reason}"
DATA="${3:?folder with train.jsonl and valid.jsonl (laptop/make_ft_data.py)}"
ITERS="${4:?number of training steps}"
QUANT="${5:-Q8_0}"
LAYERS="${6:-16}"
LR="${7:-1e-4}"
WORK="$HOME/ft/$NAME"
VENV="${FT_VENV:-$HOME/venvs/ft}"
GGUF_VENV="${GGUF_VENV:-$HOME/venvs/gguf}"
mkdir -p "$WORK"
step() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
source "$VENV/bin/activate"

step "1  LoRA: $BASE, $ITERS steps, $LAYERS layers, lr $LR, data $DATA"
# --mask-prompt: learn only from the answer part, not from re-reading the problem
mlx_lm.lora --model "$BASE" --train --data "$DATA" --iters "$ITERS" --batch-size "${BATCH:-4}" \
  --num-layers "$LAYERS" --learning-rate "$LR" --max-seq-length "${MAXLEN:-1024}" --mask-prompt \
  --steps-per-report 25 --steps-per-eval 100 --val-batches 10 --save-every 100 \
  --adapter-path "$WORK/adapters" --seed "$SEED"

step "2  fuse, restore the tokenizer and the Hugging Face weight layout"
rm -rf "$WORK/fused"
mlx_lm.fuse --model "$BASE" --adapter-path "$WORK/adapters" --save-path "$WORK/fused"
SNAP=$(python -c "from huggingface_hub import snapshot_download as s; print(s('$BASE'))")
for name in tokenizer.json tokenizer_config.json vocab.json merges.txt chat_template.jinja; do
  [ -f "$SNAP/$name" ] && cp -L "$SNAP/$name" "$WORK/fused/$name"
done
"$GGUF_VENV/bin/python" laptop/mlx_to_hf.py "$WORK/fused" --base "$SNAP"

step "3  convert to GGUF (pinned llama.cpp ${LLAMA_CPP_REF:0:7}) and quantize to $QUANT"
"$GGUF_VENV/bin/python" "$LLAMA_CPP_DIR/convert_hf_to_gguf.py" "$WORK/fused" \
  --outfile "$WORK/f16.gguf" --outtype f16 > "$WORK/convert.log" 2>&1
OUT="$MODELS_DIR/$NAME-$QUANT.gguf"
"$LLAMA_CPP_DIR/build/bin/llama-quantize" "$WORK/f16.gguf" "$OUT" "$QUANT" > "$WORK/quantize.log" 2>&1
rm -f "$WORK/f16.gguf"

step "done"
ls -lh "$OUT"
shasum -a 256 "$OUT"
