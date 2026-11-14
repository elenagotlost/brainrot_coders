#!/usr/bin/env bash
# Start llama-server on the laptop with the SAME flags as the board service
# (board/systemd/math-llm.service), so laptop answers match board answers.
#
#   bash laptop/serve.sh ~/models/Qwen_Qwen3.5-0.8B-Q8_0.gguf          # port 8081
#   bash laptop/serve.sh ~/models/Qwen_Qwen3.5-2B-Q4_K_M.gguf 8091     # a second model
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env
model="${1:?give the path of a .gguf model}"
port="${2:-$LLM_PORT}"
exec "$LLAMA_CPP_DIR/build/bin/llama-server" -m "$model" \
  --host 127.0.0.1 --port "$port" --ctx-size "$LLM_CTX" \
  --parallel 1 --cache-ram 0 --ctx-checkpoints 2 \
  --jinja --reasoning off --reasoning-budget 0 --alias "$(basename "$model" .gguf)"
