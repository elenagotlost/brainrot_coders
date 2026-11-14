#!/usr/bin/env bash
# M0: set up the model track on a Mac (Apple Silicon).
#   - builds llama.cpp at the SAME pinned commit as the board (Metal is on by default)
#   - downloads the pinned 0.8B model + embedder, and the two candidates for the bake-off
#
# Usage, from the repository root:   bash laptop/mac_setup.sh
# Needs: Xcode command line tools, cmake (brew install cmake), git, curl.
set -euo pipefail
cd "$(dirname "$0")/.."
source config.env

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
ok()  { printf '   \033[32mok\033[0m %s\n' "$*"; }

command -v cmake >/dev/null || { echo "cmake missing: brew install cmake"; exit 1; }

# ---------------------------------------------------------------- llama.cpp
short="${LLAMA_CPP_REF:0:7}"
say "llama.cpp at commit $short in $LLAMA_CPP_DIR"
if [ ! -d "$LLAMA_CPP_DIR/.git" ]; then
  git init -q "$LLAMA_CPP_DIR"
  git -C "$LLAMA_CPP_DIR" remote add origin "$LLAMA_CPP_REPO"
fi
git -C "$LLAMA_CPP_DIR" fetch -q --depth 1 origin "$LLAMA_CPP_REF"
git -C "$LLAMA_CPP_DIR" checkout -q --detach FETCH_HEAD
cmake -S "$LLAMA_CPP_DIR" -B "$LLAMA_CPP_DIR/build" -DCMAKE_BUILD_TYPE=Release >/dev/null
cmake --build "$LLAMA_CPP_DIR/build" --config Release -j \
  --target llama-server llama-cli llama-quantize >/dev/null
"$LLAMA_CPP_DIR/build/bin/llama-server" --version 2>&1 | grep -m1 version
ok "built (llama-server, llama-cli, llama-quantize)"

# ---------------------------------------------------------------- models
mkdir -p "$MODELS_DIR"
fetch() {  # repo revision file [sha256]
  local repo="$1" rev="$2" file="$3" sha="${4:-}" target="$MODELS_DIR/$3"
  if [ ! -f "$target" ]; then
    echo "   downloading $file"
    curl -fL --progress-bar -C - -o "$target.part" "https://huggingface.co/$repo/resolve/$rev/$file" \
      || { echo "   !! could not download $repo/$file (check the file name on huggingface.co)"; return 0; }
    mv "$target.part" "$target"
  fi
  local got; got=$(shasum -a 256 "$target" | cut -d' ' -f1)
  if [ -n "$sha" ] && [ "$got" != "$sha" ]; then
    echo "   !! $file SHA256 mismatch: delete it and run again"; exit 1
  fi
  ok "$file  sha256=$got"
}

say "Pinned models (same as the board)"
fetch "$LLM_REPO" "$LLM_REVISION" "$LLM_FILE" "$LLM_SHA256"
fetch "$EMBED_REPO" "$EMBED_REVISION" "$EMBED_FILE" "$EMBED_SHA256"

say "Bake-off candidates (not pinned yet: copy the sha256 into the doc when one wins)"
fetch "$LLM_REPO" "$LLM_REVISION" "Qwen_Qwen3.5-0.8B-Q4_K_M.gguf"
fetch "bartowski/Qwen_Qwen3.5-2B-GGUF" "main" "Qwen_Qwen3.5-2B-Q4_K_M.gguf"

say "Done. Next:"
echo "   bash laptop/serve.sh $MODELS_DIR/$LLM_FILE      # terminal 1"
echo "   python3 laptop/dump_probs.py --input data/val.jsonl --model q08-q8   # terminal 2"
