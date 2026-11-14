#!/usr/bin/env python3
"""Put an mlx-lm Qwen3.5 checkpoint back into the Hugging Face layout, in place.

Why: mlx-lm changes three things when it loads Qwen3.5 and keeps them when it saves
(mlx_lm.fuse). convert_hf_to_gguf.py expects the Hugging Face layout and applies its
own changes, so without this step the GGUF loads, answers, and is silently WRONG:

  1. RMSNorm weights are stored as w in Hugging Face and as w + 1 in MLX. The converter
     adds 1 again, so every norm ends up 1 too big.
  2. conv1d weights are [out, 1, kernel] in Hugging Face and [out, kernel, 1] in MLX.
  3. Keys are model.language_model.* in Hugging Face and language_model.model.* in MLX.

Run it with the Python of the GGUF environment (it needs torch and safetensors):

    ~/venvs/gguf/bin/python laptop/mlx_to_hf.py ~/ftcheck/Qwen3.5-0.8B/fused --base <snapshot folder>

Safe to run twice: a checkpoint already in the Hugging Face layout is left alone.
"""
import argparse
import glob
import json
import os
import sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file

# The same list as Qwen3_5 sanitize() in mlx-lm: the norms that MLX shifts by +1.
# linear_attn.norm.weight is NOT in it (neither side shifts that one).
NORM_SUFFIXES = (".input_layernorm.weight", ".post_attention_layernorm.weight",
                 "model.norm.weight", ".q_norm.weight", ".k_norm.weight")


def write_index(folder, keys):
    """The converter checks model.safetensors.index.json against the real tensor names."""
    with open(os.path.join(folder, "model.safetensors.index.json"), "w") as f:
        json.dump({"metadata": {"format": "pt"},
                   "weight_map": {k: "model.safetensors" for k in sorted(keys)}}, f, indent=1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("folder", help="the folder written by mlx_lm.fuse")
    ap.add_argument("--base", default="", help="folder of the original Hugging Face model: "
                    "norm weights are copied from it exactly (recommended)")
    args = ap.parse_args()
    folder = args.folder
    path = os.path.join(folder, "model.safetensors")
    base = None
    if args.base:
        parts = sorted(glob.glob(os.path.join(args.base, "*.safetensors")))
        if len(parts) != 1:
            sys.exit(f"expected one .safetensors file in {args.base}, found {len(parts)}")
        base = safe_open(parts[0], "pt")
    with safe_open(path, "pt") as f:
        if (f.metadata() or {}).get("format") != "mlx":
            write_index(folder, f.keys())
            print(f"{path}: not in MLX layout (already converted?), nothing to do")
            return
        tensors = {k: f.get_tensor(k) for k in f.keys()}

    out, norms, convs = {}, 0, 0
    for key, value in tensors.items():
        if key.startswith("language_model.model."):
            new_key = "model.language_model." + key[len("language_model.model."):]
        elif key.startswith("language_model."):
            new_key = key[len("language_model."):]  # lm_head.weight, when not tied
        else:
            sys.exit(f"unexpected key {key}: this script only knows the Qwen3.5 text model")
        if value.ndim == 1 and key.endswith(NORM_SUFFIXES):
            shifted = value.to(torch.float32)
            if base is not None:
                # LoRA never trains norms, so take the exact original: MLX did its "+ 1" in
                # 16-bit floats, which rounds weights near 4 to steps of 0.03.
                value = base.get_tensor(new_key)
                gap = (value.to(torch.float32) + 1.0 - shifted).abs().max().item()
                if gap > 0.05:
                    sys.exit(f"{key}: differs from the base model by {gap:.3f}; wrong --base?")
            else:
                # float32 so that the converter's "+ 1" gives back exactly the MLX value
                value = shifted - 1.0
            norms += 1
        elif "conv1d.weight" in key and value.ndim == 3 and value.shape[-1] == 1:
            value = value.movedim(2, 1).contiguous()
            convs += 1
        out[new_key] = value

    if not norms or not convs:
        sys.exit(f"found {norms} norm and {convs} conv1d tensors: expected both, refusing to write")
    tmp = path + ".tmp"
    save_file(out, tmp, metadata={"format": "pt"})
    os.replace(tmp, path)
    write_index(folder, out)
    print(f"{path}: {len(out)} tensors back in Hugging Face layout "
          f"({norms} norm weights {'copied from the base' if base is not None else 'un-shifted'}, "
          f"{convs} conv1d weights re-shaped)")


if __name__ == "__main__":
    main()
