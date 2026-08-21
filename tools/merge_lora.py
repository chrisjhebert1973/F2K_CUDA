#!/usr/bin/env python3
"""Bake a diffusers-trained FLUX.2 LoRA into the base BF16 transformer weights,
producing a plain safetensors that f2k_convert can quantize — the C++ inference
stack never needs to know LoRAs exist.

  merge_lora.py <base_transformer_dir> <lora.safetensors> <out.safetensors> [--scale S]

base_transformer_dir: the diffusers transformer dir (shards + index), e.g.
  ~/models/flux2-klein-4B/transformer
lora.safetensors: pytorch_lora_weights.safetensors from
  train_dreambooth_lora_flux2_klein.py (keys: transformer.<module>.lora_{A,B}.weight)

W' = W + scale * (B @ A), computed in fp32, stored back as BF16.
--scale defaults to 1.0, correct when lora_alpha == rank (our training default).
Multiple --lora can be given to bake several characters into one checkpoint.
"""
import argparse, glob, os, re, sys

import torch
from safetensors import safe_open
from safetensors.torch import save_file


def load_dir(d):
    out = {}
    for shard in sorted(glob.glob(os.path.join(d, "*.safetensors"))):
        with safe_open(shard, framework="pt") as f:
            for k in f.keys():
                out[k] = f.get_tensor(k)
    if not out:
        sys.exit(f"no safetensors found in {d}")
    return out


def merge_one(base, lora_path, scale):
    pairs = {}
    with safe_open(lora_path, framework="pt") as f:
        for k in f.keys():
            m = re.match(r"(?:transformer\.)?(.+)\.lora_(A|B)\.weight$", k)
            if not m:
                print(f"  skipping non-LoRA key: {k}")
                continue
            pairs.setdefault(m.group(1), {})[m.group(2)] = f.get_tensor(k)

    merged = 0
    for mod, ab in sorted(pairs.items()):
        if "A" not in ab or "B" not in ab:
            sys.exit(f"incomplete LoRA pair for {mod}")
        key = mod + ".weight"
        if key not in base:
            sys.exit(f"LoRA targets {key} but base has no such tensor")
        w = base[key].float()
        delta = (ab["B"].float() @ ab["A"].float()) * scale
        if delta.shape != w.shape:
            sys.exit(f"{key}: delta {tuple(delta.shape)} != base {tuple(w.shape)}")
        base[key] = (w + delta).to(torch.bfloat16)
        merged += 1
    print(f"  merged {merged} modules from {os.path.basename(lora_path)} (scale={scale})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("base_dir")
    ap.add_argument("lora", nargs="+", help="one or more LoRA safetensors to bake in")
    ap.add_argument("out")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="LoRA strength multiplier (alpha/rank already folded in when alpha==rank)")
    args = ap.parse_args()

    base = load_dir(args.base_dir)
    print(f"base: {len(base)} tensors from {args.base_dir}")
    for lp in args.lora:
        merge_one(base, lp, args.scale)
    save_file(base, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
