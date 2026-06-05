#!/usr/bin/env python3
"""
Run diffusers Flux2KleinPipeline._get_qwen3_prompt_embeds to obtain a known-good
[seq=512, 12288] BF16 conditioning tensor, save it as a raw binary file the C++
generate tool can read via --embeds.

Layout of output file:
  int32   seq_len
  int32   dim         (12288)
  bf16[seq_len * dim] flat row-major embedding
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/home/chris/models/flux2-klein-9B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seq", type=int, default=512)
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--mask-padding", action="store_true",
                    help="Zero out padding-position rows in the saved embedding "
                         "(workaround until the FluxTransformer supports attention masks).")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="Multiply the saved embedding by this scalar (use values "
                         "< 1.0 to tame Qwen3 outliers before NVFP4 quantization).")
    args = ap.parse_args()

    import torch
    from transformers import AutoTokenizer, AutoModel
    from diffusers import Flux2KleinPipeline

    root = Path(args.model_root)
    print(f"Loading tokenizer + Qwen3 from {root}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(root / "tokenizer"))
    enc = AutoModel.from_pretrained(
        str(root / "text_encoder"), torch_dtype=torch.bfloat16, device_map=args.device
    )
    enc.eval()

    with torch.no_grad():
        prompt_embeds = Flux2KleinPipeline._get_qwen3_prompt_embeds(
            text_encoder=enc,
            tokenizer=tok,
            prompt=args.prompt,
            max_sequence_length=args.seq,
        )
    prompt_embeds = prompt_embeds[0].to(torch.bfloat16).cpu()   # [seq, 12288]

    if args.scale != 1.0:
        prompt_embeds = (prompt_embeds.float() * args.scale).to(torch.bfloat16)
        print(f"scaled by {args.scale}: new rms="
              f"{prompt_embeds.float().pow(2).mean().sqrt().item():.4f}",
              file=sys.stderr)

    if args.mask_padding:
        # Re-tokenize to get the attention mask (cheap; matches what diffusers
        # internally builds before calling the encoder).
        messages = [{"role": "user", "content": args.prompt}]
        text = tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
        )
        am = tok(text, return_tensors="np", padding="max_length",
                 truncation=True, max_length=args.seq)["attention_mask"][0]
        keep = int(am.sum())
        print(f"mask_padding: keeping {keep}/{len(am)} real positions, "
              f"zeroing {len(am) - keep}", file=sys.stderr)
        mask = torch.tensor(am, dtype=torch.bfloat16).unsqueeze(-1)
        prompt_embeds = prompt_embeds * mask
    seq, dim = prompt_embeds.shape
    print(f"prompt_embeds shape: {tuple(prompt_embeds.shape)}", file=sys.stderr)
    print(f"  abs mean={prompt_embeds.abs().mean().item():.4f}  "
          f"max={prompt_embeds.abs().max().item():.3f}  "
          f"rms={prompt_embeds.float().pow(2).mean().sqrt().item():.4f}",
          file=sys.stderr)

    raw = prompt_embeds.view(torch.uint16).numpy()
    with open(args.out, "wb") as f:
        f.write(struct.pack("ii", seq, dim))
        f.write(raw.tobytes())
    print(f"Wrote {args.out}  "
          f"({Path(args.out).stat().st_size / 1048576:.1f} MiB)",
          file=sys.stderr)


if __name__ == "__main__":
    main()
