#!/usr/bin/env python3
"""
Qwen3-8B golden-value extractor.

Runs HuggingFace's Qwen3 forward on a prompt with output_hidden_states=True and
dumps the per-layer hidden states (and input token IDs) to a binary file that
the C++ test can read for numerical comparison against QwenEncoder.forward.

Layout of out.bin:
  int32  seq_len
  int32  n_layers          (always 36 for Qwen3-8B)
  int32  hidden            (always 4096)
  int32[seq_len]           token IDs
  bfloat16[n_layers+1][seq_len][hidden]
      index 0   : embed_tokens output  (before any transformer layer)
      index 1..N: post-layer i hidden state

Usage:
  pip install transformers torch safetensors accelerate
  python3 tools/qwen3_golden.py \
      --model /home/chris/models/flux2-klein-9B/text_encoder \
      --prompt "A photo of a cat" \
      --seq 128 \
      --out /tmp/qwen3_golden.bin
"""

import argparse
import struct
import sys
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="Path to text_encoder/ dir")
    ap.add_argument("--tokenizer", default=None,
                    help="Tokenizer dir (defaults to same as --model unless its parent has tokenizer/)")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--seq", type=int, default=128, help="Pad/truncate to this length")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    import torch
    from transformers import AutoModel, AutoTokenizer

    model_path = Path(args.model)
    tok_path = Path(args.tokenizer) if args.tokenizer else (model_path.parent / "tokenizer")
    if not tok_path.exists():
        tok_path = model_path

    print(f"Loading tokenizer from {tok_path}", file=sys.stderr)
    tok = AutoTokenizer.from_pretrained(str(tok_path))

    # Encode (no chat template — caller can apply one in the prompt string if needed).
    ids = tok.encode(args.prompt, add_special_tokens=False)
    if len(ids) > args.seq:
        ids = ids[: args.seq]
        print(f"WARNING: truncated to {args.seq} tokens", file=sys.stderr)
    pad = args.seq - len(ids)
    eos = tok.eos_token_id if tok.eos_token_id is not None else 151643
    ids = ids + [eos] * pad
    assert len(ids) == args.seq
    print(f"Prompt tokens (len={len(ids)}): {ids[:16]}...", file=sys.stderr)

    print(f"Loading model from {model_path} (bf16)", file=sys.stderr)
    model = AutoModel.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16,
        device_map=args.device,
    )
    model.eval()

    in_ids = torch.tensor([ids], dtype=torch.long, device=args.device)
    with torch.no_grad():
        out = model(input_ids=in_ids, output_hidden_states=True, use_cache=False)
    # hidden_states is a tuple of length n_layers+1
    hs = out.hidden_states
    n_layers = len(hs) - 1
    hidden = hs[0].shape[-1]
    print(f"Got {n_layers} layer hidden states, hidden={hidden}", file=sys.stderr)

    # Stack into a single [n_layers+1, seq, hidden] bfloat16 array.
    all_hs = torch.stack([h[0].to(torch.bfloat16).cpu() for h in hs], dim=0)
    np_hs = all_hs.view(torch.uint16).numpy()  # raw BF16 bytes
    print(f"Hidden states shape: {tuple(all_hs.shape)}", file=sys.stderr)

    # Write binary.
    with open(args.out, "wb") as f:
        f.write(struct.pack("iii i", args.seq, n_layers, hidden, 0))
        f.write(np.array(ids, dtype=np.int32).tobytes())
        f.write(np_hs.tobytes())
    print(f"Wrote {args.out} ({Path(args.out).stat().st_size / 1048576:.1f} MiB)", file=sys.stderr)


if __name__ == "__main__":
    main()
