#!/usr/bin/env python3
"""
Tokenize a prompt with the Qwen2/3 tokenizer and write fixed-length int32 IDs.

Format of output file:
  int32  seq_len
  int32[seq_len]   token IDs (padded with eos_token_id)

Usage:
  python tools/encode_prompt.py "A photo of a cat on a sofa" /tmp/tokens.bin
  python tools/encode_prompt.py --seq 128 "..." /tmp/tokens.bin
"""

import argparse
import struct
import sys
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("prompt")
    ap.add_argument("out_path")
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--tokenizer",
                    default="/home/chris/models/flux2-klein-9B/tokenizer")
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # Diffusers Flux2 convention: wrap prompt with chat template, then pad to
    # max_sequence_length (default 512) using the tokenizer's `padding` behavior.
    messages = [{"role": "user", "content": args.prompt}]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=False
    )
    enc = tok(text, return_tensors="np",
              padding="max_length", truncation=True, max_length=args.seq)
    ids = enc["input_ids"][0].tolist()
    assert len(ids) == args.seq, f"got {len(ids)} != {args.seq}"

    print(f"Prompt: {args.prompt!r}", file=sys.stderr)
    print(f"Templated text: {text!r}", file=sys.stderr)
    print(f"Tokens ({len(ids)}): {ids[:16]}...", file=sys.stderr)

    with open(args.out_path, "wb") as f:
        f.write(struct.pack("i", args.seq))
        import array
        array.array("i", ids).tofile(f)
    print(f"Wrote {args.out_path}", file=sys.stderr)


if __name__ == "__main__":
    main()
