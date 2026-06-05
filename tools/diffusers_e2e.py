#!/usr/bin/env python3
"""
Run diffusers Flux2KleinPipeline end-to-end on this machine as ground truth.
Saves PIL image to disk; confirms whether the model + weights + diffusers code
produces a recognizable cat-image (i.e., that the issue is in our C++ stack,
not in the model/weights themselves).
"""

import argparse
import sys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/home/chris/models/flux2-klein-9B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width",  type=int, default=256)
    ap.add_argument("--steps",  type=int, default=4)
    ap.add_argument("--seed",   type=int, default=0xCAFEBABE)
    ap.add_argument("--out",    required=True)
    args = ap.parse_args()

    import torch
    from diffusers import Flux2KleinPipeline

    print(f"Loading Flux2KleinPipeline from {args.model_root}", file=sys.stderr)
    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_root, torch_dtype=torch.bfloat16
    ).to("cuda")

    g = torch.Generator(device="cuda").manual_seed(args.seed)
    print(f"Generating: prompt={args.prompt!r}, {args.height}x{args.width}, "
          f"{args.steps} steps, seed={args.seed}", file=sys.stderr)
    image = pipe(
        prompt=args.prompt,
        height=args.height,
        width=args.width,
        num_inference_steps=args.steps,
        generator=g,
    ).images[0]

    image.save(args.out)
    print(f"Wrote {args.out} ({image.size})", file=sys.stderr)


if __name__ == "__main__":
    main()
