#!/usr/bin/env python3
"""
Run the full diffusers Flux2 pipeline but stop at output_type="latent", dumping
the [1, 32, H, W] latent that feeds vae.decode. Lets us push diffusers' OWN
ground-truth latent through our C++ VAE decoder to isolate VAE bugs from the
transformer denoise trajectory.

File: int32 C, int32 H, int32 W, float32[C*H*W] row-major (channel-major).
"""
import argparse, struct, sys
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/home/chris/models/flux2-klein-9B")
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=0xCAFEBABE)
    ap.add_argument("--out", default="/tmp/diff_latent.bin")
    args = ap.parse_args()

    import torch
    from diffusers import Flux2KleinPipeline

    pipe = Flux2KleinPipeline.from_pretrained(
        args.model_root, torch_dtype=torch.bfloat16
    ).to("cuda")
    g = torch.Generator(device="cuda").manual_seed(args.seed)
    lat = pipe(
        prompt=args.prompt, height=args.height, width=args.width,
        num_inference_steps=args.steps, generator=g, output_type="latent",
    ).images  # [1, C, H, W]
    lat = lat.float().cpu().numpy()[0]  # [C, H, W]
    C, H, W = lat.shape
    print(f"latent: [{C},{H},{W}] mean={lat.mean():.4f} std={lat.std():.4f} "
          f"absmax={np.abs(lat).max():.3f}", file=sys.stderr)
    with open(args.out, "wb") as f:
        f.write(struct.pack("iii", C, H, W))
        f.write(lat.astype(np.float32).tobytes())
    print(f"wrote {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
