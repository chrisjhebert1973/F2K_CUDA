#!/usr/bin/env python3
"""
Golden dump for the VAE *encoder*. Loads an image, preprocesses it to model
space [-1,1], runs diffusers AutoencoderKLFlux2.encode, and writes BOTH:
  --out_img : the preprocessed input image  [3, H, W]      (so the C++ encoder
              consumes the EXACT same pixels — no resize/PNG-decode mismatch)
  --out_lat : the posterior mean latent      [32, H/8, W/8] (the deterministic
              latent used for img2img)

File format (matches diffusers_latent_dump.py):
  int32 C, int32 H, int32 W, float32[C*H*W] row-major (channel-major).
"""
import argparse, struct, sys
import numpy as np


def write_chw(path, arr):  # arr: [C,H,W] float
    C, H, W = arr.shape
    with open(path, "wb") as f:
        f.write(struct.pack("iii", C, H, W))
        f.write(arr.astype(np.float32).tobytes())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/home/chris/models/flux2-klein-9B")
    ap.add_argument("--image", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--out_img", default="/tmp/vae_enc_img.bin")
    ap.add_argument("--out_lat", default="/tmp/vae_enc_lat.bin")
    args = ap.parse_args()

    import torch
    from PIL import Image
    from diffusers import AutoencoderKLFlux2

    img = Image.open(args.image).convert("RGB").resize((args.size, args.size), Image.BICUBIC)
    a = np.asarray(img, dtype=np.float32) / 255.0       # [H,W,3] in [0,1]
    a = a * 2.0 - 1.0                                   # [-1,1]
    chw = np.transpose(a, (2, 0, 1)).copy()             # [3,H,W]
    write_chw(args.out_img, chw)

    vae = AutoencoderKLFlux2.from_pretrained(
        args.model_root, subfolder="vae", torch_dtype=torch.bfloat16).to("cuda").eval()
    x = torch.from_numpy(chw[None]).to("cuda", torch.bfloat16)   # [1,3,H,W]
    with torch.no_grad():
        posterior = vae.encode(x).latent_dist
        mean = posterior.mean.float().cpu().numpy()[0]  # [32,H/8,W/8]
    print(f"mean latent: {list(mean.shape)} mean={mean.mean():.4f} std={mean.std():.4f} "
          f"absmax={np.abs(mean).max():.3f}", file=sys.stderr)
    write_chw(args.out_lat, mean)
    print(f"wrote {args.out_img} and {args.out_lat}", file=sys.stderr)


if __name__ == "__main__":
    main()
