#!/usr/bin/env python3
"""
Run the diffusers Flux2 transformer for a SINGLE forward on fixed inputs and
dump the input packed latent + the output velocity, so the C++ FluxTransformer
can be compared numerically on identical inputs (localises any structural bug
independently of the denoise loop / scheduler / VAE).

Files written (each: int32 rows, int32 cols, bf16[rows*cols] row-major):
  --latent_out   the packed latent fed to the transformer   [256, 128]
  --vel_out      the transformer output (noise_pred)         [256, 128]

The text conditioning is read from an existing --embeds file (the same one the
C++ generate tool consumes), so both sides see identical prompt_embeds.
"""
import argparse, struct, sys
from pathlib import Path
import numpy as np


def read_embeds(path):
    with open(path, "rb") as f:
        seq, dim = struct.unpack("ii", f.read(8))
        raw = np.frombuffer(f.read(seq * dim * 2), dtype=np.uint16)
    return seq, dim, raw


def write_bf16(path, t):  # t: torch bf16 [rows, cols]
    rows, cols = t.shape
    raw = t.contiguous().view(torch.uint16).cpu().numpy()
    with open(path, "wb") as f:
        f.write(struct.pack("ii", rows, cols))
        f.write(raw.tobytes())
    print(f"wrote {path} [{rows},{cols}]", file=sys.stderr)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/home/chris/models/flux2-klein-9B")
    ap.add_argument("--embeds", required=True)
    ap.add_argument("--hp", type=int, default=16)   # patch grid H
    ap.add_argument("--wp", type=int, default=16)    # patch grid W
    ap.add_argument("--sigma", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--latent_out", default="/tmp/cmp_latent.bin")
    ap.add_argument("--vel_out", default="/tmp/cmp_vel.bin")
    args = ap.parse_args()

    global torch
    import torch
    from diffusers.models.transformers.transformer_flux2 import Flux2Transformer2DModel

    dev = "cuda"
    seq_img = args.hp * args.wp
    in_ch = 128

    print("loading transformer...", file=sys.stderr)
    tf = Flux2Transformer2DModel.from_pretrained(
        str(Path(args.model_root) / "transformer"), torch_dtype=torch.bfloat16
    ).to(dev).eval()

    # Fixed packed latent [1, seq_img, 128]
    g = torch.Generator(device=dev).manual_seed(args.seed)
    latent = torch.randn(1, seq_img, in_ch, generator=g, device=dev, dtype=torch.bfloat16)

    # prompt_embeds from the shared file
    seq_txt, dim, raw = read_embeds(args.embeds)
    embeds = torch.from_numpy(raw.copy()).view(torch.bfloat16).view(1, seq_txt, dim).to(dev)

    # 4-axis position ids (must match diffusers _prepare_latent_ids / _prepare_text_ids)
    t = torch.arange(1, device=dev)
    img_ids = torch.cartesian_prod(t, torch.arange(args.hp, device=dev),
                                   torch.arange(args.wp, device=dev), t).to(torch.int64)  # [256,4]
    txt_ids = torch.cartesian_prod(t, t, t, torch.arange(seq_txt, device=dev)).to(torch.int64)  # [512,4]

    sigma_t = torch.tensor([args.sigma], device=dev, dtype=torch.bfloat16)

    with torch.no_grad():
        out = tf(
            hidden_states=latent,
            timestep=sigma_t,            # transformer multiplies by 1000 internally
            guidance=None,
            encoder_hidden_states=embeds,
            txt_ids=txt_ids,
            img_ids=img_ids,
            return_dict=False,
        )[0]  # [1, seq_img, 128]

    print(f"out: shape={tuple(out.shape)} mean={out.float().mean():.4f} "
          f"std={out.float().std():.4f} absmax={out.float().abs().max():.3f}", file=sys.stderr)
    write_bf16(args.latent_out, latent[0])
    write_bf16(args.vel_out, out[0])


if __name__ == "__main__":
    main()
