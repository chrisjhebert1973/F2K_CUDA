#!/usr/bin/env python3
"""
Quantization-quality study for the Flux2 transformer: fake-quantize all eligible
.weight tensors to (a) NVFP4 (our exact scheme) or (b) FP8 E4M3, run the full
diffusers pipeline, and compare images against the BF16 baseline.

Answers: how much of the "fuzziness" is fundamental to FP4, and how much would
FP8 recover?  Weight-only quant (activations stay BF16) — an upper bound on what
each weight dtype can achieve; our CUDA pipeline also quantizes activations, so
real NVFP4 is at or below the NVFP4 number here.
"""
import argparse, sys, copy
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model_root", default="/home/chris/models/flux2-klein-9B")
    ap.add_argument("--prompt", default="a photo of a cat on a sofa")
    ap.add_argument("--height", type=int, default=256)
    ap.add_argument("--width", type=int, default=256)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--configs", default="bf16,nvfp4,fp8")
    args = ap.parse_args()

    import torch
    from diffusers import Flux2KleinPipeline

    dev = "cuda"
    # E2M1 representable magnitudes (1 mantissa bit, max 6) and bucket midpoints.
    E2M1 = torch.tensor([0., .5, 1., 1.5, 2., 3., 4., 6.], device=dev)
    E2M1_BOUND = torch.tensor([.25, .75, 1.25, 1.75, 2.5, 3.5, 5.], device=dev)

    def q_e2m1(u):  # round magnitudes to the E2M1 grid, keep sign
        s = torch.sign(u)
        idx = torch.bucketize(u.abs(), E2M1_BOUND)
        return s * E2M1[idx]

    def fake_nvfp4(W):  # [N,K] -> NVFP4 (per-row, per-16 microblock E4M3 scale) -> dequant
        N, K = W.shape
        Wb = W.float().reshape(N, K // 16, 16)
        absmax = Wb.abs().amax(-1, keepdim=True)
        scale = (absmax / 6.0).clamp(min=1e-30)
        scale = scale.to(torch.float8_e4m3fn).float().clamp(min=1e-30)   # scale is E4M3
        q = q_e2m1(Wb / scale)
        return (q * scale).reshape(N, K).to(W.dtype)

    def fake_fp8(W):  # [N,K] -> E4M3 (per-row scale) -> dequant
        N, K = W.shape
        Wf = W.float()
        absmax = Wf.abs().amax(-1, keepdim=True).clamp(min=1e-30)
        scale = absmax / 448.0
        q = (Wf / scale).to(torch.float8_e4m3fn).float()
        return (q * scale).reshape(N, K).to(W.dtype)

    def eligible(name, p):  # mirror tools/f2k_convert.cu should_quantize
        return (name.endswith(".weight") and p.ndim == 2
                and p.shape[0] % 128 == 0 and p.shape[1] % 64 == 0)

    print("loading pipeline...", file=sys.stderr)
    pipe = Flux2KleinPipeline.from_pretrained(args.model_root, torch_dtype=torch.bfloat16).to(dev)

    tf = pipe.transformer
    orig = {n: p.detach().clone() for n, p in tf.named_parameters() if eligible(n, p)}
    n_q = sum(p.numel() for p in orig.values())
    print(f"{len(orig)} eligible weight tensors, {n_q/1e9:.2f}B params", file=sys.stderr)

    quantizers = {"bf16": None, "nvfp4": fake_nvfp4, "fp8": fake_fp8}
    imgs = {}
    for cfg in args.configs.split(","):
        q = quantizers[cfg]
        with torch.no_grad():
            for n, p in tf.named_parameters():
                if n in orig:
                    p.copy_(orig[n] if q is None else q(orig[n]))
        # report weight reconstruction error for this config
        if q is not None:
            num = den = 0.0
            for n, w0 in orig.items():
                wq = q(w0)
                num += (wq.float() - w0.float()).pow(2).sum().item()
                den += w0.float().pow(2).sum().item()
            print(f"[{cfg}] weight rel_l2 = {np.sqrt(num/den):.4f}", file=sys.stderr)
        g = torch.Generator(device=dev).manual_seed(args.seed)
        img = pipe(prompt=args.prompt, height=args.height, width=args.width,
                   num_inference_steps=args.steps, generator=g).images[0]
        path = f"/tmp/qsim_{cfg}.png"
        img.save(path)
        imgs[cfg] = np.asarray(img, dtype=np.float32)
        print(f"[{cfg}] wrote {path}", file=sys.stderr)

    if "bf16" in imgs:
        for cfg in imgs:
            if cfg == "bf16":
                continue
            d = np.abs(imgs[cfg] - imgs["bf16"]).mean() / 255 * 100
            print(f"image mean|Δ| vs bf16:  {cfg:6s} = {d:5.2f}%")


if __name__ == "__main__":
    main()
