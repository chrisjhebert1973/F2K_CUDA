# Chapter 23 — The End-to-End Pipeline

> *Goal of this chapter:* read `generate.cu` start to finish and see how every
> component of Parts I–VI is orchestrated into one call from a prompt to a PNG. We
> trace argument parsing, model construction, the three conditioning paths, latent
> initialization, the denoise loop, the latent→pixel boundary, and image output —
> with the timing of each stage. Anchored to `tools/generate.cu`.
>
> *Prerequisites:* all prior chapters (this is the integration).

---

## 23.1 The shape of `generate`

`generate` is the conductor; it owns no math, only orchestration. Its job:

```
 parse args → load+build models → get conditioning → init noise →
   denoise loop (call transformer ×N) → latent→pixels (VAE) → write image
```

The single self-contained invocation (Chapters 22, 0):

```bash
./generate --prompt "a cat on a skateboard" --res 1024 --precision fp8 \
           --steps 4 --seed 0xCAFEBABE --out cat.png
```

Flags: `--prompt`/`--tokens`/`--embeds` (the three conditioning entry points, §23.3),
`--res` (output resolution), `--precision {nvfp4,fp8}` (Chapter 12), `--steps`,
`--seed`, `--out` (`.png`→PNG via stb, else PPM, Chapter 22/15), plus
`--decode_latent` for VAE-isolation debugging.

## 23.2 Construction

```
 1. derive shapes from --res:  H_LAT = W_LAT = res/8,  patch grid = res/16,
        seq_img = (res/16)²  (validated: res%16==0 && seq_img%128==0)   ← the M%128 chain (Ch 13)
 2. load transformer F2K shards (transformer_mxfp8/ for fp8, transformer_f2k/ for nvfp4)
        → TensorRouter → FluxTransformer ctor  (~5.3 s, 124 Linears repacked; Ch 10/13/14)
 3. load VAE F2K → VAEDecoder ctor  (~0.5 s; Ch 20)
 4. if --prompt/--tokens: load Qwen3 F2K (4 shards) → QwenEncoder ctor
        (~10.5 s; capture_layers={8,17,26}; Ch 21)
 5. build the RoPE tables (img/txt/combined) from seq_img, seq_txt, H/W_patches (Ch 6/10)
 6. allocate device workspaces (transformer ~2.5 GiB + VAE ~3–5 GiB at 1024px; Ch 11)
```

All weights are mmap'd (no device copy; Chapter 14); "construction" cost is the
one-time weight repack into the tensor-core layout, not I/O.

## 23.3 Getting the conditioning (three paths)

```
 if --embeds <file>:   load precomputed [512×12288] BF16 directly  → d_txt
 elif --prompt "<t>":  BpeTokenizer.encode_for_flux(t, 512)  → ids  → QwenEncoder.forward → d_txt
 elif --tokens <file>: read [512] ids from file              → QwenEncoder.forward → d_txt
 else:                 random buffer (noise images; smoke only)
```

The `--prompt` path (Chapter 22 tokenizer → Chapter 21 encoder) is the native,
self-contained one (~0.28 s text encode). `--tokens` skips the tokenizer; `--embeds`
skips the encoder too (feeding diffusers' known-good conditioning — the key debugging
lever, Chapter 25). All three produce the same `d_txt` `[512×12288]` that the
ContextEmbedder consumes (Chapter 5/10).

## 23.4 Latent initialization

```
 d_latent ← N(0, 1) sampled with the seeded RNG, shape [seq_img × 128]
```

Standard normal, **not uniform** — the rectified-flow ODE starts at the Gaussian
noise endpoint $p_1=\mathcal N(0,\mathbf I)$ (Chapter 02); a uniform init starts off
the trained distribution and lands off-manifold (a real early bug, Appendix B). The
seed makes runs reproducible and lets you reroll composition (`--seed`).

## 23.5 The denoise loop

The heart, ~6.5 s of the run at 1024px (Chapter 0):

```c++
FlowMatchScheduler sched = FlowMatchScheduler::flux2_dynamic(n_steps, seq_img);  // Ch 2
for (int i = 0; i < n_steps; ++i) {
    auto temb = compute_timestep_embedding(sched.t(i) * 1000, 256);   // Ch 2
    mod = ModulationMLP.forward(temb);                                // 17 vectors, Ch 7
    v   = FluxTransformer.forward(d_latent, d_txt, mod, rope_tables); // velocity, Ch 4–10
    axpy_bf16(d_latent, v, sched.dt(i), n);                           // latent += dt·v, Ch 2
}
```

Each iteration is one Euler step of the ODE: build the timestep embedding for this
$t$, expand it into the 17 modulation vectors, evaluate the 9B velocity network, and
take the (variable-size, dynamic-shift) step. The transformer forward is the
expensive part (Parts III–IV); `axpy` is trivial. After 4 steps `d_latent` is the
clean latent.

## 23.6 The latent→pixel boundary

```
 1. bn de-normalize d_latent (the 128-ch patch latent):  z·√(var+ε)+mean   (Ch 3/20)
 2. unpatchify:  [seq_img × 128] → [32, H_LAT, W_LAT]                       (Ch 3)
 3. VAEDecoder.forward:  [32,H/8,W/8] → [3, H, W]   (applies post_quant_conv; Ch 19/20)
```

The bn de-norm (on the 128-channel patch latent, *before* unpatchify) and
`post_quant_conv` (inside the decoder) are the two boundary transforms whose absence
caused the patch-grid and mush bugs (Chapter 3/20; Appendix B). Order matters: de-norm
→ unpatchify → decode.

## 23.7 Output

```
 copy pixels to host → map [-1,1] → [0,255] → write_image(out_path)
        .png → stbi_write_png (RGB8)   else → PPM P6
```

`write_image` dispatches on the extension (Chapter 22/15). Done — a PNG you can open
in VS Code.

## 23.8 The full timing ledger (1024px, FP8, 4 steps)

| stage | time | chapter |
|---|---|---|
| transformer build | ~5.3 s | 10, 13, 14 |
| VAE build | ~0.5 s | 20 |
| Qwen3 build (—prompt/—tokens) | ~10.5 s | 21 |
| **text encode** (tokenize + encoder) | **~0.28 s** | 21, 22 |
| **denoise loop** (4 steps) | **~6.5 s** | 2, 4–18 |
| **VAE decode** | **~1.3 s** | 19, 20 |
| **total image** (excl. builds) | **~8 s** | — |

Builds are one-time per process; an interactive UI (ImGui, the project's other half)
amortizes them across many images. The per-image cost is dominated by the denoise
loop, and within it the GEMMs (Part III) and attention (Part IV) — which is why those
got the optimization effort. Compare to the project's start: ~133 s/image before the
attention (2.87×) and cuDNN-conv (13.8×) work; ~30 s after, and ~8 s/image once the
encoder/tokenizer went native and builds are amortized.

## 23.9 Isolation modes built into the conductor

`generate`'s flags double as debugging instruments — the conductor is built to let
you cut the pipeline at any stage and inject a known-good piece (Chapter 25):

- `--embeds` — bypass tokenizer+encoder (test the transformer alone).
- `--tokens` — bypass the tokenizer (test the encoder).
- `--decode_latent <file>` — bypass the transformer entirely, decode a given latent
  (test the VAE alone — this is how the "mush" bug was localized).
- `--seed` — vary only the noise (isolate latent vs prompt influence).
- `--steps` — vary the sampler depth (distinguish undercooking from structural bugs).

This is not incidental; the ability to inject a reference intermediate at each
boundary is what made the multi-stage port debuggable (Chapter 25).

## 23.10 Summary and what to carry forward

- `generate.cu` is pure **orchestration**: parse → build (mmap, repack) → condition
  (one of 3 paths) → init $\mathcal N(0,\mathbf I)$ → **denoise loop** (Euler steps
  calling the transformer) → **bn de-norm → unpatchify → VAE** → write image.
- The **denoise loop** (~6.5 s) dominates the per-image cost; builds (~16 s) are
  one-time and amortized by a UI.
- The **three conditioning paths** and `--decode_latent`/`--seed`/`--steps` are
  built-in **isolation modes** for debugging (Chapter 25).
- The two **latent-boundary transforms** (bn de-norm, post_quant_conv) and the
  **Gaussian init** are easy-to-miss correctness requirements (Appendix B).

Chapter 24 steps back to the *method*: how every bottleneck in that ledger was
actually found and fixed.

---

### Exercises

1. **Trace a run.** For `--prompt "x" --res 512 --precision nvfp4 --steps 4`, list
   every stage, its shape, and which model dir it loads. What is `seq_img`?
2. **Why Gaussian.** Explain (Chapter 2) why the latent must be $\mathcal N(0,\mathbf
   I)$ and what a uniform init does to the ODE trajectory.
3. **Amortization.** Given the build vs per-image times, compute the break-even image
   count where native build cost is negligible, and argue why an interactive UI
   changes the optimization priorities.
4. **Isolation.** You get a wrong image. Design a sequence of `--embeds`,
   `--decode_latent`, `--seed` runs that localizes the fault to tokenizer, encoder,
   transformer, or VAE.

*Next: [Chapter 24 — Performance methodology](24-performance.md).*
