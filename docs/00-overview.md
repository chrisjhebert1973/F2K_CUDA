# Chapter 00 — Overview & the Big Picture

> *Goal of this chapter:* build an accurate mental model of the whole system —
> what data flows where, which file does what, and what the hardware constraints
> are — so that every later chapter has a place to attach. Nothing here is deep;
> everything here is a map.

---

## 0.1 What the program does, in one sentence

Given a text prompt and a random seed, `generate` produces an image by **starting
from pure Gaussian noise in a compressed "latent" space and repeatedly nudging it
toward the prompt**, then decoding the cleaned-up latent into pixels.

That sentence hides three different neural networks and a sampling loop. The rest
of the course unpacks it. This chapter just names the pieces.

## 0.2 The three models

FLUX.2-klein is not one model; it is a *pipeline* of three, each doing a distinct
job. This is the single most important structural fact to internalize early.

1. **The text encoder (Qwen3-8B).** A large language model, used here not to
   generate text but to *read* the prompt and turn it into a numerical summary —
   a `[512 × 12288]` tensor of "conditioning" that the image model can attend to.
   Surprisingly, FLUX.2-klein's text encoder is a full 8-billion-parameter LLM.
   (Why an LLM and not a smaller text model is the subject of Chapter 05.)

2. **The diffusion transformer (the MMDiT).** This is the heart of FLUX. It takes
   a noisy image latent plus the text conditioning plus a "how noisy is this right
   now" timestep, and predicts a **velocity** — the direction to move the latent
   to make it less noisy and more like the prompt. It is called repeatedly. It is
   ~9 billion parameters arranged as 8 "double-stream" blocks followed by 24
   "single-stream" blocks (Chapters 08–09).

3. **The VAE decoder.** The transformer works in a compressed latent space
   (roughly 1/8 the spatial resolution, with 16/32 channels instead of 3). Once
   the latent is clean, the VAE decoder — a stack of convolutions — expands it
   back into a full-resolution RGB image (Chapters 19–20).

Three models, three jobs: **read the prompt, denoise the latent, decode to
pixels.**

## 0.3 The dataflow, stage by stage

Here is the full path. Keep this diagram nearby; later chapters zoom into each box.

```
                         ┌─────────────────────────────────────────────────────────┐
   PROMPT (string)       │  TEXT PATH                                              │
   "a cat ..."   ──────▶ │  ┌──────────────┐    token ids [512]   ┌──────────────┐ │
                         │  │ BPE tokenizer│ ───────────────────▶ │ Qwen3-8B     │ │
                         │  │ (ch 22)      │  + chat template     │ encoder      │ │
                         │  └──────────────┘                      │ (ch 21)      │ │
                         │                                        └──────┬───────┘ │
                         │              conditioning [512 × 12288]       │         │
                         └───────────────────────────────────────────────┼─────────┘
                                                                          │
   SEED (uint32)                                                          │
        │                                                                 ▼
        ▼                                                  ┌──────────────────────────────┐
  ┌───────────┐  latent noise                              │  DENOISE LOOP (ch 02, 23)    │
  │ N(0,1)    │  [seq_img × 128]  ───────────────────────▶ │  for step in 0..3:           │
  │ init      │                                            │    v = Transformer(latent,   │
  └───────────┘                                            │            cond, t_step)     │
                                                           │    latent += dt · v          │
                                  ┌────────────────────────┤  (the Transformer is ch 04–10)│
                                  │   each step calls the   └──────────────┬───────────────┘
                                  │   full 32-block model                  │ clean latent
                                  │                                        ▼
                                  │                          ┌──────────────────────────┐
                                  │                          │ unpatchify → [32,H/8,W/8]│ (ch 03)
                                  │                          │ VAE decoder (cuDNN, ch 19)│
                                  │                          └──────────────┬───────────┘
                                  ▼                                         ▼
                          (transformer internals)                  RGB [3,H,W] → PNG (ch 23)
```

Two things to notice now:

- **The transformer is called once per denoise step.** With the distilled
  4-step schedule that is 4 forward passes of a 9B model. Everything about
  performance is dominated by making that forward pass cheap. (This is why
  Chapters 11–18 exist.)

- **The text path runs once**; its `[512 × 12288]` output is reused across all 4
  steps. The latent path runs every step.

## 0.4 A first contact with the numbers

For a 1024×1024 image, FP8 precision, 4 steps, on the GB10 (measured this session):

| Stage | Time | Notes |
|---|---|---|
| Build transformer (load + repack weights) | ~5.3 s | one-time per process |
| Build VAE | ~0.5 s | one-time |
| Load + build Qwen3 encoder | ~10.5 s | one-time; only with `--prompt`/`--tokens` |
| **Text encode** (tokenize + encoder forward) | **~0.28 s** | native; replaced a 98 s Python path (ch 22) |
| **Denoise loop** (4 steps) | **~6.5 s** | the hot path; ~1.6 s/step |
| **VAE decode** | **~1.3 s** | cuDNN graph API (ch 19) |

The denoise loop is the thing worth optimizing, and within it, the attention and
the GEMMs. Chapter 17 tells the story of taking *just the attention* from 56.5 ms
to 19.7 ms per call — a 2.87× win that is felt directly in that 6.5 s.

## 0.5 The repository map

The code is organized so that the curriculum can follow it almost one-to-one.

```
F2KTest/
├── src/
│   ├── common/                  # host-side, no CUDA — formats, loading, tokenizer
│   │   ├── f2k_format.{h,cpp}      # the on-disk weight container (ch 14)
│   │   ├── f2k_model_loader.*      # multi-shard mmap loader (ch 14)
│   │   ├── tensor_router.*         # tensor name → structural role (ch 14)
│   │   ├── safetensors.*           # reads HF .safetensors during conversion
│   │   └── bpe_tokenizer.{h,cpp}   # native byte-level BPE tokenizer (ch 22)
│   │
│   └── backend/cuda/            # the GPU implementation
│       ├── fp4_gemm.*  fp8_gemm.*  # CUTLASS block-scaled tensor-core GEMM (ch 13)
│       ├── linear.*                # the quantized Linear layer over those GEMMs (ch 13)
│       ├── attention.*             # the flash-attention kernels (ch 16–18)
│       ├── rope*, rmsnorm, ...     # kernels/ : the small reusable kernels (ch 15)
│       ├── modulation_mlp.*        # timestep → modulation vectors (ch 07)
│       ├── double_stream_block.*   # text+image joint-attention block (ch 08)
│       ├── single_stream_block.*   # fused QKV+MLP block (ch 09)
│       ├── flux_transformer.*      # the top-level 32-block model (ch 10)
│       ├── qwen_*.*                # the Qwen3 text encoder (ch 21)
│       ├── conv2d.* vae_*.*        # the VAE decoder (ch 19–20)
│       └── sampler.*               # the flow-matching scheduler & step (ch 02)
│
├── tools/
│   ├── generate.cu                 # the end-to-end CLI: prompt → PNG (ch 23)
│   ├── f2k_convert.cu              # safetensors → F2K quantized weights (ch 12, 14)
│   ├── bench_attention.cu          # the attention microbenchmark (ch 17, 24)
│   ├── mma_unit.cu                 # standalone PTX/ldmatrix validation (ch 18)
│   └── *.py                        # diffusers/HF reference dumpers for validation (ch 25)
│
├── tests/                          # one ctest per component; the correctness contract
├── third_party/                    # cutlass, cudnn(via system), imgui, glfw, stb, json (submodules)
└── docs/                           # this course
```

A useful habit: when a chapter discusses a concept, open the named file beside it.
The implementation chapters are written to be read *with the source*, not instead
of it.

## 0.6 The one hardware fact that explains everything

We will repeat this until it is reflex. The GB10 offers on the order of **1
PFLOP/s of FP4 compute** but only **~273 GB/s of memory bandwidth**. The ratio of
compute to bandwidth is enormous — thousands of FLOPs are available for every byte
that can be moved.

Inference is not like training. In inference, each weight is used a small number
of times per forward pass, so the workload is dominated by *reading the weights
and activations from memory*, not by the arithmetic. On this machine that means:

1. **Shrink the bytes.** Store weights in 4-bit (NVFP4) or 8-bit (MXFP8) instead
   of 16-bit, and you move 4×/2× fewer bytes — a near-linear speedup on a
   bandwidth-bound workload. This is why Chapters 12–13 are about quantization.

2. **Keep data on-chip.** A kernel that re-reads the same data from HBM is
   wasting the scarcest resource. Flash attention exists precisely to stop
   re-reading K and V from memory (Chapter 16).

3. **"Fast enough" = "saturates HBM."** The attention kernel in Chapter 17 is
   declared finished not when the tensor cores are busy but when it hits ~279
   GB/s — essentially the memory ceiling. There is nothing left to win because
   the bytes simply cannot arrive faster.

4. **Unified memory changes the data-movement model.** CPU and GPU share the same
   128 GB, cache-coherently. There is no `cudaMemcpy` of weights across PCIe;
   the model is mmap'd and the GPU reads it in place (Chapter 14). A 9B model in
   FP8 (~9 GB) or NVFP4 (~5 GB) simply fits, with room to spare.

If you ever wonder "why did they do it *that* way," the answer is usually
"because the machine is bandwidth-bound." Chapter 11 turns this intuition into a
roofline you can compute.

## 0.7 Building and running

The full build & toolchain details are Appendix C; here is the orientation
version.

```bash
# configure + build (CUDA 13, cuDNN 9, CMake; targets sm_121)
cmake -S . -B build
cmake --build build -j

# run the whole pipeline from a prompt — single self-contained call, no Python
./build/generate --prompt "a cat on a skateboard" \
                 --res 1024 --precision fp8 --steps 4 --out cat.png

# correctness suite (each component has a golden test)
cd build && ctest
```

`generate` has three ways to supply text conditioning, which map to three points
in the pipeline and are useful for isolating bugs:

- `--prompt "<text>"` — the full native path (tokenizer → Qwen3 → transformer).
- `--tokens <file>` — skip the tokenizer; feed precomputed token ids.
- `--embeds <file>` — skip the encoder too; feed a precomputed `[512 × 12288]`
  conditioning tensor (used to compare against the reference diffusers pipeline,
  Chapter 25).

And `--precision {nvfp4,fp8}` selects the weight format (Chapter 12), `--res` the
output resolution, `--steps` the number of denoise steps, `--seed` the RNG seed.

## 0.8 How the model gets onto disk

One preparatory step happens before any image is generated: the original
FLUX.2-klein weights (distributed by Black Forest Labs as `.safetensors`) are
**converted and quantized** into this project's own `.f2k1` container by
`tools/f2k_convert.cu`. That tool reads each tensor, decides whether it is
eligible for quantization, packs it into NVFP4 or MXFP8 with the appropriate
block scales, and writes a memory-mappable file. The runtime then mmaps those
files and never parses anything on the hot path. Chapters 12 (the formats) and 14
(the container) cover this; you only need to know now that it happened.

## 0.9 What to read next

- If you want the **theory** first: go to Chapter 01 (diffusion) and read Part I
  straight through. You will understand *why* the pipeline has the shape it does
  before seeing any CUDA.
- If you want the **implementation** first: jump to Chapter 11 (the Blackwell
  target) and Part III. You will see how the machine is used before learning the
  math it is computing.
- If you came for the **attention optimization** specifically (the headline
  engineering result): Part IV (Chapters 16–18) is self-contained enough to read
  on its own, given this overview.

Either way, keep §0.6 in mind. Everything bends toward bandwidth.

---

*Next: [Chapter 01 — Diffusion from first principles](01-diffusion.md).*
