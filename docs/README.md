# F2K_CUDA — From Diffusion Theory to a Blackwell GPU Implementation

*A long-form course on how FLUX.2-klein works and how it is implemented, from
scratch, in C++/CUDA/cuDNN for an NVIDIA GB10 (Blackwell) GPU.*

---

## Who this is for

You are comfortable with C++ and CUDA at a systems level — kernels, shared
memory, warps, occupancy — but you want to **understand the machine learning**:
what a diffusion transformer actually is, why FLUX.2-klein is shaped the way it
is, and how every mathematical object in the model becomes a tensor-core
instruction on a real GPU.

The course assumes no prior machine-learning background. It does assume you can
read CUDA C++ without hand-holding. Where we discuss GPU mechanics we go deep;
where we discuss the math we build it up from first principles.

## What this teaches

By the end you will be able to explain, and modify, **every stage** of the path
from a text prompt to a 1024×1024 PNG:

```
 "a cat on a skateboard"
        │
        ▼
 ┌──────────────┐   token ids    ┌──────────────────┐  [512 × 12288]   ┌───────────────────────┐
 │ BPE tokenizer│ ─────────────▶ │ Qwen3-8B encoder │ ───────────────▶ │                       │
 └──────────────┘                └──────────────────┘  conditioning    │   FLUX.2-klein MMDiT   │
                                                                        │   transformer         │
 ┌──────────────┐   N(0,1)       latent noise  [seq_img × 128]          │  (8 double + 24 single │
 │ noise + RNG  │ ──────────────────────────────────────────────────▶  │   stream blocks)      │
 └──────────────┘                                                       │                       │
                                                                        └───────────┬───────────┘
                                       4-step flow-matching denoise loop            │ velocity
                                       (calls the transformer each step)            ▼
                                                                        ┌───────────────────────┐
                                                                        │  unpatchify → latent  │
                                                                        │  → VAE decoder (cuDNN)│
                                                                        └───────────┬───────────┘
                                                                                    ▼
                                                                              1024×1024 RGB → PNG
```

Concretely, you will understand: rectified-flow diffusion and why it samples in 4
steps; the MMDiT architecture (joint text/image attention, AdaLN modulation,
4-axis RoPE); the Qwen3 text encoder; the VAE; and on the implementation side:
NVFP4/MXFP8 block-scaled quantization, CUTLASS tensor-core GEMM, a flash-attention
kernel taken all the way to peak HBM bandwidth with `mma.sync`/`ldmatrix`, the
cuDNN-9 graph API for convolutions, a from-scratch byte-level BPE tokenizer, and
the orchestration that ties it together. Part VIII then goes *beyond* text-to-image:
the **VAE encoder** closes the autoencoder loop, which unlocks **img2img,
inpainting, and upscaling** (all one idea — denoise from a noised real latent), and
a **persistent worker + streaming web UI** serve it interactively, watching the
image form live from a phone.

## The target machine

Everything here targets one specific computer, and the choices only make sense in
that light:

| | |
|---|---|
| **GPU** | NVIDIA GB10 (Blackwell), compute capability **sm_121** |
| **Compute** | 5th-gen tensor cores; ~1 PFLOP/s sparse FP4 |
| **Memory** | **128 GB unified LPDDR5X @ ~273 GB/s**, shared CPU+GPU (NVLink-C2C, cache-coherent) |
| **SMs** | 48, ~99 KiB opt-in shared memory per block |
| **CPU** | 10× Cortex-X925 + 10× Cortex-A725 (ARM, aarch64) |
| **Toolchain** | CUDA 13.0, cuDNN 9, g++ 13, CMake |

The single most important fact: **1 PFLOP/s of FP4 divided by 273 GB/s means this
machine is overwhelmingly memory-bandwidth-bound for inference.** That ratio
drives nearly every implementation decision — quantize weights to shrink bytes
moved, keep data on-chip, and treat a kernel as "done" when it saturates HBM, not
when it saturates the tensor cores. Chapter 11 makes this quantitative.

## How the course is organized

It is split into parts. Each *theory* chapter is followed by the *implementation*
that realizes it in this repo, with direct references to source files and the
tests that prove them correct. You can read straight through, or use the math
chapters as reference and live in the implementation chapters.

### Part 0 — Orientation
- **[00 — Overview & the big picture](00-overview.md)** — the dataflow end to end,
  the repo map, the hardware, and how to build and run.

### Part I — The mathematics of generation
- **01 — Diffusion from first principles** — forward/reverse processes, score,
  why iteratively denoising noise produces samples.
- **02 — Flow matching & rectified flow** — the objective FLUX actually trains on,
  velocity prediction, the sampler/scheduler, and why a *distilled* model needs
  only 4 steps.
- **03 — Latent diffusion & the VAE** — why we diffuse in latent space, the
  autoencoder, and the FLUX VAE's channel/patch layout.

### Part II — The FLUX.2-klein architecture
- **04 — Diffusion transformers (MMDiT)** — patches as tokens, the two-stream idea.
- **05 — Text conditioning with Qwen3** — why a 8B LLM is the text encoder, and the
  three-layer hidden-state concatenation.
- **06 — Rotary position embeddings** — RoPE, and the 4-axis variant for 2D images.
- **07 — Modulation / AdaLN** — turning a timestep into per-block scale/shift/gate.
- **08 — Double-stream blocks** — joint attention between text and image streams.
- **09 — Single-stream blocks** — the fused QKV+MLP layers.
- **10 — Assembling the transformer** — embedders, the block stack, final projection.

### Part III — GPU implementation foundations
- **11 — The Blackwell target & the roofline** — sm_121, unified memory, and making
  "bandwidth-bound" quantitative.
- **12 — Numerical formats & quantization** — BF16, FP8 (E4M3/MXFP8), NVFP4
  (E2M1 + block scales): what they are, why, and the measured quality tradeoffs.
- **13 — Tensor-core GEMM with CUTLASS** — block-scaled MMA, `fp4_gemm`/`fp8_gemm`,
  and the `Linear` class that wraps them.
- **14 — Weights on disk: the F2K format, loader & router** — a custom mmap-and-go
  container, multi-shard loading, and name→role tensor routing.
- **15 — The core kernel library** — RMSNorm, LayerNorm, GroupNorm, SiLU/SwiGLU,
  modulation, patchify, RoPE: small kernels, written once, reused everywhere.

### Part IV — Attention, in depth
- **16 — Attention and the memory wall** — the math, and why naïve attention is a
  bandwidth disaster.
- **17 — The kernel journey** — row-at-a-time → flash (online softmax) → WMMA
  tensor cores → register-resident O (occupancy) → `mma.sync`+`ldmatrix` (peak
  bandwidth). The full optimization story with real measurements (56.5 → 19.7 ms).
- **18 — `mma.sync` & `ldmatrix` up close** — fragment thread→register layouts, the
  inline PTX, and how we validated it bit-exactly.

### Part V — The VAE decoder on the GPU
- **19 — Convolutions with the cuDNN 9 graph API** — why the legacy API misses
  Blackwell tensor cores, NHWC, and engine selection (18.7 s → 1.35 s).
- **20 — Assembling the decoder** — resnet/groupnorm/upsample/attention, and the
  `post_quant_conv` and batch-norm de-normalization subtleties.

### Part VI — Text encoder & tokenizer
- **21 — The Qwen3 encoder** — grouped-query attention, per-head q/k norm, and the
  capture-layer extraction.
- **22 — A native byte-level BPE tokenizer** — bytes-to-unicode, the GPT-2/Qwen
  pretokenizer regex, BPE merges, and the chat template — matching HF bit-for-bit.

### Part VII — Integration & engineering practice
- **23 — The end-to-end pipeline** — `generate.cu` start to finish.
- **24 — Performance methodology** — profiling, occupancy analysis, the roofline,
  and how each bottleneck was actually found.
- **25 — Validation methodology** — golden tests against diffusers, cosine
  similarity, bit-exactness, and the debugging sagas that hardened the pipeline.

### Part VIII — Image conditioning & serving
- **26 — The VAE encoder** — the mirror of Chapter 20 that closes the autoencoder
  loop (image → latent), the asymmetric downsample pad, verified at cos 0.9997.
- **27 — Image conditioning** — img2img, inpainting, and upscaling as one idea:
  start the flow-matching denoise from a *noised real latent*. Strength schedules,
  RePaint masking, the hi-res pass.
- **28 — Serving & live preview** — the persistent worker (resident models, a
  line protocol), the web UI over Tailscale, and streaming the image as it forms.

### Appendices
- **A — Glossary** of every term and acronym.
- **B — The bug museum** — the real bugs (modulation scale/shift swap, the VAE
  "mush", the FP4 layout trap, the `gridDim.y` 2K wall, the streaming `SIGPIPE`)
  and what each one teaches.
- **C — Build & reproduce** — toolchain, CMake, running the tests, regenerating
  every figure and number in the course.

---

## A note on sourcing

Every implementation claim is anchored to a file in this repository and, where
possible, to a passing test under `tests/`. Performance numbers were measured on
the GB10 described above with `tools/bench_attention.cu` and `tools/generate.cu`.
When the course says "we measured X," there is a way to reproduce X in Appendix C.

> Status: **complete** — all 29 chapters (00–28) and three appendices (A–C) are
> written, each anchored to the source and verified against it. Parts 0–VII cover
> text-to-image; Part VIII adds the VAE encoder, image conditioning (img2img /
> inpainting / upscaling), and the serving layer with live preview.
