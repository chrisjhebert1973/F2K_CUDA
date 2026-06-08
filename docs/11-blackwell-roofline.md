# Chapter 11 — The Blackwell Target & the Roofline

> *Goal of this chapter:* turn "the machine is bandwidth-bound" from a slogan
> (Chapter 00) into a **computable** statement. We specify the GB10, build the
> roofline model, compute its ridge point at each precision, and then compute the
> arithmetic intensity of the three workloads that matter — the big transformer
> GEMMs, attention, and small-batch projections — to see *honestly* which are
> memory-bound and which sit near the compute ceiling. The conclusion is more
> nuanced than the slogan, and it is what justifies every choice in Part III–IV.
>
> *Prerequisites:* Part II; basic GPU execution model (SMs, warps, HBM).

---

## 11.1 The GB10, precisely

| | |
|---|---|
| GPU | NVIDIA GB10 (Blackwell), compute capability **sm_121** |
| SMs | **48**; ~99 KiB opt-in shared memory / block |
| Memory | **128 GB LPDDR5X, unified, ~273 GB/s**, cache-coherent CPU+GPU (NVLink-C2C) |
| Tensor-core peak (theoretical) | ~1 PFLOP/s **sparse** FP4 (≈500 TFLOP/s dense FP4) |
| Tensor-core peak (measured here) | FP4 GEMM **252 TFLOP/s**, FP8 GEMM **161 TFLOP/s** (`test_fp4_gemm`/`test_fp8_gemm`, 2048³) |
| Host | 10× Cortex-X925 + 10× Cortex-A725 (aarch64) |

Two facts shape everything. First, **unified memory**: the 128 GB is shared
coherently between CPU and GPU, so model weights are mmap'd and read in place — no
PCIe `cudaMemcpy`, no host/device staging on the hot path (Chapter 14). A 9B model
in FP8 (~9 GB) or NVFP4 (~5 GB) simply *fits*, with tens of GB to spare for
activations. Second, the **compute:bandwidth ratio is enormous** — even at the
*measured* 252 TFLOP/s of dense FP4, the chip can do ~920 FLOPs for every byte it
can read from HBM. Whether a kernel is limited by compute or by memory depends on
whether it does more or fewer than ~920 FLOPs per byte.

## 11.2 The roofline model

The roofline (Williams et al., 2009) is the right lens. Plot achievable
performance $P$ (FLOP/s) against **arithmetic intensity** $I$ (FLOPs per byte of
HBM traffic). Two ceilings bound any kernel:

$$
P \le \min\big(\;\underbrace{P_{\max}}_{\text{compute ceiling}},\; \underbrace{B_{\max}\cdot I}_{\text{memory ceiling}}\;\big),
$$

where $P_{\max}$ is peak FLOP/s and $B_{\max}$ is peak bytes/s (273 GB/s). The two
lines cross at the **ridge point**

$$
I_{\text{ridge}} = \frac{P_{\max}}{B_{\max}} .
$$

- If a kernel's intensity $I < I_{\text{ridge}}$, it is **memory-bound**: it cannot
  feed the tensor cores fast enough, and the only way to go faster is to move fewer
  bytes (quantize, fuse, keep data on-chip) or move them faster.
- If $I > I_{\text{ridge}}$, it is **compute-bound**: limited by FLOP throughput,
  and the lever is using a faster instruction (lower precision, sparsity, better
  tensor-core utilization).

Ridge points for the GB10 at the precisions we use (using **measured** peaks; the
theoretical ones are ~2× higher and shift the ridge right):

| precision | peak (measured) | $I_{\text{ridge}}$ = peak / 273 GB/s |
|---|---|---|
| FP4 (NVFP4) | 252 TFLOP/s | **≈ 923 FLOP/byte** |
| FP8 (MXFP8) | 161 TFLOP/s | **≈ 590 FLOP/byte** |
| BF16 (CUDA cores / lower TC) | ~tens of TFLOP/s | a few hundred FLOP/byte |

Now we can classify the real workloads by computing their $I$.

## 11.3 Workload 1 — a big transformer GEMM

Take a hidden-to-hidden projection in the transformer at 1024px: $Y = X W$ with
$M = 4608$ tokens, $K = N = 4096$. FLOPs $= 2MNK$. Bytes $=$ read $X$ ($MK\cdot
b_a$) + read $W$ ($KN\cdot b_w$) + write $Y$ ($MN\cdot b_a$), with $b_a=2$ (BF16
activations) and $b_w$ the weight byte width.

$$
\text{FLOPs} = 2\cdot4608\cdot4096\cdot4096 \approx 1.55\times10^{11}.
$$

With **FP4** weights ($b_w = 0.5$):

$$
\text{bytes} = \underbrace{4608\cdot4096\cdot2}_{X\,=\,37.7\text{MB}}
             + \underbrace{4096\cdot4096\cdot0.5}_{W\,=\,8.4\text{MB}}
             + \underbrace{4608\cdot4096\cdot2}_{Y\,=\,37.7\text{MB}}
             \approx 8.4\times10^{7},
$$

$$
I = \frac{1.55\times10^{11}}{8.4\times10^{7}} \approx \mathbf{1845\ \text{FLOP/byte}}.
$$

That is **above** the FP4 ridge of ~923 — so at 1024px the big GEMMs are
**compute-bound**, not memory-bound. The activations (not the 4-bit weights)
dominate the byte count here because there are so many tokens. Lowering weight
precision still helps (FP4 doubles the FLOP ceiling vs FP8, raising $P_{\max}$),
but more bytes won't be the lever for these.

> This refines Chapter 00's slogan. At high resolution the dense projections are
> compute-bound; the win from FP4 there is the *higher FLOP ceiling*, and the win
> from fusion (Chapter 09) is reading the shared activation once.

## 11.4 Workload 2 — attention

Attention's traffic is dominated by streaming K and V. For one head over $S$ tokens
at head-dim $D$, a flash kernel reads K and V once per query *tile* (Chapter 16),
and the useful FLOPs are $QK^\top$ + $PV$ $= 4 S^2 D$ per head. At $S=4608$,
$D=128$, $H=32$, the benchmark (Chapter 17) measured the *final* kernel at **~279
GB/s and ~17.6 TFLOP/s** — i.e. it is pinned to the **memory ceiling** (279 ≳ 273
GB/s peak) and nowhere near any compute ceiling. Attention's intensity is low
because each byte of K/V is used for only a handful of multiply-adds before being
discarded.

$$
I_{\text{attn}} \approx \frac{17.6\,\text{TFLOP/s}}{279\,\text{GB/s}} \approx 63\ \text{FLOP/byte} \;\ll\; 923 = I_{\text{ridge}}.
$$

**Attention is decisively memory-bound.** This is *why* Part IV is an exercise in
moving fewer bytes and keeping K/V on-chip, and why the kernel is declared finished
when it hits ~279 GB/s — there is no compute headroom to chase, only bytes to save,
and the bytes are already saturating HBM.

## 11.5 Workload 3 — small-batch projections (the text encoder, low-res)

The Qwen3 encoder runs at $S=512$ tokens; many of its projections have small $M$.
And at low image resolutions (256px, $M=256$ tokens) the transformer GEMMs shrink
too. Redo §11.3 with $M=256$, FP4 weights:

$$
\text{bytes} \approx \underbrace{256\cdot4096\cdot2}_{X=2.1\text{MB}} + \underbrace{8.4\text{MB}}_{W} + \underbrace{2.1\text{MB}}_{Y} \approx 1.26\times10^7,\quad
I = \frac{2\cdot256\cdot4096^2}{1.26\times10^7} \approx \mathbf{682\ \text{FLOP/byte}}.
$$

Below the FP4 ridge — **memory-bound**, and now the **weight** bytes dominate
(8.4 MB of W vs 4.2 MB of activations). Here 4-bit weights pay off directly: NVFP4
moves $4\times$ fewer weight bytes than BF16, a near-linear speedup. The general
rule: **small batch → weight-bandwidth-bound → quantization is a direct win; large
batch → compute-bound → low-precision FLOP ceiling is the win.** Either way, lower
precision helps; the *mechanism* differs.

## 11.6 The honest synthesis

| workload | regime at 1024px | dominant cost | the lever |
|---|---|---|---|
| big transformer GEMMs | compute-bound (just above ridge) | tensor-core FLOPs | low-precision FLOP ceiling (FP4>FP8), fusion |
| attention | **memory-bound** (at HBM peak) | K/V streaming | flash tiling, on-chip reuse (Part IV) |
| small-batch / low-res GEMMs | memory-bound | weight bytes | quantization (NVFP4) |
| VAE convolutions | memory-bound + kernel-selection | conv engine | cuDNN graph API, NHWC (Chapter 19) |
| weight loading at build | one-time, byte-bound | reading the model | mmap, on-disk quant (Chapter 14) |

So Chapter 00's "everything bends toward bandwidth" is the right *design pressure*
— the things that hurt (attention, weight loading, low-res/small-batch) are
memory-bound, and the architecture's own choices (fused GEMMs, quantized weights)
are bandwidth moves. But the rigorous statement is: **the workload is a mix, and the
roofline tells you, per kernel, whether to cut bytes or raise the FLOP ceiling.**
Both roads pass through low precision — which is why Chapter 12 is next.

## 11.7 Why low precision is the master key

Notice that **every** row of the table above is helped by quantization, for one of
two reasons:

1. **Memory-bound kernels** move fewer bytes when weights/activations are smaller —
   a near-linear speedup right up to the point they become compute-bound.
2. **Compute-bound kernels** run on faster instructions: Blackwell's tensor cores
   do FP4 at ~2× the rate of FP8 and far above BF16, so dropping precision *raises
   the compute ceiling* $P_{\max}$.

Quantization simultaneously slides a kernel down-and-right on the roofline (fewer
bytes → higher $I$) **and** lifts the compute ceiling (faster instruction). The
only cost is numerical accuracy — and Chapter 12 measures exactly how much, and why
FP8 buys back most of what FP4 gives up.

## 11.8 Two more Blackwell-specific levers

- **Shared memory as the on-chip buffer.** The ~99 KiB/block opt-in smem is the
  staging area that makes memory-bound kernels fast: flash attention tiles K/V into
  it (Chapter 16–17), and the occupancy story of Chapter 17 is entirely about
  *fitting more CTAs per SM* by using less smem — a direct roofline move (more CTAs
  → more outstanding memory requests → closer to $B_{\max}$).
- **Unified memory removes a whole category of traffic.** On a discrete GPU you'd
  budget PCIe transfers for the 5–9 GB model and the reference images; here there
  are none. The roofline's $B_{\max}$ is the *only* memory ceiling that matters,
  and there is no separate "get the data onto the device" phase.

## 11.9 Where this lives in the code / how to reproduce

- GEMM peaks: `tools/` builds `test_fp4_gemm` / `test_fp8_gemm` (Chapter 13),
  printing sustained TFLOP/s.
- Attention bandwidth/TFLOP/s: `tools/bench_attention.cu` (Chapter 17) prints
  GB/s and TFLOP/s per shape, plus the achieved occupancy.
- The device properties (SMs, smem, peak) are printed by `bench_attention` at
  startup. Appendix C lists the exact commands.

## 11.10 Summary and what to carry forward

- The **roofline** $P\le\min(P_{\max}, B_{\max} I)$ with ridge $I_{\text{ridge}} =
  P_{\max}/B_{\max}$ classifies every kernel. For the GB10, $I_{\text{ridge}}\approx
  923$ FLOP/byte at (measured) FP4.
- **Attention** ($I\approx63$) and **small-batch/low-res GEMMs** ($I\approx680$) are
  **memory-bound**; **big GEMMs at 1024px** ($I\approx1845$) are **compute-bound**.
  The workload is a *mix*.
- **Low precision is the master key**: it cuts bytes (helps memory-bound) *and*
  raises the FLOP ceiling (helps compute-bound). Cost = accuracy (Chapter 12).
- **Unified memory** removes transfer traffic entirely; **shared memory** and
  **occupancy** are the on-chip levers (Part IV).

With the roofline as our scoreboard, Chapter 12 introduces the numerical formats —
BF16, FP8, NVFP4 — that let us trade precision for position on it, and measures the
image-quality cost of each.

---

### Exercises

1. **Ridge points.** Recompute $I_{\text{ridge}}$ for FP4 and FP8 using the
   *theoretical* peaks (~500 / ~250 TFLOP/s). How far right does the ridge move,
   and which workloads cross from memory- to compute-bound?
2. **Resolution sweep.** Recompute the big-GEMM intensity $I$ for $M\in\{256, 1024,
   4608\}$ (256/512/1024px). At which resolution do the projections cross the FP4
   ridge?
3. **Attention is stuck.** Given the measured 279 GB/s ≈ peak, explain why no amount
   of tensor-core cleverness speeds attention up, and what the *only* remaining
   levers are.
4. **Quantization on the roofline.** Draw (or describe) how switching a memory-bound
   GEMM from BF16 to NVFP4 weights moves its point on the roofline, and do the same
   for a compute-bound GEMM. Why does precision help both?

*Next: [Chapter 12 — Numerical formats & quantization](12-quantization.md).*
