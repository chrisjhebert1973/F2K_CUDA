# Chapter 19 — Convolutions with the cuDNN 9 Graph API

> *Goal of this chapter:* the VAE decoder is convolutions, and getting them onto
> Blackwell's tensor cores was a 14× story with a sharp lesson. We cover what a
> conv is (and why it is "just" a GEMM), why the *legacy* cuDNN API silently misses
> the fast Blackwell engines, the two compounding causes (a stale library link and
> the legacy API itself), and the cuDNN 9 **graph API + NHWC** rewrite that took VAE
> decode from **18.7 s to 1.35 s** at 1024px. Anchored to `conv2d.{h,cu}`.
>
> *Prerequisites:* Chapters 11 (roofline), 13 (GEMM, tensor cores), 03 (the VAE).

---

## 19.1 Convolution is a GEMM in disguise

A 2D convolution slides a filter bank $W\in\mathbb R^{C_\text{out}\times C_\text{in}
\times R\times S}$ over an input $x\in\mathbb R^{C_\text{in}\times H\times W}$,
producing $y\in\mathbb R^{C_\text{out}\times H'\times W'}$ where each output pixel is
a dot product of the filter with a local input patch. The classic way to see it as
a matmul is **im2col**: gather every $C_\text{in}\times R\times S$ input patch into a
column, stack the columns into a matrix $\tilde X$ of shape
$[(C_\text{in}RS)\times(H'W')]$, reshape $W$ to $[C_\text{out}\times(C_\text{in}RS)]$,
and then $y = W\,\tilde X$ is an ordinary GEMM. Modern libraries use *implicit GEMM*
(they never materialize $\tilde X$; they index it on the fly), but the point stands:
**a conv is a GEMM**, and on Blackwell it wants the same tensor cores as Chapter 13.

The FLUX VAE uses only $3\times3$ and $1\times1$ convolutions, single-stride,
dilation 1, no grouping (`conv2d.h`). $1\times1$ convs are literally per-pixel
matmuls; $3\times3$ are the im2col GEMM above with a 9-patch gather.

## 19.2 The symptom: convolutions 170× too slow

When the VAE decoder was first built on cuDNN, a single $128$-channel $1024^2$
$3\times3$ convolution took **~510 ms**. For reference, PyTorch does the same conv
in ~3 ms, and the eventual fixed version here does it in ~29 ms. At ~510 ms/conv the
whole VAE decode was **18.7 s** at 1024px — larger than the entire denoise loop.
Something was dispatching to a slow, non-tensor-core kernel.

## 19.3 The diagnosis: two compounding causes

**Cause 1 — a stale cuDNN link.** CMake linked the *unversioned* dev symlink
`libcudnn.so`, which on this box pointed at cuDNN **8.9** (the apt package). cuDNN
8.9 **predates Blackwell** — it has no sm_121 convolution engines at all, so it falls
back to a generic kernel. The system *also* had cuDNN **9.23** installed
(`libcudnn.so.9` plus `engines_{precompiled,runtime_compiled,tensor_ir}.so.9`), and
the headers in `/usr/include` already reported `CUDNN_MAJOR 9` — so the code compiled
against 9's headers but *linked* 8.9's runtime, a silent mismatch. **Fix:** link
`/usr/lib/aarch64-linux-gnu/libcudnn.so.9` explicitly in CMake.

**Cause 2 — the legacy API.** Linking cuDNN 9 alone did *not* fix it: the legacy
`cudnnConvolutionForward` path stayed ~510 ms even on 9. On Blackwell, the fast
tensor-core conv engines are reachable **only** through the cuDNN 9 **backend graph
API**, and **only with NHWC** layout. The legacy API and NCHW both route to slow
fallback kernels. So the fix was not just "newer cuDNN" but "newer cuDNN, *graph
API*, *NHWC*."

> **The lesson.** A 170× performance bug had *two* causes, neither of which threw an
> error: a build-system link pointing at the wrong runtime, and an API that "works"
> but never touches the fast path. Both are invisible to correctness tests (the math
> was right the whole time) — only a *performance* measurement against a known-good
> reference (PyTorch's 3 ms) revealed the gap. Always benchmark a new dependency path
> against a reference implementation, not just check that it produces correct output.

## 19.4 The cuDNN 9 graph API

The modern cuDNN interface is a **dataflow graph**: you describe the operation as a
graph of tensor nodes and op nodes, hand it to a **heuristic** that proposes
**engine configurations** (concrete kernels + tile/stage choices), pick one, and
**execute** it with a workspace. It is more verbose than the one-call legacy API,
but it is the only way to reach the autotuned, architecture-specific engines.

The rewritten `conv2d.cu`:

1. **Builds an op-graph** with **NHWC** tensors and a **KRSC** filter (the layouts
   the Blackwell engines want).
2. Asks the heuristic in **`ENGINEHEUR` mode A** for engine configurations. The
   heuristic returns them **sorted by predicted performance**, so **`config[0]` is
   the Blackwell tensor-core engine**. Benchmarking *all* configs picked the same
   engine but cost ~13 s of construction; taking `config[0]` directly cuts
   construction to **~0.47 s** with the identical kernel.
3. **Executes** the plan with a device workspace.

```
 build graph (NHWC in/out, KRSC filter, conv node)
   → heuristic (ENGINEHEUR mode A) → configs sorted by predicted perf
     → take config[0]  (the tensor-core engine; no full autotune sweep)
       → build execution plan + workspace
         → execute(x_nhwc, W_krsc, y_nhwc)
```

## 19.5 Keeping an NCHW interface

The rest of the VAE (resnet blocks, group norm, upsample — Chapter 20) and the
patchify/transpose conventions are all **NCHW**. Rather than ripple NHWC through
everything, `Conv2d` keeps an **NCHW interface** and transposes internally:

```
 x [N,C,H,W]  --nchw_to_nhwc-->  x_nhwc  --cuDNN graph-->  y_nhwc  --nhwc_to_nchw-->  y [N,C',H',W']  (+ bias)
```

The weight is reordered **KCRS → KRSC once at construction**. The transposes use the
existing `transpose_chw` kernel (Chapter 15); they cost some bandwidth but are far
cheaper than the kernel-selection win. The workspace is `x_nhwc + y_nhwc +
plan_workspace` — ~5.1 GiB at $1024^2$, which unified memory absorbs (Chapter 11).
Callers and the other VAE kernels are unchanged.

## 19.6 The payoff

| measurement | before (legacy/8.9) | after (graph API/9/NHWC) | speedup |
|---|---|---|---|
| one 128-ch $1024^2$ $3\times3$ conv | ~510 ms | ~29 ms | ~18× |
| **VAE decode @ 1024px** | **18.7 s** | **1.35 s** | **13.8×** |
| VAE forward @ 128px | 314 ms | **10 ms** | **31×** |
| total 1024px image | ~133 s | ~30 s | (this + attention + denoise) |
| construction (engine pick) | (n/a) | 0.47 s (config[0]) vs 13 s (full sweep) | 28× |

Correctness held throughout: `tests/test_conv2d.cu` shows cos 1.00000 vs an FP32
reference, and the 1024px image is bit-identical to the legacy path up to BF16
rounding (mean Δ 0.055/255 — a *different conv kernel*, same math). After this fix
the VAE decode (1.35 s) is no longer a bottleneck; the denoise loop (Part IV) is the
dominant cost again.

## 19.7 Why the VAE conv was memory-bound-ish too

On the roofline (Chapter 11), these convs are large (high channel counts, big
spatial extent) but not as compute-dense as the transformer's huge GEMMs, and the
NHWC transposes add traffic — so the win is partly *reaching the tensor cores at all*
(Cause 2) and partly *moving the right bytes in the right layout*. The deeper point:
even with perfect math, you can sit at a tiny fraction of peak if you are on the
wrong kernel/layout/library — performance is a property of the *whole dispatch
path*, not just the algorithm.

## 19.8 Where this lives in the code

| Concept | Code |
|---|---|
| conv wrapper (graph API, NHWC, NCHW interface) | `conv2d.{h,cu}` |
| NCHW↔NHWC transpose | `kernels/transpose_chw.*` |
| cuDNN 9 link | CMake (`libcudnn.so.9`, not the `.so` symlink) |
| correctness | `tests/test_conv2d.cu` |
| users | `vae_resnet.*`, `vae_decoder.*` (Chapter 20) |

## 19.9 Summary and what to carry forward

- A convolution **is a GEMM** (im2col / implicit GEMM); on Blackwell it wants the
  same tensor cores as `Linear`.
- The VAE convs were **170× too slow** from **two silent causes**: CMake linked
  **cuDNN 8.9** (pre-Blackwell) via the unversioned symlink, *and* the **legacy API**
  never reaches the fast engines even on cuDNN 9.
- The fix is the cuDNN 9 **backend graph API + NHWC + KRSC**, picking the heuristic's
  **`config[0]`** (tensor-core engine) without a full autotune sweep. `Conv2d` keeps
  an NCHW interface and transposes internally.
- Result: **VAE decode 18.7 s → 1.35 s (13.8×)**, 128px forward 31×, bit-identical
  output. *Benchmark new dependency paths against a reference — correctness tests
  won't reveal a wrong-kernel dispatch.*

Chapter 20 assembles these convs into the full decoder — resnet blocks, group norm,
upsampling, the mid-block spatial attention, and the `post_quant_conv`/de-norm
boundary from Chapter 03.

---

### Exercises

1. **Conv as GEMM.** For a $3\times3$, $C_\text{in}=512\to C_\text{out}=512$ conv at
   $128\times128$, write the im2col matrix shapes and the GEMM dimensions $M,N,K$.
   Is it above or below the FP-relevant ridge (Chapter 11)?
2. **Two silent bugs.** Explain why neither the stale cuDNN link nor the legacy API
   produced *wrong* output, and why only a performance comparison to PyTorch
   revealed them. What test would catch this in CI?
3. **config[0] vs sweep.** Why does taking the heuristic's first config give the same
   kernel as a full benchmark sweep here, and when might it *not*? What did the sweep
   cost?
4. **Layout tax.** The NCHW interface transposes to NHWC and back per conv. Estimate
   that traffic at $512$ch $128^2$ BF16 and argue why it is worth paying for the
   tensor-core engine.

*Next: [Chapter 20 — Assembling the VAE decoder](20-vae-decoder.md).*
