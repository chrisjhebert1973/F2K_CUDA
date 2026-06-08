# Chapter 24 — Performance Methodology

> *Goal of this chapter:* generalize the optimization lessons scattered through the
> course into a reusable method. We collect the project's actual wins, extract the
> recurring discipline (measure the ceiling before pulling a lever), name the
> instruments, and state the principles that let a 9B image model go from ~133 s to
> ~8 s per image on the GB10.
>
> *Prerequisites:* Chapters 11 (roofline), 17 (the attention journey), 19 (the cuDNN
> story).

---

## 24.1 The wins, collected

Four large optimizations, each from a different cause:

| optimization | result | the actual lever | chapter |
|---|---|---|---|
| attention kernel | 56.5 → **19.7 ms** (2.87×) | occupancy diagnosis → cut smem → mma+ldmatrix | 17 |
| VAE convolutions | 18.7 → **1.35 s** (13.8×) | right cuDNN version + graph API + NHWC | 19 |
| model build | 76 → **5.3 s** (14×) | quantize once, store on disk (on-disk MXFP8) | 14 |
| text encode | 98 → **0.28 s** (~350×) | native encoder+tokenizer vs PyTorch load | 21, 22 |

Plus the architectural/quantization wins baked in from the start: NVFP4/MXFP8
weights (Chapter 12), fused single-stream GEMMs (Chapter 9), unified-memory mmap
(Chapter 14). Together: **~133 s → ~30 s → ~8 s/image** (with builds amortized).

Notice the levers are all *different*: an occupancy limit, a dependency
misconfiguration, a redundant recomputation, and a wrong runtime entirely. There is
no single trick — each required *diagnosing the specific cause*.

**Part VIII costs, for reference.** Image conditioning (Chapter 27) adds a VAE
encode: ~0.1 s at 512px, dominated by the same convs as decode. The live-preview
stream (Chapter 28) decodes ~4 low-res JPEGs per run at ~40 KB each — a small,
bounded tax for watching the image form. The one *expensive* mode is **2048px (true
2K)**: ~80–90 s, because the denoise is $O(S^2)$ at sequence length ~16 896 (≈14.5 s
per step) plus ~15 s each for VAE encode and decode. That is the honest state — 2K
is *correct but unoptimized*; it is the clearest remaining target, and the same
roofline discipline (the attention is again the bandwidth-bound bottleneck) applies.

## 24.2 The core discipline: measure the ceiling first

The throughline, stated once: **before optimizing, determine which ceiling a kernel
is under** — compute, memory, occupancy, or "wrong kernel/path entirely" — because
the lever that helps one is useless (or harmful) for another.

The canonical example (Chapter 17): the WMMA attention at 97.7 GB/s looked like it
needed "more K/V reuse / bigger tiles." But 97.7 GB/s is only 36% of peak *and* the
TFLOP/s was low — so it was neither memory- nor compute-bound; it was
**occupancy-bound** (1 CTA/SM from 93 KiB smem). The correct lever was the
*opposite* of the intuitive one: **less** shared memory, to fit more CTAs. Chasing
"bigger tiles" would have been wasted effort against a bottleneck that wasn't there.

The dual example (Chapter 19): the VAE convs at ~510 ms weren't an *algorithm*
problem at all — the math was correct — they were dispatching to a slow kernel
because of a stale library link and the legacy API. No amount of kernel tuning would
have helped; the fix was the *dispatch path*. Only a comparison to a reference
(PyTorch's 3 ms) revealed that the ceiling was "you're not on the fast path."

> **Principle.** Optimization without a confirmed bottleneck is guessing. The most
> expensive mistakes are *plausible* levers aimed at the wrong ceiling.

## 24.3 The instruments

The project used a small, blunt toolkit — no exotic profilers required:

- **The roofline arithmetic** (Chapter 11). Compute $I=\text{FLOP}/\text{byte}$ for a
  kernel and compare to $I_{\text{ridge}}=P_{\max}/B_{\max}$. This *predicts*
  memory- vs compute-bound before you run anything, and tells you which lever class
  applies.
- **Microbenchmarks with a bandwidth model.** `tools/bench_attention.cu` prints, per
  shape: latency, achieved **GB/s** (against the 273 peak), **TFLOP/s**, the K/V
  re-read HBM model, and — via probes — the achieved **CTAs/SM**. Those four numbers
  diagnosed the occupancy limit directly.
- **`cudaOccupancyMaxActiveBlocksPerMultiprocessor`** + smem arithmetic. Confirms
  CTAs/SM analytically (93 KiB → 1, 43 KiB → 2) so you know the occupancy lever's
  payoff *before* writing the kernel.
- **A reference implementation for the path.** PyTorch's conv time (3 ms) was the
  tell that 510 ms meant "wrong kernel," not "hard problem." Always have a known-good
  number for what *should* be achievable.
- **`nvtop`/the GPU utilization.** The 98 s text-encode showed the **GPU idle** —
  immediately reframing it as an I/O/load problem (PyTorch streaming 15 GB), not a
  compute one. A glance at utilization is a free first diagnostic.

## 24.4 The optimization sequence (a template)

From the attention journey (Chapter 17), generalized to any hot kernel on a
bandwidth-bound machine:

1. **Eliminate redundant HBM traffic first.** Algorithm beats micro-optimization:
   online softmax + tiling (don't re-read K/V) dwarfed any tuning of the wasteful
   row kernel. *Are you moving bytes you don't need to?*
2. **Use the right compute units.** Tensor cores for matmuls (Chapter 13), the fast
   cuDNN engines for convs (Chapter 19). *Are you even on the intended hardware path?*
3. **Measure which ceiling you're under** (§24.2). GB/s vs peak, TFLOP/s vs peak,
   CTAs/SM. *Don't proceed on an unconfirmed bottleneck.*
4. **Pull the matching lever.** Occupancy → cut smem; memory-bound → cut bytes
   (quantize, fuse); compute-bound → faster instruction (lower precision); wrong path
   → fix the dispatch.
5. **Validate every step** (Chapter 25). A faster wrong kernel is worthless; check
   against an independent oracle, bit-exact where possible, and distinguish
   reordered-FP drift (e.g. 1.8% image diff from summation order) from bugs.
6. **Stop at the ceiling.** The attention kernel hit 279 GB/s ≈ peak — there is
   nothing left to win because the bytes can't arrive faster. Declare victory at the
   hardware limit, not when the code stops being abstractly improvable.

## 24.5 Know when *not* to optimize

Equally important: the project explicitly *declined* several optimizations because
the diagnosis said they wouldn't pay (Chapter 17):

- **Bigger query tiles / persistent CTAs** — target a bandwidth/launch bottleneck the
  occupancy-bound kernel didn't have.
- **`ldmatrix.x4`, K/V double-buffering** — micro-opts on a kernel already at peak HBM
  bandwidth; can't meaningfully help a memory-bound kernel at the ceiling.
- **A bespoke $D=512$ VAE-attention kernel** — the VAE decode is dominated by convs,
  not attention, so the flash kernel was good enough; effort went where the time was.

The discipline is to spend optimization effort *proportional to a stage's share of
the wall clock, and only on a confirmed bottleneck.* The denoise loop got the deepest
work because it is the per-image cost; the builds got the on-disk-quant win because
they were 76 s; the text encode got a rewrite because it was 98 s of idle GPU.

## 24.6 Bandwidth as the design compass

Underneath every win is Chapter 11's thesis. On a machine with ~920 FLOP/byte of
headroom, the recurring questions are: *how many bytes does this move, and can it
move fewer?* That single compass explains quantization (fewer weight bytes), fused
GEMMs (read the activation once), flash attention (don't re-read K/V), unified-memory
mmap (no transfer), and the on-disk-quant build win (don't recompute). Even the
compute-bound big GEMMs are helped by lower precision, which raises the *FLOP ceiling*
on the same roofline. When unsure what to do, the answer is usually "move fewer bytes,
or move them on a faster instruction."

## 24.7 Summary and what to carry forward

- Four big wins, **four different levers**: occupancy (attention), dispatch path
  (cuDNN convs), redundant recompute (on-disk quant build), wrong runtime (native
  encoder/tokenizer). **No single trick.**
- **Measure the ceiling before optimizing** — compute/memory/occupancy/wrong-path —
  because the matching lever differs and the intuitive one is often wrong (attention's
  "bigger tiles," the conv's "tune the kernel").
- The toolkit is blunt and sufficient: **roofline arithmetic, a microbenchmark with a
  bandwidth model, `cudaOccupancyMaxActiveBlocks`, a reference number, and a glance at
  GPU utilization.**
- Sequence: eliminate redundant traffic → right compute units → **measure** → pull the
  matching lever → validate → **stop at the ceiling**. And spend effort proportional
  to wall-clock share, declining opts the diagnosis says won't pay.
- The compass is **bandwidth**: move fewer bytes, or on a faster instruction.

Chapter 25 covers the other half of trustworthy engineering — how correctness was
established and how the hardest bugs were found.

---

### Exercises

1. **Diagnose three.** For (a) a kernel at 36% of peak BW with low TFLOP/s, (b) a
   kernel at 99% of peak BW, (c) a correct kernel 170× slower than a reference — name
   the ceiling and the lever for each.
2. **Predict before measuring.** Using the roofline, predict whether a 256px
   (M=256) vs 1024px (M=4608) `Linear` is memory- or compute-bound (Chapter 11), and
   say which optimization helps each.
3. **The idle-GPU tell.** Explain why `nvtop` showing an idle GPU during the 98 s
   text-encode immediately rules out "the encoder kernel is slow" and points at I/O.
4. **When to stop.** Argue why hitting 279 GB/s on the attention kernel means further
   kernel work is wasted, and what would have to change about the *problem* to make
   more speed possible.

*Next: [Chapter 25 — Validation methodology](25-validation.md).*
