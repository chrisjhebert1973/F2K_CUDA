# Chapter 12 — Numerical Formats & Quantization

> *Goal of this chapter:* understand the low-precision formats that let us trade
> accuracy for position on the roofline — **BF16, FP8 (E4M3/MXFP8), and NVFP4
> (E2M1 + block scales)** — at the bit level, including *why* block scaling is
> necessary, the exact quantization procedure this project uses, and the
> **measured** image-quality cost of each. We end knowing why FP4 is fast-but-soft,
> why FP8 buys back almost all the quality, and which tensors are quantized at all.
> Anchored to `tools/f2k_convert.cu`, `tools/quant_sim.py`, and `linear.cu`.
>
> *Prerequisites:* Chapter 11 (the roofline); IEEE floating point basics.

---

## 12.1 Floating point in one screen

A floating-point number is $(-1)^s \cdot (1.m) \cdot 2^{e-\text{bias}}$: a sign $s$,
a mantissa $m$ (the fractional bits, giving **precision**), and an exponent $e$
(giving **dynamic range**). For a fixed bit budget you trade range against
precision by moving bits between $e$ and $m$. The notation **E$x$M$y$** means $x$
exponent bits, $y$ mantissa bits (plus 1 sign bit).

| format | bits | layout | mantissa bits | relative step (≈ $2^{-m}$) | role |
|---|---|---|---|---|---|
| FP32 | 32 | E8M23 | 23 | $10^{-7}$ | references, scales |
| **BF16** | 16 | E8M7 | 7 | $\approx0.8\%$ | activations, "lossless" baseline |
| FP16 | 16 | E5M10 | 10 | $0.1\%$ | (not used here) |
| **FP8 E4M3** | 8 | E4M3 | 3 | $\approx6\text{–}12\%$ | MXFP8 weights/acts |
| FP8 E5M2 | 8 | E5M2 | 2 | $\approx25\%$ | (more range, less precision) |
| **FP4 E2M1** | 4 | E2M1 | 1 | $\approx50\%$ | NVFP4 weights/acts |

The key intuition: **mantissa bits set the relative quantization step.** BF16's 7
mantissa bits give ~0.8% steps — visually lossless, our baseline. Drop to FP8's 3
bits and steps grow to ~6–12%. Drop to FP4's *single* mantissa bit and the step is
~50% — enormous. The whole game of low-precision inference is making that coarse
grid survive by choosing the right *scale* for small groups of values.

> Note BF16 keeps FP32's 8 exponent bits (same range as FP32, just coarser
> mantissa) — which is why it is the friendly default for activations: it almost
> never overflows, it just rounds. FP16 trades range for precision and is more prone
> to overflow, so the ML world standardized on BF16.

## 12.2 The FP4 grid, concretely

E2M1 has 2 exponent bits and 1 mantissa bit. Enumerate it: the representable
*magnitudes* are

$$
\{0,\ 0.5,\ 1,\ 1.5,\ 2,\ 3,\ 4,\ 6\}
$$

(times a scale), plus their negatives — **16 values total** in 4 bits. Look at the
gaps: from 4 to 6 is a 50% jump; there is nothing between 4 and 6, nothing between
2 and 3. A weight of 5.0 must snap to 4 or 6 — a ~20% error on that element, and
worst-case (a value at 5) the step is ~50% of the spacing. This coarseness is
**fundamental to 1 mantissa bit**, not a bug. It is why pure FP4 images are soft
(§12.6). E4M3, with 3 mantissa bits, has ~8× finer spacing within each binade
($1.000, 1.125, 1.250, \dots$), hence the ~6–12% steps.

## 12.3 Why a single scale is not enough: block scaling

A 4-bit code can only represent 16 distinct magnitudes. To cover a real weight
tensor — whose values might span $[-22528, 22528]$ (the Qwen3 outliers of Chapter
05!) — you multiply the grid by a **scale**: store $\hat w = \text{round}(w/S)$ in
FP4 and recover $w \approx S\hat w$. But one scale for an entire matrix is a
disaster: a single large outlier forces $S$ huge, and then every *normal* value
($\sim$1) rounds to the nearest multiple of $6S$ — i.e. to **zero**. The outlier
survives; everything else is annihilated.

The fix is **block (a.k.a. microblock) scaling**: partition each row into small
contiguous blocks and give *each block its own scale*. An outlier now only inflates
the scale of its own little block; the other blocks keep fine scales and full
precision. This is the central idea of NVFP4 and MXFP8.

```
 row of weights:  [ w0 w1 ... w15 | w16 ... w31 | ... ]
                    └── block 0 ──┘ └─ block 1 ─┘
 each block: values in FP4 (or FP8) + ONE small scale (E4M3 or UE8M0)
 plus one FP32 per-tensor scale on top (NVFP4)
```

The cost is the scales' own bytes — but a per-16 E4M3 scale adds only $8/16 = 0.5$
bits per weight, so NVFP4 is effectively ~4.25 bits/weight, still ~3.5× smaller
than BF16. The benefit is that block scaling makes 4-bit *usable* on real,
outlier-laden weights.

## 12.4 The two formats this project uses

### NVFP4 (Blackwell-native 4-bit)

- **Values:** E2M1 (the §12.2 grid), packed two-per-byte (`[N, K/2]`).
- **Block scale:** one **E4M3** scale per **16** consecutive elements
  (`[N, K/16]`).
- **Tensor scale:** one **FP32** scalar for the whole tensor.
- **Effective:** ~4.25 bits/weight; ~3.5× smaller than BF16.
- Dequant: $w \approx (\text{FP32 tensor scale}) \cdot (\text{E4M3 block scale}) \cdot (\text{E2M1 value})$.

### MXFP8 (the "microscaling" 8-bit)

- **Values:** E4M3, one byte each (`[N, K]`).
- **Block scale:** one **UE8M0** scale per **32** elements (`[N, K/32]`). UE8M0 is
  an *8-bit exponent-only* scale — a power of two, so dequant is a bit-shift, no
  mantissa.
- **Effective:** ~8.25 bits/weight; ~2× smaller than BF16, ~2× larger than NVFP4.
- Dequant: $w \approx 2^{\text{UE8M0}} \cdot (\text{E4M3 value})$.

Both are *block-scaled tensor-core formats* — Blackwell's MMA units consume the
packed values and block scales directly (Chapter 13). The F2K container stores them
in exactly this layout so the runtime mmaps and feeds them with no parsing
(Chapter 14).

## 12.5 The quantization procedure

This project quantizes weights offline in `tools/f2k_convert.cu` (and
`tools/quant_sim.py` mirrors it in PyTorch for validation). The procedure per block:

**NVFP4** (block of 16): the E2M1 grid maxes at 6, so set the block scale so the
block's largest-magnitude value lands at 6:

$$
S_{\text{block}} = \text{quantize\_to\_E4M3}\!\left(\frac{\max_i |w_i|}{6}\right),\qquad
\hat w_i = \text{round\_to\_E2M1}\!\left(\frac{w_i}{S_{\text{block}}}\right).
$$

**MXFP8** (block of 32): E4M3 maxes at 448, and the scale is a pure power of two:

$$
X = 2^{\big\lceil \log_2(\max_i|w_i| / 448)\big\rceil},\qquad
\hat w_i = \text{round\_to\_E4M3}\!\left(\frac{w_i}{X}\right).
$$

The `⌈log2⌉` makes the scale a UE8M0 power of two (a shift). Both schemes are
"absmax" calibrations — scale so the block's extreme value is representable, accept
the rounding of the rest. (More elaborate calibrations — percentile clipping,
GPTQ-style error feedback — are possible and were on the design table; the shipped
converter uses absmax, which is already within FP4's noise floor for these
weights.)

### What gets quantized

Not every tensor is eligible. `f2k_convert` quantizes a tensor only if it is a
rank-2 `.weight` with $N\%128=0$ and $K\%64=0$ — the tensor-core tile constraints
(Chapter 13). Everything else — biases, norm gains, embeddings, odd-shaped tensors
— stays **BF16** passthrough. For Qwen3, `model.embed_tokens.weight` is explicitly
kept BF16 (a `--keep-bf16` flag) and `lm_head` is dropped (unused as an encoder).
So "the model is FP4" really means *its big matmul weights* are FP4; a long tail of
small tensors stays BF16, which costs little and avoids quantizing things that
don't tolerate it.

## 12.6 The measured quality cost

This is the heart of the chapter, and it is *measured*, not asserted
(`tools/quant_sim.py` fake-quantizes the diffusers Flux2 transformer, weight-only,
full pipeline, cat @ seed 42 / 4 steps):

| format | weight rel-L2 error | image mean&#124;Δ&#124; vs BF16 |
|---|---|---|
| **NVFP4** (E2M1 + per-16 E4M3) | **0.101** | **12.95%** |
| **FP8** (E4M3, per-row) | **0.027** | **2.12%** |

FP8 is ~4× better per weight and ~6× better per image. End to end, the project's
own CUDA pipeline (which also quantizes *activations*, not just weights) measures
the transformer's single-forward cosine similarity against a diffusers BF16
reference (`tools/cmp_transformer.cu`, Chapter 25):

$$
\text{NVFP4: } \cos = 0.963,\ \text{rel-L2} = 0.27 \qquad\longrightarrow\qquad
\text{MXFP8: } \cos = 0.99895,\ \text{rel-L2} = 0.046 .
$$

Visually: the NVFP4 cat is soft/washed; the **MXFP8 cat is sharp**, essentially
indistinguishable from BF16 (fur texture, facial features, lane markings all
recovered). The FP4 "fuzziness" is **fundamental to 1 mantissa bit**, confirmed by
the fake-quant study — it is not a kernel bug. (An earlier, long investigation
chased the softness as a possible bug before this study settled it; Chapter 25.)

> **Resolution interacts with this.** The FP4 softness is most visible at low
> resolution; at 1024px the extra spatial detail masks much of it, so NVFP4 1024px
> images look quite good. But FP8 is the right default when quality matters, at the
> cost of ~2× the weight bytes (~5 GB → ~9 GB, fine in 128 GB) and ~2× the per-step
> GEMM time (still bandwidth-fine for the memory-bound parts; Chapter 11).

## 12.7 Tying back to the roofline

Recall Chapter 11: low precision helps memory-bound kernels (fewer bytes) *and*
compute-bound kernels (faster instruction). Now we can be precise about the
trade-curve:

- **NVFP4:** ~4.25 bits/weight, ~2× the FP8 FLOP ceiling. Best for *speed* —
  smallest model (~5 GB transformer), highest tensor-core rate (measured 252
  TFLOP/s). Quality cost ~13% image error.
- **MXFP8:** ~8.25 bits/weight, ~161 TFLOP/s. ~2× slower and ~2× larger than NVFP4,
  but ~6× better image fidelity (cos 0.999). Best for *quality*.

`generate --precision {nvfp4,fp8}` lets you pick the point on this curve per run;
the architecture and kernels are identical, only the `Linear`'s format differs
(Chapter 13). The on-disk model exists in both NVFP4 (`transformer_f2k`) and MXFP8
(`transformer_mxfp8`) forms, produced by `f2k_convert --quant {nvfp4,mxfp8}`.

## 12.8 Where this lives in the code

| Concept | Code |
|---|---|
| offline quantization (absmax, both formats) | `tools/f2k_convert.cu` (`quantize_mxfp8`, NVFP4 packing) |
| eligibility ($N\%128, K\%64$, `.weight` rank-2) | `tools/f2k_convert.cu` classifier |
| fake-quant quality study | `tools/quant_sim.py` |
| on-device activation quant | `linear.cu` (`quant_for_A_kernel`, `quant_for_A_fp8_kernel`) |
| precision selection | `Linear::Precision`, `generate --precision` |
| dtype codes (F8_E4M3=3, NVFP4=5, …) | `f2k_format.h` (Chapter 14) |

## 12.9 Summary and what to carry forward

- **Mantissa bits = relative step.** BF16 ≈0.8% (baseline), FP8 E4M3 ≈6–12%, FP4
  E2M1 ≈50% — the FP4 grid is the coarse $\{0,.5,1,1.5,2,3,4,6\}$.
- **Block scaling** is what makes 4/8-bit usable: per-16 (NVFP4, E4M3 scale) or
  per-32 (MXFP8, UE8M0 scale) scales contain outliers so normal values keep
  precision. NVFP4 ≈4.25 b/w, MXFP8 ≈8.25 b/w.
- Quantization is **offline, absmax** (`f2k_convert`); only big rank-2 `.weight`
  matmul tensors are eligible, the rest stay BF16.
- **Measured cost:** NVFP4 ≈13% image error / cos 0.963; **MXFP8 ≈2% / cos 0.999**.
  FP4 softness is fundamental to 1 mantissa bit. Pick per run with `--precision`.
- Both formats help on the roofline (fewer bytes + higher FLOP ceiling); the
  curve is **speed (NVFP4) ↔ quality (MXFP8)**.

Chapter 13 shows how Blackwell's tensor cores actually *consume* these block-scaled
formats — the CUTLASS GEMM and the `Linear` class built on it.

---

### Exercises

1. **Enumerate E2M1.** Derive the 8 magnitudes $\{0,.5,1,1.5,2,3,4,6\}$ from 2
   exponent + 1 mantissa bits, and compute the worst-case relative rounding error.
2. **Outlier annihilation.** With one per-tensor scale and a single value at 22528
   among values near 1, show that the near-1 values round to 0 in FP4. Then show a
   per-16 block scale rescues them.
3. **Byte budget.** Compute bits/weight for NVFP4 (E2M1 + per-16 E4M3 + per-tensor
   FP32) and MXFP8 (E4M3 + per-32 UE8M0). Confirm ~4.25 and ~8.25.
4. **Pick a format.** For (a) a quick 256px draft and (b) a final 1024px portrait,
   which precision and why? Tie your answer to §12.6's measured numbers and Chapter
   11's roofline.

*Next: [Chapter 13 — Tensor-core GEMM with CUTLASS](13-cutlass-gemm.md).*
