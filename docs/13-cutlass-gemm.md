# Chapter 13 — Tensor-Core GEMM with CUTLASS

> *Goal of this chapter:* implement the matrix multiply that every `Linear` in the
> model reduces to, using Blackwell's **block-scaled tensor cores** via CUTLASS. We
> cover what tensor-core MMA is, how Blackwell consumes the NVFP4/MXFP8 formats of
> Chapter 12 *natively*, the CUTLASS type configuration we mirror from example 79a,
> the scale-factor layout that caused a subtle accuracy bug (and the CuTe fix), and
> the `Linear` class that wraps it — construction-time weight repacking, on-device
> activation quantization, and the alignment constraints. Anchored to
> `fp4_gemm.{h,cu}`, `fp8_gemm.{h,cu}`, `linear.{h,cu}`.
>
> *Prerequisites:* Chapters 11–12; the GEMM $Y=XW^\top$; CUDA tiling at a high
> level (the PTX detail is Chapter 18).

---

## 13.1 Everything is a GEMM

Count the matmuls in one forward pass: each double-stream block has 12 linears,
each single-stream block 2 (large, fused), plus embedders, modulation, final
projection — 124 `Linear`s (Chapter 10), each a GEMM $Y = X W^\top$. The text
encoder adds dozens more. **The GEMM is the workhorse**, and its throughput
(Chapter 11: 252 TFLOP/s FP4, 161 FP8) sets the per-step cost. This chapter is how
that throughput is reached.

A `Linear` computes, for $M$ token rows, $K$ input features, $N$ output features:

$$
Y_{[M,N]} = X_{[M,K]}\, W^\top_{[K,N]} \;(+\,\text{bias}),
$$

with $X$ the activations (BF16) and $W$ the quantized weights (NVFP4 or MXFP8).

## 13.2 Tensor cores and MMA, briefly

A **tensor core** executes a small matrix multiply-accumulate in one instruction —
`D = A·B + C` for fixed tile shapes (e.g. $16\times8\times16$), with the inputs in
a low-precision format and the accumulation in FP32. A GEMM kernel tiles the big
matrices into these MMA-sized fragments, streams them through shared memory and
registers, and accumulates. The arithmetic density is enormous: one instruction,
hundreds of MACs. (The exact register/fragment layouts and the raw `mma.sync` PTX
are Chapter 18, where we hand-write them for attention. Here CUTLASS generates them
for us.)

Blackwell's 5th-gen tensor cores add **block-scaled** MMA: the instruction takes
not just the packed low-precision operands but also their **per-block scale
factors**, and applies the scales *inside* the MMA. This is the hardware
counterpart of Chapter 12's block scaling — NVFP4 (E2M1 values + per-16 E4M3
scales) and MXFP8 (E4M3 values + per-32 UE8M0 scales) are *native operand formats*,
not something we dequantize first. That is the whole reason these formats are fast:
no separate dequant pass, the tensor core reads packed values + scales and produces
FP32.

## 13.3 Why CUTLASS

Writing a peak-throughput Blackwell GEMM by hand means orchestrating TMA loads,
multi-stage shared-memory pipelines, the block-scaled MMA atoms, and the epilogue —
months of work to match the vendor library. **CUTLASS** (NVIDIA's open template
library) already encodes these for every architecture, including the block-scaled
Blackwell path. This project vendors CUTLASS and adapts its example
`79a_blackwell_geforce_nvfp4_bf16_gemm` — the canonical GeForce/consumer Blackwell
NVFP4→BF16 GEMM. (The decision to use the templated CUTLASS path first, and keep
hand-rolling `tcgen05` MMA as a later option, was deliberate; the hand-rolled path
is exactly what Chapter 18 does for *attention*, where the access pattern is special
enough to beat the library.)

The header `fp4_gemm.h` keeps the interface **CUTLASS-free** — all the template
machinery lives in `fp4_gemm.cu` — so callers (blocks, the UI, tests) don't drag
CUTLASS's heavy includes into every translation unit. `FP4Gemm` is a persistent
handle: configure once with the shape, call `run()` each step.

## 13.4 The type configuration

The CUTLASS kernel is selected by a stack of template types. The ones that matter,
mirrored from example 79a:

```cpp
ArchTag        = cutlass::arch::Sm120;                       // covers sm_120 AND sm_121 (GB10)
OperatorClass  = cutlass::arch::OpClassBlockScaledTensorOp;  // block-scaled MMA
ElementA/B     = cutlass::nv_float4_t<cutlass::float_e2m1_t>;// NVFP4: E2M1 values + block scale
ElementC/D     = cutlass::bfloat16_t;                        // BF16 in/out
// cluster = 1×1×1 (GeForce), accumulation FP32
```

The FP8 variant (`fp8_gemm.cu`) swaps the operand type to
`cutlass::mx_float8_t<cutlass::float_e4m3_t>` (MXFP8: E4M3 values + UE8M0 per-32
scales), mirroring CUTLASS example 79c. A few notes:

- **`Sm120` is correct for sm_121.** CUTLASS's `Sm120` arch tag guards on
  `CUTLASS_ARCH_MMA_SM120_SUPPORTED || ..._SM121_SUPPORTED`, so the GB10 (sm_121)
  uses this path. (And the build must actually target `sm_121a` — getting the CMake
  CUDA arch wrong silently compiles for Turing and the MMA fails at runtime;
  Appendix C / a hard-won CMake lesson.)
- **Accumulation is FP32** even with 4-bit inputs — the products are tiny-precision
  but the sum is full-precision, which is what keeps GEMM error at the Chapter-12
  noise floor rather than compounding.

Measured: **251.8 TFLOP/s** at 2048³ NVFP4, **161 TFLOP/s** FP8 — ~50% / ~64% of
the relevant dense peaks with zero hand-tuning, bit-identical to CUTLASS's host
reference. That validated the whole toolchain (CMake arch wiring, the block-scaled
MMA path) before any model code existed.

## 13.5 The scale-factor layout — and a subtle bug

Here is the most instructive part of this chapter. The block scales are not stored
in the obvious flat layout; CUTLASS uses an internal **tiled/swizzled** layout for
the scale-factor tensors (`SFA` for the activation, `SFB` for the weight) so the
MMA can fetch a tile's scales efficiently (TMA-aligned). When you write the
quantized weights, you must write the scales into *that* layout, not a naïve one.

The project learned this the hard way. The first weight-quant code computed FP4
byte indices for the column-major B (weight) with a **hand-rolled formula**
`byte_idx = k·(N/2) + (n/2)`, assuming a flat column-by-column packing. It produced
*correct-looking* results on structured test inputs (all-ones, row-varying,
K-axis-constant-within-block) but **uncorrelated output on random per-element
weights**. The reason: the naïve formula matched CUTLASS's real layout only for
column 0; for $k>0$ the stride was wrong, so CUTLASS re-read column-0 bytes for
every $K$ position. With weights constant along $K$ (the structured tests) that
looks fine — column 0 is representative — but with real K-varying weights the
variation was erased and the output collapsed toward the all-ones answer.

**The fix:** never hand-roll sub-byte indexing under a CUTLASS layout. Route the
write through CuTe, which *knows* the real layout:

```cpp
auto stride_B = cutlass::make_cute_packed_stride(
    typename Gemm::GemmKernel::StrideB{}, {N, K, 1});
auto layout_B = cute::make_layout(cute::make_shape(N, K, 1), stride_B);
auto w_tensor = cute::make_tensor(
    cute::recast_ptr<cutlass::float_e2m1_t>(W_packed.data()), layout_B);
// assign through the tensor — CuTe handles packing/swizzle/tiling:
w_tensor(n, k, 0) = cutlass::float_e2m1_t(value / scale);
```

After the fix, random-data `Linear` hit **cos 0.989 / rel-L2 0.15** vs an FP32
reference — squarely within FP4 noise (Chapter 12). The lesson generalizes to *any*
sub-byte format under a tiled layout: use `cute::recast_ptr` + `cute::make_tensor`
and assign through the tensor; the same trap waits in the activation-quant kernel if
it ever hand-indexes. (Appendix B has the full debugging trace; this is also why
the structured-input tests *passed* while the model was wrong — a cautionary tale
about test coverage.)

## 13.6 The `Linear` class

`Linear` (`linear.{h,cu}`) wraps `FP4Gemm`/`FP8Gemm` into the layer the model uses:

```
 y = activation(BF16) @ W^T(quantized) + bias(BF16, optional)
```

Two phases:

**Construction (once).** The weight is brought into the GEMM's expected operand
layout. There are two source paths, unified by the `PreQuantNVFP4` descriptor
(which, despite the name, carries either):

- **W_preq (NVFP4 from disk):** the F2K file already holds packed E2M1 `[N,K/2]` +
  E4M3 scales `[N,K/16]` (Chapter 14). Construction copies these into the CUTLASS
  SFB layout via the CuTe path of §13.5 — no quantization, just a relayout.
- **W_bf16 (MXFP8):** the F2K holds BF16 weights; construction **quantizes** them to
  MXFP8 (E4M3 + UE8M0) at build time, staging into the FP8 SFB layout. (There is
  *also* an on-disk MXFP8 path — `f2k_convert --quant mxfp8` — so construction can
  instead copy pre-quantized FP8, cutting build time from ~76 s to ~5.3 s; Chapter
  14.)

`make_weight_view` routes on the on-disk dtype: an `F8_E4M3` tensor → pre-quant
copy; a `BF16` tensor → quantize-at-construction.

**Forward (every step).** The BF16 activation is **dynamically quantized on the
device** into a matching FP4+SFA (or FP8) pair (`quant_for_A_kernel` /
`quant_for_A_fp8_kernel` — a warp computes each row's block scales and packs the
values), then `FP4Gemm::run()` executes, then bias is added. Activation quant is
*per-forward* and on-device because activations change every step (unlike weights,
quantized once). Quantizing activations too — not just weights — is why the end-to-end
error (Chapter 12, cos 0.963 NVFP4) is a touch worse than weight-only fake-quant
(12.95% image): both operands are low-precision.

## 13.7 The alignment constraints (and where they come from)

The Blackwell block-scaled GEMM tiles impose:

$$
M\ (\text{batch rows}) \%\,128 = 0,\qquad N\ (\text{out features}) \%\,128 = 0,\qquad K\ (\text{in features}) \%\,64 = 0.
$$

These are not arbitrary — they are the MMA/TMA tile sizes. Consequences felt
throughout the model:

- **$N\%128, K\%64$** is exactly the eligibility test in `f2k_convert` (Chapter 12):
  a weight that doesn't tile stays BF16.
- **$M\%128$** means the *token count* fed to a `Linear` must be a multiple of 128.
  This is why `seq_img` must be a multiple of 128 (Chapter 03's resolution
  validation), why the `ModulationMLP` pads to an internal $M=128$ and reads row 0
  (Chapter 07), and why prompts are padded to 512 (Chapter 05). The constraint
  propagates from the MMA tile all the way up to the CLI's allowed resolutions.

(The header notes these may later be relaxed with a padding workspace; for now the
whole pipeline is arranged so they hold naturally.)

## 13.8 Where this lives in the code

| Concept | Code |
|---|---|
| NVFP4 GEMM handle | `fp4_gemm.{h,cu}` (`FP4Gemm`), from CUTLASS ex 79a |
| MXFP8 GEMM handle | `fp8_gemm.{h,cu}` (`FP8Gemm`), from CUTLASS ex 79c |
| the layer | `linear.{h,cu}` (`Linear`, `PreQuantNVFP4`, `Precision`) |
| weight relayout (CuTe) | `linear.cu` construction |
| on-device activation quant | `linear.cu` `quant_for_A_kernel` / `..._fp8_kernel` |
| smoke/perf/correctness | `tests/test_fp4_gemm.cu`, `test_fp8_gemm.cu`, `test_linear*.cu` |

## 13.9 Summary and what to carry forward

- Every `Linear` is a GEMM $Y=XW^\top$; GEMM throughput sets the per-step cost.
- Blackwell's tensor cores consume **block-scaled NVFP4/MXFP8 natively** (scales
  applied inside the MMA, FP32 accumulate) — no separate dequant.
- We use **CUTLASS** (`Sm120` arch tag covers sm_121, `OpClassBlockScaledTensorOp`)
  rather than hand-rolling; measured 252/161 TFLOP/s, bit-exact to the reference.
- The scale-factor tensors use a **tiled/swizzled layout**; write them through
  **`cute::recast_ptr` + `make_tensor`**, never a hand-rolled byte index — the
  hand-rolled version passed structured tests but corrupted random data.
- `Linear` repacks the weight once (copy pre-quant NVFP4, or quantize BF16→MXFP8)
  and **dynamically quantizes activations on-device** each forward.
- The tile constraints **$M\%128, N\%128, K\%64$** propagate up to resolution,
  sequence, and padding choices across the whole pipeline.

Chapter 14 covers where these quantized weights live — the F2K on-disk format, the
mmap loader, and the router that turns 233 tensor names into structured roles.

---

### Exercises

1. **Why FP32 accumulate.** Argue why accumulating 4-bit products in FP32 (not FP4)
   keeps GEMM error at the per-element quantization floor rather than growing with
   $K$.
2. **The structured-test trap.** Explain precisely why the hand-rolled byte index
   passed all-ones and K-constant tests but failed on random weights. What test
   *would* have caught it, and why is it now `tests/test_linear.cu`'s random case?
3. **Constraint propagation.** Trace $M\%128=0$ from the MMA tile to: the allowed
   resolutions, the modulation MLP's internal padding, and the 512 prompt length.
4. **Two weight paths.** Compare W_preq (NVFP4 copy) vs W_bf16 (MXFP8 quant at
   ctor) vs on-disk MXFP8 (FP8 copy) in build time and what each reads from disk.
   Why does on-disk MXFP8 cut build from 76 s to 5.3 s?

*Next: [Chapter 14 — Weights on disk: the F2K format, loader & router](14-f2k-format.md).*
