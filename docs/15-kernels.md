# Chapter 15 — The Core Kernel Library

> *Goal of this chapter:* survey the small, reusable CUDA kernels that glue the
> GEMMs (Chapter 13) and attention (Part IV) together — the normalizations,
> activations, and data-movement ops. None is individually hard; together they are
> the connective tissue of every block, and they share a handful of patterns worth
> internalizing. We give the math for the norms and activations and the standard
> implementation shape. Anchored to `src/backend/cuda/kernels/*`.
>
> *Prerequisites:* Chapters 04–10 (where each kernel is used); CUDA warp/block
> reductions.

---

## 15.1 The cast of kernels

From `src/backend/cuda/kernels/`:

| kernel | op | used by |
|---|---|---|
| `rmsnorm` | RMS normalization | every block's norms + QK-norm (ch 08–09), Qwen3 (ch 21) |
| `layernorm` | mean/var layer norm | the centered LayerNorm in the model |
| `groupnorm` | grouped channel norm | VAE resnet/conv blocks (ch 19–20) |
| `silu_mul` / `swiglu` | $\text{SiLU}(g)\cdot u$ gated activation | MLP sub-layers (ch 08–09) |
| `modulation` | `(1+scale)·x+shift`, `y+=gate·δ` | AdaLN (ch 07) |
| `rope` / `rope_4axis` | rotary position | Q/K (ch 06) |
| `patchify` | latent grid ↔ token sequence | I/O of the transformer (ch 03) |
| `embed_lookup` | vocab gather | Qwen3 token embeddings (ch 21) |
| `qkv_mlp_split` / `seq_ops` | split/concat/take-tail | fused blocks, stream transitions (ch 09–10) |
| `transpose_chw` | NCHW↔NHWC | VAE conv layout (ch 19) |
| `upsample` | 2× nearest upsample | VAE decoder (ch 20) |

These are all **memory-bound** (Chapter 11): they read and write activations with
little arithmetic per byte. Individually they are negligible next to the GEMMs and
attention — but there are many, in every block, so they must be clean and fused
where cheap. The design philosophy is *write each once, reuse everywhere*: the same
`rmsnorm_bf16` serves the block-entry norm, the per-head QK-norm, and the Qwen3
encoder, just called with different `batch_rows` and gains.

## 15.2 The shared implementation patterns

Almost every kernel here follows one of two shapes:

**(a) One CTA (or warp) per row, with a reduction.** Norms reduce over the feature
dimension. The standard pattern: each thread accumulates a partial sum over its
strided slice of the row, a **warp reduce** (`__shfl_xor`) combines lanes, a
**block reduce** (via a 32-slot shared scratch) combines warps, and the result is
broadcast back. This is exactly the `warp_reduce_sum/max` + `block_reduce` helpers
that also appear in the attention fallback kernel (Part IV).

**(b) One thread per element (elementwise).** Activations, modulation, RoPE,
patchify: a flat grid where each thread handles one (or a few) output elements.

Two conventions hold throughout:

- **BF16 storage, FP32 compute.** Inputs/outputs are BF16 (to move fewer bytes —
  Chapter 11), but reductions and transcendentals accumulate in FP32 to avoid
  precision loss. Norms especially need FP32 sums; a BF16 sum over 4096 elements
  would lose low bits.
- **Broadcast parameters are small BF16 vectors.** Scale/shift/gate and norm gains
  are `[hidden]` vectors broadcast across all rows (Chapter 07); RoPE tables are
  FP32 `[seq, D/2]` (Chapter 06).

## 15.3 The normalizations

Three normalizations appear, differing in *what* they normalize over.

**RMSNorm** — the dominant one in the transformer. It rescales a vector by its
root-mean-square, with a learned per-channel gain $\gamma$ (no mean subtraction, no
bias):

$$
\text{RMSNorm}(x)_i = \frac{x_i}{\sqrt{\tfrac{1}{d}\sum_j x_j^2 + \epsilon}}\;\gamma_i,
\qquad \epsilon = 10^{-6}.
$$

It is cheaper than LayerNorm (one reduction, no mean) and empirically as good for
transformers, which is why modern models (LLaMA, Qwen3, FLUX) use it. In FLUX the
block-entry RMSNorm uses an **all-ones** gain (statistics-only — the learned affine
is replaced by AdaLN modulate, Chapter 07), while the **QK-norm** uses *learned*
per-head gains (Chapter 08). Same kernel, different `gain` argument and
`batch_rows = S·n_heads` for the per-head case.

**LayerNorm** — subtract the mean, divide by the std, affine:

$$
\text{LN}(x)_i = \frac{x_i - \mu}{\sqrt{\sigma^2 + \epsilon}}\,\gamma_i + \beta_i,
\quad \mu=\tfrac1d\sum_j x_j,\ \sigma^2=\tfrac1d\sum_j (x_j-\mu)^2 .
$$

Used where the model specifies a centered norm. Two reductions (mean, then
variance) instead of RMSNorm's one.

**GroupNorm** — used in the **VAE** (Chapter 19–20), where activations are
$[C, H, W]$ convolutional feature maps, not token vectors. It splits the $C$
channels into $G$ groups and normalizes over each group's channels *and* spatial
extent:

$$
\text{GN}(x)_{c,h,w} = \frac{x_{c,h,w} - \mu_g}{\sqrt{\sigma_g^2 + \epsilon}}\,\gamma_c + \beta_c,
$$

with $\mu_g,\sigma_g$ computed over all $(c\in g, h, w)$. The VAE uses GroupNorm-32.
Its reduction is over a much larger set (channels × spatial), so the kernel reduces
over a 3D extent per group.

## 15.4 The activations

**SiLU** (a.k.a. swish): $\text{SiLU}(x) = x\,\sigma(x) = \dfrac{x}{1+e^{-x}}$. A
smooth, non-monotonic activation. Appears alone (`silu_inplace_bf16`, in the
modulation MLP, Chapter 07) and as the gate in:

**SwiGLU** — the gated MLP activation (Chapters 08–09). Given the MLP-input
projection split into a *gate* half $g$ and an *up* half $u$:

$$
\text{SwiGLU}(g,u) = \text{SiLU}(g)\odot u .
$$

`silu_mul`/`swiglu` fuse the SiLU and the elementwise multiply into one kernel — no
intermediate write of $\text{SiLU}(g)$. Gated activations consistently beat plain
$\text{Linear}\to\text{act}\to\text{Linear}$ at equal parameters, which is why the
whole model's MLPs use them.

## 15.5 The data-movement kernels

These do no arithmetic — they rearrange — but their layouts are contracts (get one
wrong and the model is subtly broken, as Chapter 03's patchify order showed).

- **`patchify` / `unpatchify`** — latent grid $\leftrightarrow$ token sequence, with
  the `c·p·p+ph·p+pw` packing (Chapter 03). One CTA per token.
- **`qkv_mlp_split` / `split_half`** — view a fused GEMM output as its constituent
  tensors (Q,K,V,gate,up or gate,up). No copy where contiguous; a placement copy
  for the `[attn ‖ mlp]` concat (Chapter 09).
- **`seq_ops`: `concat_two_streams`, `take_tail`** — the double→single stream merge
  (text-first) and the image-tail extraction (Chapter 10). One CTA per output row,
  striding the hidden dim.
- **`transpose_chw`** — NCHW ↔ NHWC for the VAE convolutions, because the fast cuDNN
  Blackwell conv engines require NHWC (Chapter 19). A tiled transpose.
- **`upsample`** — 2× nearest-neighbor spatial upsample in the VAE decoder (Chapter
  20).
- **`embed_lookup`** — gather rows of the BF16 token-embedding table by token id, for
  the Qwen3 encoder (Chapter 21).

## 15.6 Accuracy: small kernels, tight tolerances

Each kernel has a golden test against an FP32 host reference (`tests/test_*`):
`test_rmsnorm`, `test_groupnorm`, `test_swiglu`, `test_modulation`, `test_rope`,
`test_rope_4axis`, `test_patchify`, `test_upsample`, … Typical results are
max-abs-error ~0.004 (pure BF16 rounding) — these kernels are *exact* up to the
storage precision. Because they are simple and individually tested, debugging
focuses on the hard parts (GEMM layout, attention, the architecture wiring), with
the glue trusted. This is the payoff of the "write once, test once, reuse
everywhere" discipline: the connective tissue is not where bugs hide.

## 15.7 Where this lives in the code

All under `src/backend/cuda/kernels/`: `rmsnorm`, `layernorm`, `groupnorm`,
`silu_mul`, `modulation`, `rope`, `rope_4axis`, `patchify`, `embed_lookup`,
`qkv_mlp_split`, `seq_ops`, `transpose_chw`, `upsample` (each `.h`/`.cu`), with
`tests/test_<kernel>.cu` for the validated ones. The attention kernels live one
level up in `attention.{h,cu}` and are Part IV.

## 15.8 Summary and what to carry forward

- The kernel library is the **memory-bound connective tissue**: norms, activations,
  and data movement, in every block, cheap individually but numerous.
- Two patterns: **row+reduction** (norms — warp→block reduce, FP32 accumulate) and
  **elementwise** (activations, RoPE, patchify). Convention: **BF16 storage, FP32
  compute**; broadcast params are small BF16 vectors.
- **RMSNorm** dominates (statistics-only at block entry, learned-gain for QK-norm);
  **LayerNorm** where centered; **GroupNorm-32** in the VAE.
- **SwiGLU** (`SiLU(g)·u`, fused) is the MLP activation.
- Data-movement kernels (`patchify`, `seq_ops`, `transpose_chw`, …) carry **layout
  contracts**, not math — and those contracts are where Chapter-03-style bugs live.
- Each is golden-tested to BF16 precision, so the glue is trusted and debugging
  focuses on GEMM/attention/wiring.

That completes Part III — the GPU foundations: the roofline (11), the formats (12),
the GEMM (13), the weight pipeline (14), and the kernels (15). Part IV now goes deep
on the one kernel that earned its own part: **attention**, the memory-bound heart of
every block, taken from a naïve loop to peak HBM bandwidth.

---

### Exercises

1. **RMS vs Layer.** Write both norms' reductions and explain why RMSNorm is cheaper
   and why "all-ones gain" makes it statistics-only. Why does FLUX pair that with
   AdaLN modulate?
2. **FP32 accumulate.** Estimate the error of summing 4096 BF16 squares in BF16 vs
   FP32 for an RMSNorm, and explain why the kernel accumulates in FP32.
3. **Fused SwiGLU.** Show the bytes saved by `silu_mul` fusing SiLU and the multiply
   vs computing `SiLU(g)` to a temp and multiplying separately, at $S=4608$,
   ffn=12288.
4. **A layout contract.** Pick `concat_two_streams` (text-first) and list everything
   downstream that breaks if you concatenate image-first instead (Chapter 6/10).

*Next: [Chapter 16 — Attention and the memory wall](16-attention-math.md), opening Part IV.*
