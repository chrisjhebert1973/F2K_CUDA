# Chapter 20 — Assembling the VAE Decoder

> *Goal of this chapter:* build the full decoder that turns the clean
> $[32, H/8, W/8]$ latent into an $[3, H, W]$ image. We cover the ResnetBlock
> primitive, the mid-block **spatial self-attention** (which reuses the Part-IV
> attention kernel at $D=512$), upsampling, the assembled architecture, the
> `post_quant_conv` / batch-norm boundary (Chapter 03), the workspace strategy, and
> the decisive correctness test. Anchored to `vae_decoder.{h,cu}`,
> `vae_resnet.{h,cu}`, `vae_attn.{h,cu}`.
>
> *Prerequisites:* Chapters 03 (latent/VAE role, the bugs), 15 (GroupNorm/SiLU/
> upsample), 16–18 (attention), 19 (conv).

---

## 20.1 The decoder's job and shape

From Chapter 03: the transformer emits a clean latent, `unpatchify` turns it back
into a $[32, H/8, W/8]$ grid, and the decoder expands it **8×** spatially to
$[3, H, W]$ RGB. The expansion is a stack of residual convolution blocks with three
$2\times$ upsamples (2×2×2 = 8×), a self-attention at the lowest resolution, and a
final projection to 3 channels. All the heavy lifting is the convolutions of
Chapter 19; this chapter is how they are wired.

## 20.2 The ResnetBlock primitive

The decoder is mostly **ResnetBlocks** (`vae_resnet.h`), the standard pre-activation
residual conv block:

```
 residual = GroupNorm32(x) → SiLU → conv1 (3×3, may change channels)
          → GroupNorm32     → SiLU → conv2 (3×3, channels stay)
 skip     = x              if C_in == C_out
          = conv_shortcut(x) (1×1)   if C_in != C_out
 y        = skip + residual
```

Two GroupNorms (Chapter 15, 32 groups, FP32-accumulated stats), two SiLUs, two
$3\times3$ convs (Chapter 19), and a residual add. The **`conv_shortcut`** is a
$1\times1$ conv that appears only when a block changes channel count (e.g. 512→256),
projecting the skip path to match. Everything is BF16 with FP32 accumulation inside
GroupNorm and the cuDNN GEMM. This one primitive, instantiated at various channel
counts, fills the mid-block and all four up-blocks.

## 20.3 The mid-block spatial attention

At the lowest resolution (deepest, 512 channels) the decoder has a **self-attention
over spatial positions** — the same attention math as the transformer, but here the
"tokens" are the $H\times W$ spatial locations and the "channels" are the head
dimension. It is **1 head, head-dim 512** (`vae_attn.h`): a GroupNorm, then `to_q`,
`to_k`, `to_v` projections, the attention, and a `to_out.0` projection, added back as
a residual.

This is *why the Part-IV attention kernel was written to also serve $D=512$*
(Chapter 16–17). At the VAE's resolutions the sequence length $S=H\times W$ can be
huge — **16384** at the mid-block for 1024px output — so this attention is the
$S=16384, D=512$ shape in the benchmarks. It uses the **warp-per-query flash
kernel** (not the $D=128$ mma kernel, whose tensor-core tiling doesn't fit $D=512$'s
smem) — online softmax keeps its smem bounded by the tile, so even $S=16384$ fits.
The mid-block attention lets the decoder model long-range spatial structure that
pure local convolutions cannot.

## 20.4 The assembled architecture

From `vae_decoder.h`, the full decoder (latent `[32, H/8, W/8]` → image `[3, H, W]`):

```
 (post_quant_conv 1×1, 32→32)          # §20.5
 conv_in            32 → 512  (3×3)
 mid_block:    Resnet512 → SelfAttn(1 head, 512) → Resnet512
 up_blocks[0]: 3× Resnet512        → Upsample ×2 → conv 512→512
 up_blocks[1]: 3× Resnet512        → Upsample ×2 → conv 512→512
 up_blocks[2]: Resnet512→256(shortcut), 2× Resnet256 → Upsample ×2 → conv 256→256
 up_blocks[3]: Resnet256→128(shortcut), 2× Resnet128                (no upsample)
 conv_norm_out (GroupNorm32) → SiLU → conv_out 128 → 3  (3×3)
 image [3, H, W]
```

The three `Upsample ×2` stages (in up-blocks 0, 1, 2) give the 8× total; up-block 3
has no upsample (the spatial size is already $H, W$ after three doublings from
$H/8$). Channels taper 512 → 512 → 512 → 256 → 128 → 3 as resolution rises — the
usual decoder shape (more channels where the map is small, fewer where it is large).
`Upsample` is the nearest-neighbor 2× kernel (Chapter 15) followed by a $3\times3$
conv.

## 20.5 The latent/decoder boundary (recap from Chapter 03)

Two transforms at the input boundary are easy to miss and were real bugs:

- **`post_quant_conv`** — a $1\times1$ conv ($32\to32$, per-pixel channel remix +
  bias) that diffusers runs *before* the decoder body
  (`autoencoder_kl_flux2._decode`: `z = post_quant_conv(z); decoder(z)`). It is a
  **sibling** of the `decoder.` prefix, not under it, so a decoder starting at
  `conv_in` skips it — and skipping it turned sharp images to "mush." It is now
  folded into `VAEDecoder` (`Config::post_quant_conv_name`), applied at the top of
  `forward`, and skipped gracefully if absent.
- **The batch-norm de-normalization** — applied to the 128-channel *patch* latent
  *before* `unpatchify` (so, in `generate.cu`, not inside the decoder): $z\cdot
  \sqrt{\text{Var}+\epsilon}+\text{mean}$, inverting the encoder's standardization.
  Skipping it caused the $2\times2$ patch-grid checkerboard. (Full treatment:
  Chapter 03; war story: Appendix B.)

## 20.6 The workspace strategy

The decoder threads activations through a small number of reused device buffers. The
classic trick is **ping-pong**: two buffers, each conv/resnet reads one and writes
the other, swapping each step, plus scratch for the upsample and the
per-sub-component working set (the cuDNN plan workspace, the attention scratch). The
arena is sized to the **maximum** over stages and reused, since a stage's
intermediates die at its boundary. At 128px the total is ~176 MiB; at 1024px it is a
few GiB (dominated by the cuDNN conv workspace and the large early feature maps),
well within unified memory.

## 20.7 Performance and the decisive correctness test

With the cuDNN graph-API convs (Chapter 19), the decoder is fast: **128px forward
~10 ms** (was 314 ms), **1024px VAE decode ~1.35 s** (was 18.7 s) — no longer a
bottleneck.

Correctness is established two ways:

- **Component tests:** `test_vae_resnet`, `test_vae_attn`, `test_groupnorm`,
  `test_upsample`, `test_conv2d` each check a piece against an FP32 reference.
- **The decisive end-to-end test:** feed diffusers' **own ground-truth latent**
  (dumped via `tools/diffusers_latent_dump.py`, `output_type="latent"`) through *our*
  decoder and compare to diffusers' decoded image. Result: a **sharp, photoreal cat
  matching diffusers**. This is the test that proved the decoder correct *and*
  isolated the earlier "mush" to the missing `post_quant_conv` — because feeding a
  known-good latent removed the transformer trajectory from suspicion, localizing the
  fault to the VAE. (`tests/test_vae_decoder.cu`; Chapter 25 for the methodology.)

> The pattern (Chapter 25): to debug a multi-stage pipeline, **inject a known-good
> intermediate** from a reference at the stage boundary. Feeding diffusers' latent
> into our decoder split "is the latent wrong?" from "is the decoder wrong?" in one
> experiment.

## 20.8 Where this lives in the code

| Concept | Code |
|---|---|
| the decoder | `vae_decoder.{h,cu}` (`VAEDecoder`) |
| resnet primitive | `vae_resnet.{h,cu}` (`ResnetBlock`) |
| mid-block attention ($D=512$) | `vae_attn.{h,cu}` → `attention.cu` (flash, $D=512$) |
| convs (graph API) | `conv2d.{h,cu}` (Chapter 19) |
| group norm / SiLU / upsample | `kernels/groupnorm.*`, `silu_*`, `upsample.*` |
| `post_quant_conv` | `VAEDecoder::Config::post_quant_conv_name` |
| bn de-norm | `generate.cu` (on the patch latent, pre-unpatchify) |
| tests | `tests/test_vae_decoder.cu` (+ component tests) |

## 20.9 Summary and what to carry forward

- The decoder expands the $[32,H/8,W/8]$ latent to $[3,H,W]$ via **ResnetBlocks**
  (GroupNorm→SiLU→conv ×2 + shortcut), **three $2\times$ upsamples** (=8×), a
  **mid-block spatial self-attention** ($1$ head, $D=512$, reusing the Part-IV flash
  kernel at $S$ up to 16384), and a final GroupNorm→SiLU→conv to 3 channels.
- Two boundary transforms — **`post_quant_conv`** (folded in) and the **bn de-norm**
  (in `generate.cu`) — are essential and were the "mush" and "patch-grid" bugs.
- A **ping-pong workspace** sized to the max stage keeps memory modest.
- With graph-API convs, **decode is ~1.35 s at 1024px**; correctness was nailed by
  decoding **diffusers' own latent** — the inject-a-known-good-intermediate technique.

That completes Part V. Part VI covers the text front-end: the Qwen3 encoder
implementation (Chapter 21) and the native BPE tokenizer (Chapter 22) that made the
pipeline self-contained.

---

### Exercises

1. **Channel/resolution taper.** Explain why decoders use *more* channels at low
   resolution and *fewer* at high, and trace the 512→256→128→3 taper against the
   three upsamples.
2. **Why attention in a conv net.** What can the mid-block self-attention model that
   stacked $3\times3$ convs cannot, and why is it placed at the *lowest* resolution
   (think $S=H\cdot W$ cost)?
3. **The boundary bugs.** Describe how `post_quant_conv` (missing) and the bn de-norm
   (missing) each corrupt the image differently (mush vs grid), and why feeding
   diffusers' latent localized the former.
4. **Workspace.** Sketch a ping-pong schedule for `conv_in → resnet → attn → resnet`
   and argue why two buffers + scratch suffice despite the channel changes.

*Next: [Chapter 21 — The Qwen3 encoder](21-qwen3-encoder.md), opening Part VI.*
