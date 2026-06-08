# Chapter 26 — Assembling the VAE Encoder

> *Goal of this chapter:* build the **encoder** half of the VAE — the map from an
> $[3, H, W]$ image back to a $[32, H/8, W/8]$ latent. The course used the decoder
> (Chapter 20) for text-to-image; the encoder is what makes *image*-conditioned
> generation possible (Chapter 27: img2img, inpainting, upscaling). It is a near-exact
> **mirror** of the decoder, reusing the same `ResnetBlock`, `VAEAttention`, `Conv2d`,
> GroupNorm and SiLU primitives — the one genuinely new piece is the **stride-2
> downsample with diffusers' asymmetric pad**. Anchored to `vae_encoder.{h,cu}`,
> `tools/cmp_vae_encode.cu`, `tools/diffusers_vae_encode_dump.py`.
>
> *Prerequisites:* Chapters 03 (latent/VAE role, the bn boundary), 19 (cuDNN conv),
> 20 (the decoder — this chapter is its reflection), 15 (GroupNorm/SiLU).

---

## 26.1 The encoder's job and shape

The decoder of Chapter 20 expands a latent $8\times$ to an image. The encoder runs the
same gauntlet backwards: it **contracts** an $[3, H, W]$ image $8\times$ spatially to a
latent grid, taking three $2\times$ **downsamples** (2×2×2 = 8×), a self-attention at the
lowest resolution, and a final projection. Where the decoder *tapered* channels up as
resolution rose (512→…→3), the encoder *grows* them as resolution falls
(3→128→256→512), the usual autoencoder hourglass.

One subtlety up front: the encoder's last conv emits **64** channels, not 32. The VAE
is variational — the encoder outputs the parameters of a Gaussian *posterior*,
$2\times$ the latent channels: 32 for the mean and 32 for the log-variance, concatenated
on the channel axis. These 64 channels are the **moments**; §26.5 turns them into the
32-channel latent we actually use.

The encoder weights were in `vae.f2k1` all along — under the `encoder.` prefix, plus a
sibling `quant_conv.` — but only the `decoder.` half had been wired into code. This
chapter wires the other half; no reconversion was needed.

## 26.2 A mirror built from Chapter 20's primitives

The encoder needs nothing new in the primitive library. It is assembled from exactly
the pieces Chapter 20 introduced:

- **`ResnetBlock`** (`vae_resnet.h`) — GroupNorm→SiLU→conv ×2 + optional `conv_shortcut`,
  identical to the decoder's. The encoder's first resnet in blocks 1 and 2 changes channel
  count (128→256, 256→512) and so carries a $1\times1$ shortcut, exactly as the decoder's
  channel-changing resnets did.
- **`VAEAttention`** (`vae_attn.h`) — the 1-head, $D=512$ spatial self-attention, the same
  object the decoder's mid-block uses, reusing the Part-IV flash kernel.
- **`Conv2d`** (`conv2d.h`) — the cuDNN-9 graph-API conv of Chapter 19, here also driven at
  **stride 2** for the downsamples.
- **GroupNorm / SiLU** kernels (Chapter 15) for `conv_norm_out`.

Because the building blocks are shared and config-driven, `VAEEncoder` is mostly
*plumbing*: instantiate the right blocks at the right channel counts and thread the data
through a ping-pong workspace (§26.6). The only code that did not already exist is the
downsample pad.

## 26.3 The asymmetric downsample (the one real subtlety)

A downsample is "a $3\times3$ conv with stride 2" — but *which padding* is a silent
contract, and getting it wrong corrupts the latent at every block boundary. diffusers'
`Downsample2D` has two modes, selected by the `downsample_padding` the encoder passes:

```python
# diffusers Downsample2D.forward, when use_conv and padding == 0:
pad = (0, 1, 0, 1)                       # right + bottom only
hidden = F.pad(hidden, pad)              # asymmetric, NOT symmetric pad-1
hidden = self.conv(hidden)               # 3×3, stride 2, padding 0
```

The VAE `Encoder` is built with **`downsample_padding=0`**, so it takes the asymmetric
branch: pad the input by **one column on the right and one row on the bottom only**, then
a stride-2 conv with *no* padding. This is *not* the same as a symmetric `padding=1`
stride-2 conv — the two differ at the borders, and a symmetric pad would shift the entire
feature map by a fraction of a pixel, which the golden test (§26.7) catches immediately.

cuDNN's 2D convolution descriptor only expresses *symmetric* per-dimension padding, so we
can't ask it for `(0,1,0,1)` directly. We pre-pad instead, with a tiny kernel:

```cuda
// pad_br_kernel: dst[N,C,H+1,W+1] = src[N,C,H,W], bottom row + right col zero-filled
__global__ void pad_br_kernel(const __nv_bfloat16* src, __nv_bfloat16* dst,
                              int N, int C, int H, int W) {
    const int Hd = H + 1, Wd = W + 1;
    /* … index (n,c,h,w); copy src where h<H && w<W, else 0 … */
}
```

then run the `Conv2d` with `stride=2, padding=0` on the $[N,C,H{+}1,W{+}1]$ buffer. The
output spatial size is $\lfloor (H{+}1-3)/2\rfloor + 1 = H/2$ for even $H$ — exactly the
halving we want. This `pad → stride-2 conv` is the whole downsample.

## 26.4 The assembled architecture

From `vae_encoder.h`, the full encoder (image `[3, H, W]` → moments `[64, H/8, W/8]`),
with `block_out_channels = [128, 256, 512, 512]`, `layers_per_block = 2`:

```
 conv_in            3 → 128  (3×3)
 down_blocks[0]: 2× Resnet128                       → Downsample 128→128 (½)
 down_blocks[1]: Resnet128→256(shortcut), Resnet256 → Downsample 256→256 (¼)
 down_blocks[2]: Resnet256→512(shortcut), Resnet512 → Downsample 512→512 (⅛)
 down_blocks[3]: 2× Resnet512                                   (no downsample)
 mid_block:    Resnet512 → SelfAttn(1 head, 512) → Resnet512
 conv_norm_out (GroupNorm32) → SiLU → conv_out 512 → 64  (3×3)
 quant_conv         64 → 64  (1×1)
 moments [64, H/8, W/8]
```

Read it against the decoder of §20.4 and it is the same diagram reflected: three
downsamples (on blocks 0–2) where the decoder had three upsamples; the final block (3)
has *no* downsample, just as the decoder's last up-block had no upsample; the mid-block is
identical (Resnet → attention → Resnet at 512). `conv_out` projects to **64** (the
moments), and a $1\times1$ **`quant_conv`** ($64\to64$) follows — the encoder-side sibling
of the decoder's `post_quant_conv`, the channel remix diffusers applies before forming the
posterior.

## 26.5 Moments, the mean, and the bn boundary

`VAEEncoder::forward` returns the 64-channel moments. diffusers wraps them in a
`DiagonalGaussianDistribution` that splits the channel axis in half:

$$\text{mean} = \text{moments}[:32], \qquad \log\sigma^2 = \text{moments}[32:].$$

For deterministic, reproducible image conditioning we use the distribution's **mode** — the
**mean** — not a sample. And because the channel axis is the outermost (NCHW), the first 32
channels of the $[64, H/8, W/8]$ moments tensor are already a contiguous $[32, H/8, W/8]$
block: the consumer (Chapter 27) just treats the moments pointer as a 32-channel latent.

That mean lives in **VAE latent space**. To feed the transformer it must cross the same
boundary the decoder crossed in reverse (§20.5, Chapter 03). Decode applied
$z\cdot\sqrt{\text{Var}+\epsilon}+\text{mean}$ to *de*-standardize; encode applies the
**inverse**,

$$z_{\text{tf}} = \frac{z_{\text{vae}} - \text{bn.mean}}{\sqrt{\text{bn.var}+\epsilon}},$$

on the 128-channel *patch* latent after `patchify`. That normalization is part of the
*conditioning* path, not the encoder object, so it lives in Chapter 27 — but it is exactly
the bn de-norm of Chapter 20 run backwards, with the same `bn.running_{mean,var}` and
$\epsilon=10^{-4}$.

## 26.6 The workspace strategy

Same idea as the decoder (§20.6): a **ping-pong** pair of device buffers sized to the
largest stage, swapped on each block, plus a dedicated **pad buffer** for the
`(0,1,0,1)` pre-pad and the shared sub-component scratch. The largest stage is `conv_in`'s
output at full resolution, $[128, H, W]$ — the early, high-resolution feature maps
dominate, as in any encoder. At 1024px that is a few hundred MiB per buffer; comfortably
within unified memory. The arena is allocated once when the `(res, precision)` pipeline is
built (Chapter 28) and reused for every encode.

## 26.7 Correctness: the golden test

The encoder is validated with the same **inject-and-compare** discipline as the decoder
(Chapter 25), against diffusers' own `AutoencoderKLFlux2`:

1. `tools/diffusers_vae_encode_dump.py` preprocesses an image to model space $[-1,1]$ and
   dumps **two** files in the course's `int32 C,H,W + float32` layout: the *exact*
   preprocessed input pixels, and diffusers' posterior **mean** latent.
2. `tools/cmp_vae_encode.cu` uploads those exact pixels, runs `VAEEncoder`, takes the mean
   (first 32 of 64 channels), and reports cosine similarity against the golden.

Dumping the preprocessed *pixels* (not just the source image) removes any
resize/decode mismatch from the comparison — both sides see identical input. The result:

| Resolution | latent | mid-attn tokens $S$ | cos similarity |
|---|---|---|---|
| 256px | $[32,32,32]$ | 1 024 | **0.99975** |
| 512px | $[32,64,64]$ | 4 096 | **0.99970** |
| 1024px | $[32,128,128]$ | 16 384 | **0.99967** |
| 2048px | $[32,256,256]$ | 65 536 | **0.99960** |

The residual (cos $\approx 10^{-3}$ off unity) is the expected BF16-vs-FP32 accumulation
gap, not a bug — the same magnitude as every other component golden in Chapter 25.

The 2048px row carries a war story. The encoder's **mid-block attention** runs at the
latent resolution, so its sequence length is $S = (H/8)^2$ — and at 2048px that is exactly
**65 536**. The first time this ran it *failed at launch* inside the attention's `to_q`
projection, and the failure looked for all the world like "2048 needs too much memory." It
was not memory at all: it was a `gridDim.y` limit in `Linear`'s quantize kernel, tripped at
precisely $M=65536$. That is Appendix B's newest entry (B.15), and fixing it is what put
the 2048 row in the table — and unblocked native 2K generation everywhere.

## 26.8 Where this lives in the code

| Concept | Code |
|---|---|
| the encoder | `vae_encoder.{h,cu}` (`VAEEncoder`) |
| asymmetric downsample pad | `pad_br_kernel` / `pad_bottom_right_bf16` (in `vae_encoder.cu`) |
| resnet / attention / conv primitives | `vae_resnet.*`, `vae_attn.*`, `conv2d.*` (Chapters 19–20) |
| moments → mean | first 32 of 64 channels (consumed in `serve.cu`, Chapter 27) |
| bn normalization (encode side) | `serve.cu` img2img path (Chapter 27) |
| `quant_conv` | `VAEEncoder::Config::quant_conv_name` |
| golden tools | `tools/diffusers_vae_encode_dump.py`, `tools/cmp_vae_encode.cu` |

## 26.9 Summary and what to carry forward

- The encoder is the **mirror** of the decoder: `conv_in` → three downsampling blocks → a
  $512$-channel mid-block (Resnet → 1-head $D{=}512$ attention → Resnet) → `conv_norm_out`
  → `conv_out` (to **64** = moments) → `quant_conv`. It reuses every Chapter 20 primitive.
- The one new piece is the **downsample**: diffusers' `downsample_padding=0` means an
  **asymmetric `(0,1,0,1)` pad** then a stride-2, padding-0 conv — not a symmetric pad. We
  pre-pad with `pad_br_kernel` because cuDNN only does symmetric padding.
- The output is **moments**; the deterministic latent is the posterior **mean** (first 32
  channels). Crossing into transformer space applies the **inverse** of Chapter 20's bn
  de-norm.
- Verified against diffusers' encoder at **cos 0.9997** across 256–2048px — the BF16 gap,
  not a bug. The 2048px case surfaced the `gridDim.y` bug (Appendix B.15).

With both halves of the VAE in hand, the autoencoder loop is closed: we can now go *image →
latent → image*, which is the machinery the next chapter builds on.

---

### Exercises

1. **Symmetric vs asymmetric pad.** For an $H\times W$ input, compute the output size and
   the receptive-field offset of (a) `F.pad((0,1,0,1))` + stride-2 padding-0 conv versus
   (b) a symmetric padding-1 stride-2 conv. Which border does each favor, and why would (b)
   shift the whole latent?
2. **Why 64 channels.** Explain the encoder's $2\times$ output width in terms of the
   variational posterior, and why using the **mean** (mode) rather than a sample is the
   right choice for reproducible img2img.
3. **The bn round trip.** Write the encode-side normalization as the exact inverse of
   Chapter 20's decode-side de-normalization, and argue that a sign or $\epsilon$ error here
   would survive a *visual* check but fail the golden.
4. **Mid-attention cost.** The mid-block attention is $S=(H/8)^2$. Tabulate $S$ for
   256/512/1024/2048px and explain why 2048 (and not 1024) was the first to hit the
   `gridDim.y` ceiling.

*Next: [Chapter 27 — Image conditioning: img2img, inpainting & upscaling](27-image-conditioning.md).*
