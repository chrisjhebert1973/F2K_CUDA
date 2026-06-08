# Chapter 03 — Latent Diffusion & the VAE

> *Goal of this chapter:* explain *where* the diffusion of Chapters 01–02 actually
> happens. FLUX does not denoise pixels; it denoises a compressed **latent**
> produced by a variational autoencoder (VAE). We derive why latent-space
> diffusion is the right move, what a VAE is and what objective trains it, and the
> exact tensor plumbing of the FLUX VAE — channels, the 8× upsample, the
> patch-packing convention, and two normalization subtleties (`post_quant_conv`
> and a batch-norm de-normalization) that were real bugs before they were
> understood.
>
> *Prerequisites:* Chapters 01–02; convolutions; KL divergence and the Gaussian.

---

## 3.1 Why not diffuse pixels directly?

Chapter 00 noted a $1024^2$ RGB image lives in $\mathbb R^{3{,}145{,}728}$. Running
a 9B-parameter transformer's attention over three million spatial positions is
quadratically hopeless, and most of those pixels are perceptually redundant — a
patch of blue sky carries little information per pixel. Two problems, one fix:

1. **Cost.** Attention is $O(S^2)$ in the number of tokens $S$ (Chapter 16).
   Halving the spatial resolution cuts $S$ by 4× and attention cost by 16×.
2. **Redundancy.** Natural images are highly compressible; the "interesting"
   degrees of freedom are far fewer than the pixel count.

**Latent diffusion** (Rombach et al., 2022 — the "Stable Diffusion" insight) puts
both to work: first train an autoencoder that compresses an image into a small
latent and back, then run the *entire* diffusion process (Chapters 01–02) in that
latent space. The expensive transformer never sees a pixel; it sees a latent that
is ~$8\times$ smaller per side and a handful of channels deep. The VAE is trained
once, separately, and frozen; diffusion is trained on its latents.

```
  IMAGE  [3, 1024, 1024]                                  IMAGE  [3, 1024, 1024]
        │                                                        ▲
        │ VAE encoder (train time only)        VAE decoder       │ (inference)
        ▼                                       (ch 19–20)        │
  LATENT [32, 128, 128]  ───── diffuse / denoise here ─────▶  LATENT [32, 128, 128]
        (8× smaller per side, 32 channels)   (Chapters 01–02, 04–10)
```

For us only the **decoder** runs at inference (we generate latents from noise and
decode them); the encoder matters only conceptually and during the project's
validation against diffusers (Chapter 25).

## 3.2 What a VAE is, briefly but honestly

An autoencoder is an encoder $E$ and decoder $D$ trained so $D(E(x))\approx x$. A
plain autoencoder's latent space is an arbitrary, hole-riddled embedding — fine for
reconstruction, bad as a space to *generate* in. A **variational** autoencoder
regularizes the latent to be smooth and roughly Gaussian, so that nearby latents
decode to plausible images and the space has no dead zones.

Formally, the VAE maximizes the **evidence lower bound** (ELBO) on $\log p(x)$:

$$
\log p(x) \;\ge\; \underbrace{\mathbb E_{z\sim q_\phi(z\mid x)}\big[\log p_\psi(x\mid z)\big]}_{\text{reconstruction}}
\;-\; \underbrace{D_{\mathrm{KL}}\!\big(q_\phi(z\mid x)\,\|\,p(z)\big)}_{\text{regularization toward }\mathcal N(0,\mathbf I)} .
$$

The encoder $q_\phi(z\mid x)=\mathcal N(\mu_\phi(x),\sigma_\phi^2(x))$ outputs a
Gaussian per latent; the reconstruction term pushes the decoder to invert it; the
KL term pulls the posterior toward a standard normal prior so the space stays
well-behaved. Image VAEs add perceptual and adversarial losses on top for sharp
reconstructions, but the ELBO is the backbone. **For this course the takeaways
are:** (i) the latent is *approximately* standard-normal-distributed, which is
exactly the distribution rectified flow transports from (Chapter 02 — the noise
endpoint $\mathcal N(0,\mathbf I)$ lives in the *same space* as the latents), and
(ii) the decoder $D=p_\psi(x\mid z)$ is a deterministic-enough map from latent to
pixels that we can treat it as a fixed function.

> A subtlety FLUX's latents care about: the encoder's output is **scaled and
> shifted** to standardize it (so its statistics match the diffusion prior). At
> decode time that transform must be inverted. §3.5 is exactly this, and getting
> it wrong was the "patch-grid" bug.

## 3.3 The FLUX VAE in numbers

From this project's decoder (`src/backend/cuda/vae_decoder.h`) and the patchify
kernel, the concrete shapes are:

| Quantity | Value |
|---|---|
| Latent channels $C$ | **32** |
| Spatial compression | **8×** per side (latent $H_{\text{lat}}=H/8$) |
| Decoder input | `[N, 32, H/8, W/8]` |
| Decoder output | `[N, 3, H, W]` |
| Transformer patch size $p$ | **2** |
| Transformer token channel dim | $p^2 C = 4\times32 = $ **128** |

There are therefore **two** spatial reductions stacked on top of each other, and
keeping them straight is essential:

```
 pixels                VAE encode (÷8)        patchify (÷2, pack into channels)
 [3, 1024, 1024]  ────────────────────▶  [32, 128, 128]  ───────────────────▶  tokens
                                          (VAE latent)        [ (64·64)=4096 tokens, 128 ch ]
                                                              this is what the transformer sees
```

So at 1024px the transformer operates on $64\times64 = 4096$ image tokens, each a
128-dimensional vector — that is the `seq_img = 4096` you saw in Chapter 00 and the
$S=4608$ (with 512 text tokens) of the attention benchmarks. The VAE's 8× and the
patchify's 2× multiply to the 16× total between a pixel side and a token-grid side.

## 3.4 Patchify: turning a latent grid into a token sequence

A transformer consumes a *sequence* of token vectors, not a 2D grid. **Patchify**
reshapes the $[32, H/8, W/8]$ latent into $[(H/16)(W/16),\,128]$ tokens by carving
the latent into $2\times2$ spatial patches and flattening each patch's
$2\times2\times32 = 128$ values into one token vector. The *order* of that
flattening is a convention that must match the training pipeline exactly, or every
token is internally scrambled. From `kernels/patchify.h`:

```
 tokens[b, h_idx·W_p + w_idx,  c·p·p + ph·p + pw]  =  x[b, c, h_idx·p + ph, w_idx·p + pw]
        └────── token index ──────┘  └─ channel within token ─┘
   c (latent channel) is the SLOWEST-varying axis in the packed dim; pw the FASTEST.
   W_p = W/p,  H_p = H/p,  packed channel dim = p·p·C = 128.
```

In words: the token at grid position $(h_{\text{idx}}, w_{\text{idx}})$ holds, for
each latent channel $c$, that channel's $2\times2$ block laid out as
$(ph,pw)\in\{0,1\}^2$, with channel $c$ the outermost loop. This precise order —
`c·p·p + ph·p + pw`, channel-slowest — matches diffusers'
`_patchify_latents + _pack_latents`. An earlier version of this project used the
channel-*fastest* order; the model still ran and produced an image, but a subtly
wrong one, because each token's 128 numbers were permuted relative to what the
weights expected. (Appendix B: "patchify channel order.") `unpatchify` is the exact
inverse and runs after the final denoise step, before the VAE.

> **Why $p=2$ and not patch-in-the-transformer-only?** FLUX folds a $2\times2$
> pixel-shuffle into the token construction so the transformer's sequence length is
> $1/4$ of the latent's spatial size — another bandwidth/compute saving stacked on
> the VAE's. The decoder undoes the VAE's 8×; the patchify's 2× is undone by
> `unpatchify` in latent space.

## 3.5 Two normalization subtleties (and the bugs they caused)

Between "the transformer emitted a clean latent" and "the decoder produces pixels"
sit two transforms that are easy to miss because they live *outside* the obvious
decoder body. Both were real bugs in this project; both are now handled.

### 3.5.1 The batch-norm de-normalization

FLUX's pipeline standardizes the patch latent with a per-channel affine transform
(stored as a `BatchNorm` over the 128-channel *packed* latent: `running_mean`,
`running_var`) at encode time, and **inverts it before decoding**:

$$
z_{\text{denorm}} = z \cdot \sqrt{\operatorname{Var}+\epsilon} + \operatorname{mean},
\qquad \epsilon=10^{-4},\ \operatorname{Var}\approx 3.1\ (\Rightarrow \text{std}\approx1.76).
$$

This is applied to the 128-channel patch latent *before* `unpatchify`. Each of a
patch's four sub-pixels lives in a different one of the 128 channels and therefore
gets a different mean/var — so **skipping this de-norm mis-scales the four
sub-pixels of every $2\times2$ patch differently, producing a hard $2\times2$
checkerboard across the whole image.** That was the infamous "patch-grid" artifact;
the fix loads `bn.running_{mean,var}` from the VAE file and applies the affine per
channel before unpatchify (Appendix B, "VAE bn de-norm"). It is *structural*: more
denoise steps never removed it, which is how we knew it wasn't an undercooking
problem.

### 3.5.2 `post_quant_conv`

diffusers' `autoencoder_kl_flux2._decode` runs `z = post_quant_conv(z); decoder(z)`
— a $1\times1$ convolution ($32\to32$, a per-pixel channel remix plus bias) applied
to the latent *before* the decoder body. It is a sibling of the `decoder.` weight
prefix, not under it, so a decoder that starts at `decoder.conv_in` silently skips
it. Omitting it turned sharp images into "mush." The decoder now folds it in: the
`VAEDecoder::Config::post_quant_conv_name` names the tensor, the decoder builds a
$1\times1$ conv and applies it at the top of `forward`, and it is skipped
gracefully (identity) if the tensor is absent. The decisive diagnostic was feeding
diffusers' *own* ground-truth latent through our decoder and getting mush — proving
the gap was in the decoder, not the transformer trajectory (Chapter 25).

> **Lesson worth generalizing.** Reference pipelines hide affine/normalization
> steps in unobvious places (a sibling conv, a buried batch-norm). When porting,
> diff against the reference at the *latent* boundary, not just the final image —
> both these bugs were localized by comparing latents, not pixels.

## 3.6 The decoder, in outline

The full decoder is Chapters 19–20 (it is where the cuDNN-9 convolution story
lives). For the latent picture, here is its shape, straight from
`vae_decoder.h`, an 8× upsampler built from residual conv blocks:

```
 latent [32, H/8, W/8]
   │  (post_quant_conv 1×1, §3.5.2)
   ▼
 conv_in 32→512 (3×3)
 mid_block:   resnet512 → self-attention(512) → resnet512
 up_blocks[0]: 3× resnet512 → upsample ×2 → conv
 up_blocks[1]: 3× resnet512 → upsample ×2 → conv          ┐ three ×2 upsamples
 up_blocks[2]: resnet512→256, 2× resnet256 → upsample ×2  ┘ = 8× total
 up_blocks[3]: resnet256→128, 2× resnet128   (no upsample)
 conv_norm_out (GroupNorm) → SiLU → conv_out 128→3 (3×3)
   ▼
 pixels [3, H, W]   (then de-norm §3.5.1 happens earlier, on the patch latent)
```

Three $\times2$ upsamples give the 8×. The mid-block even contains a *self-attention
over spatial positions* (1 head, 512-dim) — the same attention machinery as the
transformer, which is why the attention kernel of Part IV is written to also serve
$D=512$ (Chapter 16). Everything else is convolution, group normalization, and
SiLU.

## 3.7 Where this lives in the code

| Concept | Code |
|---|---|
| patch packing `c·p·p+ph·p+pw` | `kernels/patchify.{h,cu}` (`patchify_bf16`/`unpatchify_bf16`) |
| latent→pixels decoder | `vae_decoder.{h,cu}` (Chapters 19–20) |
| `post_quant_conv` fold-in | `VAEDecoder::Config::post_quant_conv_name` |
| bn de-norm | applied in `generate.cu` to the 128-ch patch latent before unpatchify |
| spatial self-attention in mid-block | `vae_attn.{h,cu}` → `attention.cu` at $D=512$ |
| correctness vs diffusers latent | `tests/test_vae_decoder.cu`, `tools/diffusers_latent_dump.py` (Chapter 25) |

## 3.8 Summary and what to carry forward

- Diffusion runs in a **VAE latent space**, ~8× smaller per side with 32 channels,
  for both cost and redundancy reasons. The decoder is the only VAE part used at
  inference.
- A VAE's latent is regularized toward $\mathcal N(0,\mathbf I)$ (the ELBO's KL
  term), which is *why* the rectified-flow noise endpoint and the latents share a
  space — the diffusion in Chapter 02 transports noise to latents, not to pixels.
- FLUX stacks a **$2\times2$ patchify** on top of the VAE's 8×, giving a 16×
  pixel-to-token ratio; at 1024px that is $4096$ image tokens of 128 dims each. The
  packing order (`c·p·p+ph·p+pw`, channel-slowest) must match exactly.
- Two easily-missed transforms — the **batch-norm de-normalization** and
  **`post_quant_conv`** — sit at the latent/decoder boundary and caused the
  patch-grid and mush bugs respectively. Diff at the latent boundary when porting.

That completes Part I: we know the *objective* (rectified flow), the *loop* (Euler
on a velocity field), and the *space* (VAE latent). Part II opens the velocity
network itself — the MMDiT transformer — beginning with how images become tokens
and the two-stream architecture that lets text and image attend to each other.

---

### Exercises

1. **Token count.** Derive $\text{seq\_img}$ at 512px and 2048px from the 8× VAE
   and $p=2$ patchify. Confirm 1024px gives 4096, and relate to the $S=4608$
   attention benchmark (what are the other 512 tokens?).
2. **Patch packing.** Write out the 128-vector for one token given a latent block,
   under both `c·p·p+ph·p+pw` (correct) and the channel-fastest order, and argue
   why the wrong order produces a coherent-but-degraded image rather than noise.
3. **Why the grid is structural.** Explain why omitting the per-channel bn de-norm
   yields a fixed $2\times2$ checkerboard independent of step count, whereas an
   under-trained sampler yields blur. (Hint: which channels map to which sub-pixel?)
4. **ELBO intuition.** If you removed the KL term entirely, what would happen to
   the latent space, and why would diffusing in it (and starting from
   $\mathcal N(0,\mathbf I)$) break?

*Next: [Chapter 04 — Diffusion transformers (MMDiT)](04-mmdit.md), where the latent
tokens meet the text and the velocity network begins.*
