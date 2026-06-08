# Chapter 27 — Image Conditioning: img2img, Inpainting & Upscaling

> *Goal of this chapter:* use the encoder of Chapter 26 to condition generation on an
> existing **image** instead of pure noise. Three capabilities fall out of one idea —
> *start the denoise from a noised real latent* — controlled by a single **strength**
> dial: **img2img** (restyle a whole image), **inpainting** (regenerate only a painted
> region), and **upscaling** (a hi-res pass). All of it is latent arithmetic around the
> Chapter 2 flow-matching schedule; no model retraining, no new kernels. Anchored to
> `tools/serve.cu`.
>
> *Prerequisites:* Chapters 02 (flow matching, the schedule), 03 (latent/bn boundary),
> 23 (the txt2img loop), 26 (the encoder).

---

## 27.1 The one idea

Text-to-image (Chapter 23) starts the denoise from $\mathbf{x}_{t=1}=\boldsymbol\varepsilon\sim\mathcal N(0,\mathbf I)$
— pure noise — and integrates the flow-matching ODE down to $t=0$, a clean latent. The
velocity field the transformer predicts transports *noise* to *data* along

$$\mathbf{x}_t = (1-t)\,\mathbf{x}_0 + t\,\boldsymbol\varepsilon, \qquad t\in[1,0],$$

where $t=1$ is noise and $t=0$ is data (Chapter 2). The entire trick of image
conditioning is: **don't start at $t=1$, and don't start from random $\mathbf{x}_0$.**
Encode a real image to get its clean latent $\mathbf{x}_0$, jump to some intermediate
$t_0<1$ on the *same* interpolation, $\mathbf{x}_{t_0}=(1-t_0)\mathbf{x}_0+t_0\boldsymbol\varepsilon$,
and run only the **tail** of the schedule from $t_0$ to $0$. The result is a new image
that shares the original's structure (because the low-frequency content survived the
partial noising) but is re-rendered toward the prompt (because the model freely denoises
from $t_0$). How far back you jump — $t_0$ — *is* the strength dial.

## 27.2 From image to transformer-space latent

Conditioning on an image means turning its pixels into the exact tensor the transformer's
denoise loop expects: a 128-channel patch latent in *transformer* space. That is the
encoder of Chapter 26 plus the boundary crossings of Chapter 03, run in order
(`serve.cu`, img2img branch):

```
 image [3,res,res], model space [-1,1]
   │  VAEEncoder.forward                       (Ch 26)
   ▼
 moments [64, res/8, res/8] ── take mean = first 32 channels ──▶ z_vae [32, res/8, res/8]
   │  patchify (p=2)                           (Ch 03)
   ▼
 z_vae tokens [S, 128]   ── bn-normalize  (z − bn.mean)/√(bn.var+ε) ──▶  x0 [S, 128]
```

The last step is the **inverse** of Chapter 20's decode-side bn de-normalization (§26.5):
decode multiplies by $\sqrt{\text{var}+\epsilon}$ and adds the mean; encode subtracts the
mean and divides. After it, $\mathbf{x}_0$ is the clean latent *in the space the
transformer was trained on* — the same space a finished txt2img denoise lands in. We keep
$\mathbf{x}_0$ (and a fixed noise tensor $\boldsymbol\varepsilon$) on the host, because the
inpainting loop (§27.4) will need them every step.

## 27.3 Strength as a partial schedule

The scheduler (`FlowMatchScheduler`, Chapter 2) precomputes the $t$ values for `steps`
Euler steps, dynamically shifted, from $t_0=1$ down to $0$. For img2img we keep the *same*
schedule but enter it partway:

```
 n_run  = round(steps · strength)              # how many tail steps to run
 i_start = steps − n_run                        # skip the high-noise head
 t0      = sched.t(i_start)                      # the entry timestep
 x       = (1 − t0)·x0 + t0·ε                    # noise the latent to t0
 for i in [i_start, steps):  x += dt(i) · model(x, t_i)     # run the tail
```

A few consequences worth internalizing:

- **`strength = 1`** gives `i_start = 0`, and the dynamic-shift schedule has
  $t(0)=1$ exactly, so $\mathbf{x}=(1-1)\mathbf{x}_0+1\cdot\boldsymbol\varepsilon=\boldsymbol\varepsilon$
  — pure noise. Image conditioning at full strength **degenerates to txt2img**; the init
  image is forgotten. That is the correct, continuous limit.
- **Lower strength → fewer steps actually run** (`n_run = round(steps·strength)`). At
  `strength 0.3` with 8 steps only ~2 steps execute, which can look undercooked — so the
  UI advises raising `steps` when using low strength.
- The dial is **monotone and intuitive**: $t_0$ small ⇒ little noise added ⇒ stays near the
  original; $t_0\to1$ ⇒ near-total regeneration. Measured: a red coastal car → *blue at
  sunset* was strength 0.7 (same car, new colour/light); 0.3 was near-identical, 0.9 fully
  re-rendered.

This whole mechanism is ~20 lines around the existing denoise loop. The encoder did the
hard part.

## 27.4 Inpainting: RePaint masked denoise

Inpainting changes *only* a painted region and leaves the rest untouched. The standard
diffusion technique — **RePaint** — drops straight onto the flow-matching loop: at every
step, **overwrite the kept (unmasked) region with its own known trajectory**, and let the
model regenerate only the masked region, conditioned each step on correct surrounding
context.

Let $m_r\in[0,1]$ be a per-token mask ($1$ = regenerate, $0$ = keep). The known trajectory
of a kept token is just its noised clean latent at the current $t$, the same
interpolation as everywhere else: $\text{known}_t = (1-t)\mathbf{x}_0 + t\,\boldsymbol\varepsilon$.
The loop becomes:

```
 x = (1 − t0)·x0 + t0·ε                          # init (as img2img)
 for i in [i_start, steps):
     t = sched.t(i)
     x ← m · x  +  (1 − m) · ((1−t)·x0 + t·ε)     # lock kept region to known-at-t
     x += dt(i) · model(x, t)                      # model regenerates masked region
 x ← m · x  +  (1 − m) · x0                        # final lock: kept region = exact x0 (t=0)
```

The kept region is pinned to the original's correct noised value *before the model sees
it*, so the network always denoises the masked area against a faithful context; the masked
area evolves freely. At $t=0$ the kept tokens equal $\mathbf{x}_0$ exactly, so decoding
reproduces the original there. With a **soft** mask the blend $m\cdot x+(1-m)\cdot\text{known}$
feathers the boundary instead of hard-cutting it.

Mechanically this is a host round-trip per step — copy the latent down, blend, copy up —
which is negligible beside the transformer forward that dominates each step. We force
`strength ≥ 0.6` for inpainting so the masked region gets enough steps to actually fill.

Verified end-to-end: masking the top of a sailboat scene and prompting "stormy sky with
lightning" repainted the sky — lightning and all — while the boat and sea below stayed
pixel-faithful.

## 27.5 The mask, in token space

The mask is painted in the browser at image resolution (Chapter 28); the loop needs it at
**latent-token** resolution. The token grid is $H_P\times W_P=(res/16)^2$ — each token is a
$2\times2$ latent patch, i.e. a $16\times16$ image-pixel block (2× patch × 8× VAE). So the
pixel mask is **average-pooled** over $16\times16$ blocks:

$$m_r = \frac{1}{16^2}\sum_{(y,x)\in\text{block}(r)} \frac{\text{mask}(y,x)}{255}\in[0,1],$$

and crucially the token index must match `patchify`'s order, $r = h_p\cdot W_P + w_p$ (the
channel-slowest packing of Chapter 03 / Appendix B.4). Averaging — rather than
thresholding — gives the **soft** edge weights that feather the blend in §27.4; a token
half-covered by the brush gets $m_r\approx0.5$. `load_mask_tokens` in `serve.cu` does exactly
this pooling.

## 27.6 Upscaling: low-strength img2img at higher resolution

Upscaling needs no new mechanism — it is **img2img at low strength and a larger
resolution**, the classic "hi-res fix." Take a finished image, run it as the init at
**strength ≈ 0.35** but at **2× the resolution**, reusing the source's prompt and seed.
The low strength means the structure is preserved (only the high-frequency detail is
re-synthesised at the higher native resolution); the larger grid means real new detail,
not interpolation.

The resolution ceiling here taught a lesson. Direct upscaling worked to **1536px** but
*failed at 2048* — and the failure was inside the VAE encoder's mid-attention at $S=65536$
tokens, the `gridDim.y` overflow of Appendix B.15, which also blocked native 2048
generation. Once that one-line grid fix landed, 2048 (true 2K) worked everywhere; the
encoder golden table (§26.7) gained its 2048 row, and the upscale cap rose to 2048. 2K is
*functional but slow* (~80–90 s: the denoise is $O(S^2)$ at sequence length ~16 896, plus
~15 s each for VAE encode and decode) — a correctness win, with speed left as future work.

## 27.7 Where this lives in the code

| Concept | Code |
|---|---|
| encode → transformer-space `x0` | `serve.cu` img2img branch (encoder + patchify + bn) |
| strength → partial schedule | `serve.cu` (`n_run`, `i_start`, `t0`) |
| inpainting RePaint loop | `serve.cu` denoise loop (per-step lock) + final lock |
| pixel mask → token weights | `load_mask_tokens` (`serve.cu`) |
| upscaling | `webui/app.py` `/upscale` (strength 0.35, 2× res) — Chapter 28 |
| the encoder | `vae_encoder.{h,cu}` (Chapter 26) |
| the schedule | `sampler.h` `FlowMatchScheduler` (Chapter 2) |

## 27.8 Summary and what to carry forward

- All three features are **one idea**: start the flow-matching denoise from a *noised real
  latent* rather than pure noise, and run only the tail of the schedule.
- **Encode → mean → patchify → bn-normalize** turns an image into a transformer-space
  $\mathbf{x}_0$ (Chapter 26 + the inverse bn of Chapter 20).
- **Strength** is a partial schedule: `i_start = steps − round(steps·strength)`,
  $\mathbf{x}_{t_0}=(1-t_0)\mathbf{x}_0+t_0\boldsymbol\varepsilon$. Strength 1 = txt2img;
  low strength runs fewer steps (raise `steps` to compensate).
- **Inpainting** is RePaint: each step, lock the kept region to its known noised trajectory
  $(1-t)\mathbf{x}_0+t\boldsymbol\varepsilon$ and let the model fill the masked region in
  context; a soft, token-pooled mask feathers the seam.
- **Upscaling** is low-strength 2× img2img. True 2K waited on the `gridDim.y` fix
  (Appendix B.15) and is correct-but-slow.

These are pure inference-time techniques — the model never knew about img2img or
inpainting; the flow-matching geometry gave them for free. The final chapter productionizes
all of it.

---

### Exercises

1. **The strength limit.** Show algebraically that strength $=1$ reduces img2img to
   txt2img, using $t(0)=1$ of the dynamic-shift schedule. What happens at strength $=0$,
   and why does the implementation clamp `n_run ≥ 1`?
2. **RePaint vs naive masking.** A tempting shortcut is to denoise the whole latent freely
   and paste the original back only at the end. Explain what that loses versus per-step
   locking, in terms of the context the model attends to.
3. **Soft vs hard masks.** Derive the visible difference at a region boundary between a
   thresholded ($m\in\{0,1\}$) and an average-pooled ($m\in[0,1]$) mask, and relate it to
   the $16\times16$ pixel→token block.
4. **Upscale strength.** Why does upscaling want *low* strength (~0.35) rather than high?
   What would strength 0.9 do to the original composition, and why is that wrong for an
   upscaler?

*Next: [Chapter 28 — Serving: the persistent worker & live preview](28-serving.md).*
