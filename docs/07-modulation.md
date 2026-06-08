# Chapter 07 — Modulation / AdaLN

> *Goal of this chapter:* explain how the **timestep** (and, in non-distilled
> models, guidance) steers every layer of the transformer through **adaptive layer
> normalization (AdaLN)**. We derive what scale/shift/gate do, trace the
> `ModulationMLP` that turns a 256-dim timestep embedding into **17** modulation
> vectors, and dissect the `modulate` and `gated_residual` kernels. We end on the
> **scale/shift ordering bug** — the single most important bug in this project's
> history, and a perfect lesson in convention-matching. Anchored to
> `modulation_mlp.{h,cu}` and `kernels/modulation.{h,cu}`.
>
> *Prerequisites:* Chapters 02 (timestep), 04 (block topology); layer norm.

---

## 7.1 The job: tell each layer "how noisy is it right now"

The transformer is called once per denoise step, at different noise levels $t$
(Chapter 02). The *same weights* must behave differently at $t\approx1$ (almost
pure noise — commit broad structure) than at $t\approx0$ (almost clean — refine
fine detail). The network needs a per-step "knob" that re-tunes every layer's
behavior based on $t$.

That knob is **modulation**: from the timestep embedding, compute a small set of
vectors that *scale, shift, and gate* the activations inside each block. This is
the mechanism by which a single set of weights implements a whole trajectory of
behaviors across noise levels. In conditional models the same mechanism also folds
in class/guidance signals; in distilled klein the only input is the timestep
(`guidance_embeds=false`, Chapter 02).

## 7.2 AdaLN: adaptive layer normalization

Standard layer norm normalizes a token vector to zero mean / unit variance, then
applies a *learned, fixed* affine $\gamma x + \beta$. **Adaptive** layer norm
(AdaLN, from the DiT paper) replaces the fixed $\gamma,\beta$ with values
*predicted from a conditioning signal* — here, the timestep. A block's normalized
activations are modulated as

$$
\text{modulate}(x;\,\text{scale},\text{shift}) = (1+\text{scale})\odot \hat x + \text{shift},
$$

where $\hat x$ is the normalized activation and $\text{scale},\text{shift}$ are
per-channel vectors predicted from $t$. The "$1+$" means *zero* modulation is the
identity (scale 0, shift 0 → pass-through), which makes initialization and
zero-gate behavior clean.

AdaLN-**Zero** (what DiT/FLUX use) adds a third predicted vector, a **gate**, that
multiplies the block's *output* before it is added back to the residual stream:

$$
x \leftarrow x + \text{gate}\odot \text{SubLayer}\big(\text{modulate}(\text{norm}(x);\,\text{scale},\text{shift})\big).
$$

With $\text{gate}=0$ the sub-layer contributes nothing and the block is the
identity — so at initialization every block is a no-op and the network starts as a
clean residual highway, which stabilizes training. Each attention or MLP sub-layer
gets its **own** (scale, shift, gate) triple.

```
 residual x ──┬─────────────────────────────────────────────▶ (+) ──▶ x'
              │                                                 ▲
              ▼                                                 │
          LayerNorm/RMSNorm  →  modulate(scale,shift)  →  SubLayer  →  ⊙ gate
                                  (1+scale)·x̂ + shift     (attn/MLP)
```

This is the structure of *every* block in FLUX. A block has two sub-layers
(attention, then MLP), so it consumes **two** (scale, shift, gate) triples = **6
modulation vectors** per stream.

## 7.3 Counting to 17

How many modulation vectors does one forward pass need? Recall (Chapter 04) the
modulation projections are **shared across blocks of a type**, so the
`ModulationMLP` emits, once per forward pass, the full set consumed by every block:

| Consumer | triples | vectors | weight shape |
|---|---|---|---|
| Double-stream **image** (attn + MLP) | 2 | **6** | `double_stream_modulation_img.linear` `[24576, 4096]` = 6×4096 |
| Double-stream **text** (attn + MLP) | 2 | **6** | `double_stream_modulation_txt.linear` `[24576, 4096]` |
| Single-stream (one sub-layer pair) | 1 | **3** | `single_stream_modulation.linear` `[12288, 4096]` = 3×4096 |
| Final norm-out (AdaLayerNormContinuous) | — | **2** | `norm_out` (scale, shift; **no gate**) |

That is $6 + 6 + 3 + 2 = \mathbf{17}$ modulation vectors. The double/single counts
are *per block type* (shared across all 8 / all 24 blocks); the final projection
(Chapter 10) uses a 2-vector (scale, shift only) modulation. These 17 vectors are
exactly the fields of `DoubleStreamBlock::Modulation`, `SingleStreamBlock::Modulation`,
and the final norm in the code.

## 7.4 The ModulationMLP

`ModulationMLP` (`modulation_mlp.{h,cu}`) computes all 17 vectors from the 256-dim
timestep embedding:

```
 timestep_emb [256]                       (from sampler.h, cat(cos,sin), t×1000)
      │ linear_1 → [4096]
      │ SiLU
      │ linear_2 → [4096]
      │ SiLU
      ├── proj_double_img → [24576]  = 6 × 4096   (img: scale/shift/gate × {attn, mlp})
      ├── proj_double_txt → [24576]  = 6 × 4096   (txt: same)
      ├── proj_single     → [12288]  = 3 × 4096   (single-stream triple)
      └── proj_norm_out   → [8192]   = 2 × 4096   (final scale, shift)
```

Six NVFP4/MXFP8 `Linear`s and a `silu_inplace_bf16` kernel. The two-layer MLP with
SiLU gives the timestep a small nonlinear network before it fans out into the four
projections. Each projection's output is sliced into its constituent 4096-vectors,
which the blocks then index. On real weights this runs once per forward pass and
its 17 outputs are tiny perturbations (trained modulation parameters have small
magnitudes) — `max≈1.34`, mean-of-means ≈0.07 — exactly what you expect for AdaLN
deltas around the identity.

The structatic output is a struct of pointers into the projection buffers, one per
modulation vector, matching the fields each block reads. (Implementation detail:
the MLP runs at an internal batch of 128 rows to satisfy the `Linear` GEMM's
$M\%128=0$ constraint — Chapter 13 — and the real vectors live in row 0.)

## 7.5 The two kernels: `modulate` and `gated_residual`

The actual arithmetic is two tiny elementwise kernels (`kernels/modulation.{h,cu}`):

```c++
// modulate: (1 + scale) * x + shift,  scale/shift are [hidden] broadcast over rows
modulate(x, scale, shift):   x[r,c] = (1 + scale[c]) * x[r,c] + shift[c]

// gated_residual: y += gate * delta,  gate is [hidden] broadcast over rows
gated_residual(y, delta, gate):   y[r,c] += gate[c] * delta[r,c]
```

`scale`, `shift`, `gate` are BF16 `[hidden=4096]` vectors broadcast across all
token rows. Both kernels are memory-bound and trivial; the interesting part is not
the arithmetic but getting the *operands* right — which is the subject of §7.6.
(`modulate` reaches ~0.004 max abs error vs an FP32 reference — pure BF16 rounding.)

## 7.6 The bug: scale/shift order (a case study in convention-matching)

This is the most instructive bug in the project, so we dwell on it.

A modulation projection emits a block of `3 × 4096` (or `2 × 4096`) numbers that
must be *sliced* into the named vectors. The question is the **order** of the
slices. The original implementation sliced every `Flux2Modulation` output as
`[scale, shift, gate]`. But diffusers' `Flux2Modulation.split` actually emits
**`[shift, scale, gate]`** — the destructuring in `transformer_flux2.py` is
`(shift_msa, scale_msa, gate_msa)`, shift **first**.

With the wrong order, every block computed

$$
(1+\text{shift})\odot \hat x + \text{scale} \quad\text{instead of}\quad (1+\text{scale})\odot \hat x + \text{shift},
$$

i.e. it swapped the multiplicative and additive roles of two vectors in **all 8
double + 24 single blocks**. The effect was subtle and devastating: the image came
out *recognizable but barely prompt-dependent*. Quantitatively, before the fix a
cat prompt vs a truck prompt differed by 0.12% in pixels while two random seeds
differed by ~10% — the latent dominated the prompt 40-to-1. AdaLN is precisely the
channel through which conditioning modulates the network, and corrupting it
throttled the text path to a whisper.

The fix swapped the slice order to `[shift, scale, gate]` for the
img/txt/single modulations. **Effect:** prompt influence (cat vs truck) jumped from
0.12% to **12.5%** — about 100× stronger — and became *dominant* over the latent.
That one ordering change is what turned the pipeline from "produces an image that
ignores the prompt" into "produces the requested image."

**The twist:** `norm_out` is **not** a `Flux2Modulation`. It goes through
`AdaLayerNormContinuous`, whose convention *is* `scale, shift = chunk(emb, 2)` —
**scale first**. So the final projection keeps scale-first while the block
modulations are shift-first. Two different conventions in the same model, and you
have to honor both. (Appendix B documents the full saga; Chapter 25 has the
isolation method that found it.)

> **The general lesson.** When porting from a reference, the *order* of packed
> sub-tensors is a silent contract. A swap of two vectors will not crash, will not
> NaN, and will often produce *plausible* output — which is far more dangerous than
> an obvious failure. The only defense is to diff intermediate tensors against the
> reference (Chapter 25), not just eyeball the final image.

## 7.7 How a block consumes its modulation

To close the loop, here is how one double-stream image sub-layer (say attention)
uses its triple, foreshadowing Chapter 08:

```
 x_img ──┬───────────────────────────────────────────────▶ (+) ──▶ x_img'
         ▼                                                   ▲
   RMSNorm(ones)                                             │
         │ modulate(scale_attn_img, shift_attn_img)          │
         ▼                                                   │
   Q/K/V projections → RoPE → joint attention → out proj     │
         │                                                   │
         └────────── ⊙ gate_attn_img ───────────────────────┘
```

The same pattern repeats for the image MLP sub-layer (with `scale_mlp_img,
shift_mlp_img, gate_mlp_img`), and symmetrically for the text stream — six vectors
per stream, all sourced from the shared `ModulationMLP` output.

## 7.8 Where this lives in the code

| Concept | Code |
|---|---|
| timestep → 17 vectors | `modulation_mlp.{h,cu}` (`ModulationMLP`), `silu_inplace_bf16` |
| `(1+scale)·x+shift` | `kernels/modulation.{h,cu}` `modulate` |
| `y += gate·delta` | `kernels/modulation.{h,cu}` `gated_residual` |
| slice order (shift,scale,gate) | `modulation_mlp.cu` (the fix) |
| norm_out scale-first | final projection (Chapter 10) |
| tests | `tests/test_modulation.cu`, `tests/test_modulation_mlp.cu` |

## 7.9 Summary and what to carry forward

- The **timestep** steers every layer via **AdaLN**: predict per-channel
  **scale, shift, gate** and apply $(1+\text{scale})\hat x+\text{shift}$ then gate
  the sub-layer output into the residual. Zero modulation = identity.
- One forward pass needs **17** modulation vectors (6 img + 6 txt double, 3 single,
  2 norm-out), produced once by the **`ModulationMLP`** and **shared across
  blocks**.
- The arithmetic is two trivial broadcast kernels (`modulate`, `gated_residual`);
  the difficulty is **operand order**.
- The **scale/shift swap** bug throttled prompt conditioning ~100×; fixing the
  slice order to `[shift, scale, gate]` (while keeping `norm_out` scale-first) is
  what made the model follow prompts. *Diff intermediates, not just images.*

With position (Chapter 06) and modulation (this chapter) in hand, we have every
input a block needs. Chapters 08–09 assemble the two block types.

---

### Exercises

1. **Why "$1+$scale".** Show that $(1+\text{scale})\hat x+\text{shift}$ with
   scale=shift=0 is the identity, and explain why that is a desirable
   initialization for AdaLN-Zero blocks.
2. **Gate as a no-op switch.** Argue that gate=0 makes a block the identity, and
   why a network of identity-initialized residual blocks is easy to train.
3. **The 17.** Reconstruct the count $6+6+3+2=17$ from the block topology of
   Chapter 04, and explain why double has 6-per-stream but single has only 3.
4. **The swap, quantified.** Given that AdaLN is the only conditioning channel,
   explain mechanistically why swapping scale↔shift would *attenuate* prompt
   dependence rather than, say, add noise. Why did the image stay coherent?

*Next: [Chapter 08 — Double-stream blocks](08-double-stream.md).*
