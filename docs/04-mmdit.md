# Chapter 04 — Diffusion Transformers (MMDiT)

> *Goal of this chapter:* introduce the velocity network itself. We explain why a
> *transformer* replaced the U-Net for diffusion, what "tokens" are for an image
> model, and the **MMDiT** (Multi-Modal Diffusion Transformer) two-stream design
> that lets text and image attend to each other. This chapter is the map of Part
> II; Chapters 05–10 zoom into each component. We anchor everything to
> `src/backend/cuda/flux_transformer.{h,cu}` and the config drawn from
> `transformer/config.json`.
>
> *Prerequisites:* Part I; the transformer (attention + MLP) at a conceptual level
> — Chapter 16 derives attention rigorously, but you need only "tokens attend to
> tokens" here.

---

## 4.1 From U-Net to transformer

The original latent-diffusion models used a **U-Net**: a convolutional encoder/
decoder with skip connections, denoising the latent at multiple spatial scales.
U-Nets bake in locality (convolutions) and multi-scale structure (down/up
sampling). They work, but they have two frictions for text-to-image: injecting
text conditioning means bolting cross-attention onto a fundamentally convolutional
backbone, and convolution's locality limits long-range interactions.

The **Diffusion Transformer (DiT)** (Peebles & Xie, 2022) threw out the U-Net and
denoised the latent with a plain transformer over image patches. It scales
cleanly, treats all positions uniformly, and — crucially for text-to-image — lets
text and image live in the *same* attention, as just more tokens. FLUX is a
descendant of this line, specifically of the **MMDiT** design introduced with
Stable Diffusion 3.

> The velocity field $v_\theta(x,t,c)$ of Chapter 02 *is* this transformer. Its
> input is the noisy latent (as tokens) plus the timestep $t$ plus the text
> conditioning $c$; its output is the velocity, same shape as the latent. Each of
> the 4 denoise steps is one forward pass of this network.

## 4.2 Tokens: the common currency

A transformer operates on a **sequence of token vectors** and learns interactions
between them via attention. FLUX has two kinds of tokens, and the whole
architecture is about how they mix.

- **Image tokens.** From Chapter 03, the patchified latent is a sequence of
  $S_{\text{img}}$ tokens (e.g. 4096 at 1024px), each a 128-dim vector. An
  *embedder* projects them to the model's hidden width.
- **Text tokens.** The Qwen3 encoder (Chapter 05) produces $S_{\text{txt}}=512$
  conditioning vectors, each 12288-dim. A *context embedder* projects them to the
  same hidden width.

Once both are projected to the common hidden dimension $d=4096$, they are
interchangeable as far as attention is concerned — image token 17 can attend to
text token 4 and vice versa. This is what "multi-modal" means here: one attention
operation over a *joint* set of text and image tokens.

## 4.3 The two-stream idea

A naïve DiT would simply concatenate text and image tokens and run identical
transformer blocks over the union. MMDiT refines this: for the early, most
important layers it keeps **two separate streams** — text and image each get their
*own* projection weights, their own MLP, their own normalization — but they
**attend jointly** (the attention is computed over the concatenation, so the
streams exchange information), and then split back apart. Each modality gets to
develop its own representation while still being able to look at the other.

```
            text stream                     image stream
            (own weights)                   (own weights)
                │                                │
          norm + modulate                  norm + modulate
                │                                │
            Q_t,K_t,V_t                     Q_i,K_i,V_i
                └──────────────┬───────────────┘
                               ▼
                JOINT ATTENTION over [txt ‖ img]
                  (every token attends to every token,
                   across both modalities)
                               │
                ┌──────────────┴───────────────┐
            text out                        image out
                │                                │
            own MLP                         own MLP
                ▼                                ▼
            text stream'                    image stream'
```

This is a **double-stream block** (Chapter 08). FLUX.2-klein uses **8** of them.

After the double-stream stage, the two streams are **concatenated into one
sequence** and processed by a stack of **single-stream blocks** — ordinary
transformer blocks over the combined `[txt ‖ img]` sequence, sharing one set of
weights for both modalities. FLUX.2-klein uses **24** of these. The intuition: the
early layers benefit from modality-specific processing; once text and image are
"aligned," a unified stream is more parameter-efficient. Single-stream blocks also
use a **fused** QKV+MLP layout that we will see is a nice bandwidth win (Chapter
09).

After the single-stream stack, the **image** tokens (the tail of the combined
sequence) are pulled back out and projected to the velocity. The text tokens have
done their job — conditioning the image — and are discarded.

## 4.4 The full topology

Here is the complete network, adapted from the diagram that ships in
`flux_transformer.h`:

```
 img_latent [B·S_img, 128]                         txt_emb [B·S_txt, 12288]
      │ ImageEmbedder (Linear 128→4096)                 │ ContextEmbedder (Linear 12288→4096)
      ▼                                                 ▼
 img_hidden [B·S_img, 4096]                        txt_hidden [B·S_txt, 4096]
      │                                                 │
      │   timestep t ──▶ sinusoid(256) ──▶ ModulationMLP ──▶ 17 modulation vectors
      │                                                 │       (Chapter 07; shared across blocks)
      ├──────────────  × 8  DoubleStreamBlock  ─────────┤   (Chapter 08; joint attention)
      │                                                 │
      │        concat → combined [B·(S_txt+S_img), 4096]│
      │                        │                        │
      │                 × 24  SingleStreamBlock          │   (Chapter 09; fused QKV+MLP)
      │                        │                        │
      │        take tail S_img rows (the image tokens)  │
      ▼                        │
 img_hidden_final ◄────────────┘
      │ FinalProjection (RMSNorm → modulate → Linear 4096→128)
      ▼
 velocity [B·S_img, 128]      ← this is v_θ; fed to the Euler update (Chapter 02)
```

Every arrow is a real component with its own chapter. Note the **concat/split**
transitions (`concat_two_streams`, `take_tail` kernels) at the double→single
boundary and at the output — these are pure data-movement kernels, no math.

## 4.5 The dimensions, exactly

From `flux_transformer.h::Config` and `transformer/config.json`:

| Parameter | Value | Meaning |
|---|---|---|
| `num_double_blocks` | **8** | two-stream blocks |
| `num_single_blocks` | **24** | unified blocks → 32 layers total |
| `n_heads` | **32** | attention heads |
| `head_dim` | **128** | per-head dimension → this is the $D=128$ of Part IV |
| hidden $d$ | **4096** | $=32\times128$ |
| `ffn_dim` | **12288** | MLP inner width ($\text{mlp\_ratio}=3.0$) |
| `t5_dim` / `joint_attention_dim` | **12288** | context-embedder input (3 × Qwen3's 4096; Chapter 05) |
| `in_channels` | **128** | image-token channel dim (Chapter 03) |
| `time_dim` | **256** | timestep embedding width |
| `axes_dims_rope` | **[32,32,32,32]** | **4-axis** RoPE (Chapter 06) |
| `rope_theta` | **2000** | RoPE base — *not* the usual 10000 |
| RMSNorm `eps` | **1e-6** | |
| `guidance_embeds` | **false** | distilled klein: no classifier-free guidance (Chapter 02) |

A few of these deserve a flag now, expanded later:

- **`t5_dim = 12288` is a misnomer.** It is not T5. FLUX.2-klein's text encoder is
  **Qwen3-8B**, and 12288 = 3 × Qwen3's hidden size 4096 — the context embedder
  consumes a *concatenation of three Qwen3 layer hidden states*. Chapter 05.
- **`rope_theta = 2000`** (not 10000) and **4-axis** RoPE are FLUX specifics that
  the position-encoding kernels must implement exactly. Chapter 06.
- **The single-stream blocks fuse their linears.** The QKV projection and the MLP
  in/out are stored as single large matrices (`to_qkv_mlp_proj` is `[36864, 4096]`
  = 9×4096; `to_out` is `[4096, 16384]` = 4×4096). This is deliberate — one big
  GEMM moves weights more efficiently than several small ones on a bandwidth-bound
  machine. Chapter 09.

## 4.6 A structural surprise: shared modulation weights

A timestep produces **modulation** signals — the per-layer scale/shift/gate that
tell each block how hard to act at this noise level (AdaLN, Chapter 07). In many
DiTs each block has its own modulation projection. In FLUX.2-klein the modulation
projections are **shared across all blocks of a type**: there is one
`double_stream_modulation_img.linear`, one `..._txt.linear`, one
`single_stream_modulation.linear`, each emitting all the vectors that *every* block
of that type consumes. (Confirmed by the tensor router: 8 double blocks × 16
tensors + 24 single blocks × 4 tensors + 9 globals = 233 tensors, with no
per-block modulation linears — Chapter 14.) This is a distilled-model economy: it
cuts parameters with little quality loss. The `ModulationMLP` (Chapter 07) computes
all 17 modulation vectors once per forward pass from the single timestep embedding.

## 4.7 Construction and cost

`FluxTransformer`'s constructor builds 124 `Linear` instances (8 double-blocks ×
12 + 24 single-blocks × 2 + embedders + modulation + final projection), each
repacking its quantized weights into the tensor-core-friendly layout (Chapter 13).
On the GB10 this is ~5.3 s one-time (the "transformer build" line in Chapter 00),
and the forward pass over all 32 layers is the per-step cost in the denoise loop.
The workspace — all the intermediate token buffers — is a few GiB at 1024px, which
unified memory absorbs without complaint (Chapter 11).

The whole thing is verified end to end by `tests/test_flux_transformer.cu` (runs a
forward on real weights, checks finiteness and magnitudes) and, more stringently,
by `tools/cmp_transformer.cu` against a diffusers reference at cos-similarity 0.963
in NVFP4 / 0.999 in MXFP8 (Chapter 25).

## 4.8 What each remaining Part-II chapter covers

- **05 — Text conditioning with Qwen3.** Where the `[512 × 12288]` comes from, the
  three-layer concatenation, and the context embedder.
- **06 — Rotary position embeddings.** RoPE, the 4-axis variant for 2D images, and
  why $\theta=2000$.
- **07 — Modulation / AdaLN.** The timestep → 17 modulation vectors MLP and how a
  block consumes scale/shift/gate.
- **08 — Double-stream blocks.** The two-stream, joint-attention block in full,
  including the per-stream QK-norm.
- **09 — Single-stream blocks.** The fused QKV+MLP block and the split/concat
  plumbing.
- **10 — Assembling the transformer.** Embedders, the final projection, the
  stream transitions, and the top-level forward.

## 4.9 Summary and what to carry forward

- The velocity network is a **transformer over tokens**, not a U-Net; image
  patches and text vectors are both just tokens once projected to hidden width
  4096.
- **MMDiT** runs **8 double-stream blocks** (separate text/image weights, *joint*
  attention) then **24 single-stream blocks** (unified, fused linears), then pulls
  the image tokens out and projects to the velocity. 32 layers total.
- Key FLUX specifics to remember: text encoder is **Qwen3** (so `t5_dim`=12288 is
  3×4096), **4-axis RoPE** with $\theta=2000$, **fused** single-stream linears, and
  **shared modulation** weights (a distilled-model economy).
- Everything maps to `flux_transformer.{h,cu}` and is the function we spend Parts
  III–IV making fast.

Chapter 05 starts at the left edge of the diagram: how a prompt becomes the
`[512 × 12288]` conditioning that enters the context embedder.

---

### Exercises

1. **Token budget.** At 1024px, $S_{\text{img}}=4096$ and $S_{\text{txt}}=512$.
   Through which blocks does the sequence have length 4096, and through which 4608?
   Relate to the attention benchmark shapes in Part IV.
2. **Two streams vs one.** Give one reason early layers benefit from
   modality-specific weights and one reason later layers don't need them. What does
   sharing weights in the single-stream stage save on a bandwidth-bound machine?
3. **Parameter accounting.** Using the dimensions table, estimate the parameter
   count of one double-stream block (12 linears) and one single-stream block (2
   fused linears), and sanity-check that 8+24 of them approach ~9B.
4. **Shared modulation.** Why can modulation weights be shared across blocks when
   attention/MLP weights cannot? (Hint: what does modulation depend on vs what
   attention depends on?)

*Next: [Chapter 05 — Text conditioning with Qwen3](05-text-conditioning.md).*
