# Chapter 10 — Assembling the Transformer

> *Goal of this chapter:* connect every Part-II component into the complete
> velocity network — the embedders that bring image and text into hidden space, the
> modulation MLP that feeds all blocks, the 8 double + 24 single block stack with
> its stream transitions, and the final projection back to the latent. We trace
> the top-level `FluxTransformer::forward`, the construction cost, and the
> end-to-end validation. This closes the architecture; Part III makes it fast.
> Anchored to `flux_transformer.{h,cu}` and `embeddings.{h,cu}`.
>
> *Prerequisites:* Chapters 04–09.

---

## 10.1 The three embedders

The transformer's hidden width is 4096, but its inputs and outputs are not. Three
small `Linear`-based modules bridge the boundaries (`embeddings.{h,cu}`):

- **ImageEmbedder** — `x_embedder`, a `Linear` `128 → 4096`. Projects each
  patchified image token (128 dims, Chapter 03) into hidden space. On real weights
  its output has `mean ≈ 0.13` — sane.
- **ContextEmbedder** — `context_embedder`, a `Linear` `12288 → 4096`. Projects each
  Qwen3 conditioning token (Chapter 05) into hidden space.
- **FinalProjection** — the output bridge (§10.4): RMSNorm → AdaLN-modulate →
  `proj_out` `Linear` `4096 → 128`. Maps the processed image tokens back to the
  latent channel dim so the result is a velocity in latent space.

These are the left and right edges of the Chapter 04 diagram. Everything between
runs at hidden width 4096.

## 10.2 The top-level objects

`FluxTransformer` owns, from `flux_transformer.h`:

```
 1 × ImageEmbedder          (128 → 4096)
 1 × ContextEmbedder        (12288 → 4096)
 1 × ModulationMLP          (timestep 256 → 17 modulation vectors; Chapter 07)
 8 × DoubleStreamBlock      (Chapter 08)
24 × SingleStreamBlock      (Chapter 09)
 1 × FinalProjection        (4096 → 128)
 2 × RoPE table sets         (double-stream lengths, single-stream length; Chapter 06)
```

The two RoPE table sets are built once per forward (they depend on $S_{\text{img}}$,
$S_{\text{txt}}$ and the patch grid `H_patches × W_patches`): an image table
($(0,h,w,0)$), a text table ($(0,0,0,\ell)$), and the combined single-stream table
(text-first). `Config` carries the shapes (`seq_img`, `seq_txt`, `H_patches`,
`W_patches`, …) and the `precision` (NVFP4 or MXFP8, Chapter 12).

## 10.3 The forward pass, end to end

From the diagram that ships in `flux_transformer.h`:

```
 img_latent [B·S_img, 128]                       txt_emb [B·S_txt, 12288]
      │ ImageEmbedder                                  │ ContextEmbedder
      ▼                                                ▼
 img_hidden [B·S_img, 4096]                       txt_hidden [B·S_txt, 4096]
      │                                                │
   timestep ──▶ sinusoid[256] ──▶ ModulationMLP ──▶ 17 modulation vectors
      │                                                │
      ├──────────  × 8 DoubleStreamBlock  ─────────────┤   (joint attention; both updated in place)
      │                                                │
      │   concat_two_streams:  combined = [txt_hidden ‖ img_hidden]   [B·(S_txt+S_img), 4096]
      │                          │
      │                   × 24 SingleStreamBlock         (combined sequence, in place)
      │                          │
      │   take_tail:  img_hidden_final = combined[S_txt:]   (drop text; keep image tail)
      ▼                          │
 img_hidden_final ◄──────────────┘
      │ FinalProjection  (RMSNorm(ones) → modulate(norm_out scale, shift) → proj_out 4096→128)
      ▼
 velocity [B·S_img, 128]
```

Two pure data-movement kernels handle the stream transitions:

- **`concat_two_streams_bf16`** — at the double→single boundary, builds the combined
  sequence by placing text tokens first, then image tokens. (Order matters: it must
  match the combined RoPE table and the `take_tail` split.)
- **`take_tail_bf16`** — after the single-stream stack, extracts the last
  $S_{\text{img}}$ rows (the image tokens). The text tokens have done their
  conditioning job and are discarded.

No new math lives in these — they are reshapes/copies — but their ordering is a
contract (text-first) that the RoPE tables and the final split all depend on.

## 10.4 The final projection (and its scale-first modulation)

`FinalProjection` turns the processed image tokens into the velocity:

```
 x → RMSNorm(x, ones)                       # no-affine
   → modulate(scale_out, shift_out)         # AdaLN, but scale-FIRST (see below)
   → proj_out: Linear 4096 → 128
```

This is `AdaLayerNormContinuous` in diffusers, and — as flagged in Chapter 07 — its
modulation convention is **scale-first** (`scale, shift = chunk(emb, 2)`), unlike
the *shift-first* `Flux2Modulation` of the blocks. There is no gate here (only 2
vectors, the `norm_out` pair from the ModulationMLP). Getting this one module's
order *opposite* to the blocks' is required for correctness — a small but real
trap. The output `[B·S_img, 128]` is the velocity $v_\theta$ that the sampler's
Euler step consumes (Chapter 02). On real weights `FinalProjection` output has
`mean ≈ 0.96` — sane.

## 10.5 What it costs to build and run

- **Construction (~5.3 s on GB10).** The constructor builds **124 `Linear`
  instances** — 8 double × 12 + 24 single × 2 + 6 (embedders, modulation, proj_out)
  — and each repacks its quantized weights from the on-disk layout into the
  CUTLASS tensor-core layout (Chapter 13). With on-disk MXFP8 this is 5.3 s; the
  earlier construction-time quantization path took ~76 s (Chapter 12). All weights
  come from a `TensorRouter` over the F2K shards (Chapter 14).
- **Workspace.** All the intermediate token buffers — `img_hidden`, `txt_hidden`,
  the combined sequence, per-block Q/K/V/MLP scratch — total a few GiB at 1024px
  (e.g. ~2.5 GiB transformer + VAE separately), which unified memory absorbs
  (Chapter 11). The workspace is sized to the max over blocks and reused, since each
  block's intermediates die at its boundary.
- **Forward time.** All 32 layers + embedders + modulation + final projection is
  the per-step cost in the denoise loop (~1.6 s/step at 1024px FP8). Multiply by 4
  steps for the ~6.5 s denoise of Chapter 00. The dominant sub-costs are the GEMMs
  (Part III) and the attention (Part IV).

## 10.6 Proving the whole thing correct

Two levels of test, both essential:

- **Wiring (synthetic):** `tests/test_flux_transformer.cu` and the per-block
  zero-gate identity tests (Chapters 08–09) prove the residual highway and every
  connection. The denoise loop test (`tests/test_denoise_loop.cu`) runs the full
  4-step schedule on real weights and checks the latent stays finite and
  well-scaled across steps.
- **Numerical (vs reference):** `tools/cmp_transformer.cu` compares a *single*
  transformer forward, on identical inputs (packed latent, prompt embeds,
  $\sigma=1$, 16×16 grid), against a diffusers reference produced by
  `tools/diffusers_block_dump.py`. Result: **cos-similarity 0.963 in NVFP4, 0.999
  in MXFP8** — the NVFP4 error is purely directional and consistent with FP4
  quantization noise (Chapter 12), not a structural bug. This test is what proved
  the architecture port *correct* (no RoPE/attention/wiring error) and localized
  the residual softness to quantization. (Chapter 25 is the full validation story.)

## 10.7 The complete picture (Part II in one diagram)

```
 PROMPT ─▶ tokenizer ─▶ Qwen3 (capture {8,17,26}, concat) ─▶ [512×12288]
                                                                  │ ContextEmbedder
 noise latent ─▶ patchify ─▶ [S_img×128] ─▶ ImageEmbedder ─▶ [S_img×4096]
                                                                  │
 timestep ─▶ sinusoid ─▶ ModulationMLP ─▶ 17 mod vectors ─────────┤
                                                                  │
   ┌──────── 8× DoubleStreamBlock (per-stream weights, JOINT attention,
   │              QK-norm, 4-axis RoPE both streams, SwiGLU MLP) ──────────┐
   │                          concat [txt‖img]                            │
   │         24× SingleStreamBlock (shared weights, parallel attn+MLP,    │
   │              fused QKV+MLP GEMM, fused out GEMM) ─────────────────────┘
   │                          take tail (image tokens)
   └─▶ FinalProjection (scale-first AdaLN) ─▶ velocity [S_img×128]
                                                  │ Euler step (ch 02)
                                                  ▼  ×4
                              clean latent ─▶ unpatchify ─▶ VAE decode ─▶ PNG
```

That is the entire FLUX.2-klein velocity network and its place in the pipeline.
Everything in it has now been derived (Part I) and described (Part II). The
remaining parts answer the engineering question: **how do you evaluate this
function fast on a GB10?**

## 10.8 Summary and what to carry forward

- The transformer = **embedders** (image 128→4096, context 12288→4096) + a
  **ModulationMLP** + **8 double + 24 single** blocks + a **FinalProjection**
  (4096→128), with two **RoPE table sets** rebuilt per forward.
- **Stream transitions** are pure data movement: `concat_two_streams` (text-first)
  at the double→single boundary, `take_tail` to recover image tokens at the end —
  but their ordering is a contract.
- The **FinalProjection uses scale-first** AdaLN (opposite the blocks) and no gate.
- Built once (~5.3 s, 124 linears repacked), run per step; validated by zero-gate
  wiring tests and a **cos-0.963/0.999** comparison to diffusers.

Part II is complete: you can now explain every tensor from prompt to velocity. Part
III turns to the machine — the Blackwell roofline, quantization, CUTLASS GEMM, the
weight format, and the kernel library — i.e. how all of this is made to run in ~1.6
seconds per step.

---

### Exercises

1. **Trace a token.** Follow one image token from `img_latent[128]` to
   `velocity[128]`, naming every module and the width at each stage.
2. **Transition contracts.** `concat_two_streams` places text first. List
   everything downstream that depends on this ordering (RoPE table, take_tail, …).
3. **Two AdaLN conventions.** Explain why the blocks are shift-first but
   `FinalProjection` is scale-first, and what tensor each reads.
4. **Build cost.** Why does the constructor build 124 linears, and why does on-disk
   MXFP8 cut construction from ~76 s to ~5.3 s? (Foreshadows Chapters 12–13.)

*Next: [Chapter 11 — The Blackwell target & the roofline](11-blackwell-roofline.md), opening Part III.*
