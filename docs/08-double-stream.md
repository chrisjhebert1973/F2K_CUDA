# Chapter 08 — Double-Stream Blocks

> *Goal of this chapter:* assemble the first of FLUX's two block types in full.
> A double-stream block keeps **separate weights** for the text and image streams
> but couples them through **joint attention**. We walk the exact forward pass —
> norm → modulate → QKV → per-head QK-norm → RoPE → joint attention → projection →
> gated residual, then the gated MLP — name all 12 linears, and explain joint
> attention and QK-normalization. Anchored to `double_stream_block.{h,cu}` and
> `joint_attention.{h,cu}`.
>
> *Prerequisites:* Chapters 06 (RoPE), 07 (modulation); attention conceptually
> (Chapter 16 for the kernel).

---

## 8.1 The shape of a block

Every transformer block is two residual sub-layers: an **attention** sub-layer and
an **MLP** sub-layer, each wrapped in the AdaLN pattern of Chapter 07 (norm →
modulate → sub-layer → gate → add). A *double*-stream block runs this structure
**twice in parallel** — once for the image tokens with image weights, once for the
text tokens with text weights — except that the attention sub-layer's $QK^\top$ is
computed **jointly** over the concatenation of both streams, so information crosses
between them. FLUX.2-klein has **8** such blocks.

```
   image stream                              text stream
   x_img ──┐                                 x_txt ──┐
           │  ATTENTION SUB-LAYER (joint)            │
           ▼                                         ▼
   norm→modulate(img)                        norm→modulate(txt)
           │ to_q/k/v                                │ add_q/k/v_proj
        Qi,Ki,Vi                                  Qt,Kt,Vt
        qk-norm, RoPE                            qk-norm, RoPE
           └───────────────┬─────────────────────────┘
                           ▼
            JOINT ATTENTION over [txt ‖ img] tokens
                           │
            split → img_attn        txt_attn
           │ to_out                       │ to_add_out
        gate_attn ⊙, + residual    gate_attn ⊙, + residual
   x_img' ─┐                                 x_txt' ─┐
           │  MLP SUB-LAYER (per stream, gated)      │
           ▼                                         ▼
   norm→modulate(img)                        norm→modulate(txt)
   ff_in→split(gate,up)→silu·→ff_out         ff_ctx_in→…→ff_ctx_out
   gate_mlp ⊙, + residual                    gate_mlp ⊙, + residual
   x_img''                                   x_txt''
```

## 8.2 The attention sub-layer, step by step

Following `double_stream_block.cu` exactly, per stream:

**1. Normalize (no-affine RMSNorm).** `RMSNorm(x, ones)` — RMS normalization with a
gain of all-ones, i.e. statistics-only. (Why RMSNorm and not LayerNorm: §8.5.) The
learned affine that a normal norm would apply is *replaced* by the AdaLN modulate.

**2. Modulate.** `modulate(·, scale_attn, shift_attn)` — the AdaLN scale/shift for
this stream's attention sub-layer (Chapter 07). Image uses `img_scale/shift_attn`,
text uses `txt_scale/shift_attn`.

**3. QKV projections.** Three linears per stream produce queries, keys, values:
- image: `to_q`, `to_k`, `to_v` (weights `W_q/k/v`)
- text: `add_q_proj`, `add_k_proj`, `add_v_proj` (the `add_` prefix marks the text
  stream — the "additional" tokens added to the image's attention).

Each output is `[S × hidden]` and is viewed as `[S × n_heads × head_dim]` =
`[S × 32 × 128]`.

**4. Per-head QK-norm.** Each head's query and key vectors are RMS-normalized with
*learned* gains — `norm_q`/`norm_k` (image), `norm_added_q`/`norm_added_k` (text).
This is **QK-normalization** (§8.4), a stabilizer that normalizes Q and K *per head*
before the dot product. It uses the same `rmsnorm_bf16` kernel as elsewhere, called
with `batch_rows = S × n_heads`.

**5. RoPE.** 4-axis rotary embedding (Chapter 06) on Q and K. **Both** streams get
RoPE: image Q/K with image positions $(0,h,w,0)$, text Q/K with text positions
$(0,0,0,\ell)$. (The header comment in `double_stream_block.h` says "text gets NO
RoPE" — that comment is **stale**; the `.cu` applies `rope_4axis_inplace_bf16` to
`txt Q` and `txt K` with the text position table. Verifying the code over the
comment is the right habit; adding text RoPE was one of the text-path fixes,
Appendix B.)

**6. Joint attention.** The crux — §8.3. Produces `img_attn` and `txt_attn`.

**7. Output projection + gated residual.** `to_out` (image) / `to_add_out` (text)
project the attention output back to hidden width, then
`gated_residual(x, proj, gate_attn)` adds it into the stream scaled by the AdaLN
gate.

## 8.3 Joint attention: where the streams talk

This is the only place text and image interact. Instead of attending within each
stream separately, the block **concatenates** the two streams' Q, K, V and runs a
single attention over the combined sequence of length $S_{\text{txt}}+S_{\text{img}}$,
then splits the result back:

```
 Q = [Q_txt ; Q_img]   K = [K_txt ; K_img]   V = [V_txt ; V_img]     (concat along tokens)
 O = Attention(Q, K, V)          # every token attends to every token, both modalities
 O_txt = O[:S_txt]    O_img = O[S_txt:]       (split back)
```

Because the attention matrix is over the union, an image token's output is a
weighted sum over **all** text and image values, and vice versa — this is exactly
how the prompt's content flows into the image representation. `joint_attention.cu`
is a thin wrapper: it concatenates Q/K/V into workspace buffers, calls the
single `Attention` kernel (Part IV) at the combined length, and splits the output.
The concat/split are one-CTA-per-row data-movement kernels; all the math is in the
attention kernel, which is identical to the one used everywhere else. At 1024px the
combined length is $512+4096 = 4608$ — the $S=4608$ of the attention benchmarks
(Part IV). cos-similarity of the wrapper vs an FP32 reference is 1.00000 for both
output halves.

> Note the asymmetry that makes the two-stream design meaningful: the *attention*
> is shared (joint), but the *projections, norms, and MLP* are per-stream. Text and
> image develop distinct representations yet see each other through attention.

## 8.4 QK-normalization: why normalize queries and keys

Standard attention can suffer from attention-logit blow-up: if some query/key
directions grow large, the $QK^\top$ scores saturate the softmax and training
destabilizes. **QK-norm** (used by Qwen3, FLUX, and others) RMS-normalizes each
head's Q and K vectors *before* the dot product, bounding their magnitudes so the
score scale stays controlled. Crucially these are **learned** per-head gains
(`norm_q`, `norm_k`, etc.), unlike the no-affine RMSNorm at the block entry — the
model can still set the effective temperature per head. QK-norm is applied **before
RoPE** in this implementation (norm, then rotate), matching the reference. It is a
small kernel call but a real correctness requirement: the learned gains are part of
the weights and skipping the norm changes the attention distribution.

## 8.5 The MLP sub-layer: a gated feed-forward

After the attention sub-layer updates the residual, each stream runs an independent
gated MLP (the AdaLN pattern again, with `scale_mlp/shift_mlp/gate_mlp`):

```
 x → RMSNorm(ones) → modulate(scale_mlp, shift_mlp)
   → ff_in:   Linear → [2·ffn]            (a single fused projection)
   → split_half → (gate, up), each [ffn]  (split_half_bf16 kernel)
   → silu(gate) · up → [ffn]              (SwiGLU-style gating; silu_mul kernel)
   → ff_out:  Linear → [hidden]
   → gated_residual(x, ·, gate_mlp)
```

The MLP inner width is `ffn = 12288` (mlp_ratio 3.0). The first linear produces
**twice** that ($2\times12288$) and `split_half` cuts it into a *gate* and an *up*
half; `silu(gate)·up` is the gated activation (the header calls it "GeGLU" but the
activation is SiLU, i.e. a SwiGLU-style gate — what the code computes is
`silu(gate)·up`); the second linear projects back to hidden. Text uses the
parallel weights `ff_ctx_in`/`ff_ctx_out`. Gating MLPs like this consistently
outperform a plain `Linear→act→Linear` at equal parameter budget, which is why
modern transformers use them.

## 8.6 The twelve linears

A double-stream block owns **12** NVFP4/MXFP8 `Linear`s — six per stream:

| | image weights | text weights |
|---|---|---|
| Q proj | `to_q` | `add_q_proj` |
| K proj | `to_k` | `add_k_proj` |
| V proj | `to_v` | `add_v_proj` |
| attn out | `to_out` | `to_add_out` |
| MLP in (→2·ffn) | `ff_in` | `ff_ctx_in` |
| MLP out (ffn→hidden) | `ff_out` | `ff_ctx_out` |

plus 4 per-head QK-norm gains (`norm_q`, `norm_k`, `norm_added_q`,
`norm_added_k`) and the modulation it *reads* (the shared `ModulationMLP` output,
not owned by the block). 8 blocks × 16 tensors = the 128 double-block tensors the
router accounts for (Chapter 14).

## 8.7 Correctness: the zero-gate identity test

A beautiful property falls out of AdaLN-Zero (Chapter 07): if **all gates are
zero**, both sub-layers contribute nothing and the block is the **exact identity**.
`tests/test_double_stream_block.cu` exploits this: it runs a block with zeroed
gates and checks the output equals the input **bit-for-bit** (`max_err = 0`). This
single test proves the *entire wiring* is correct — every norm, modulate, QKV,
RoPE, attention, projection, split, and residual connects properly, because any
miswire would perturb the supposedly-untouched residual. A separate
real-weights smoke test (loading `transformer_blocks.0` from the F2K file) checks
the block produces finite, sanely-scaled outputs on actual data.

> The zero-gate identity test is a pattern worth stealing: design your residual
> components so that a known parameter setting makes them provably the identity,
> then test that exactly. It catches wiring bugs that numerical-tolerance tests
> miss.

## 8.8 Where this lives in the code

| Concept | Code |
|---|---|
| the block | `double_stream_block.{h,cu}` (`DoubleStreamBlock`) |
| joint attention concat/split | `joint_attention.{h,cu}` |
| QK-norm | `kernels/rmsnorm.{h,cu}` (called per-head) |
| 4-axis RoPE (both streams) | `kernels/rope_4axis.*` |
| MLP gate/up split | `kernels/seq_ops` `split_half_bf16`; `swiglu`/`silu_mul` |
| modulation it consumes | `ModulationMLP` (Chapter 07) |
| tests | `tests/test_double_stream_block.cu`, `tests/test_joint_attention.cu` |

## 8.9 Summary and what to carry forward

- A double-stream block runs the AdaLN attention+MLP pattern **separately per
  stream** (own weights), but the attention's $QK^\top$ is **joint** over
  `[txt ‖ img]`, which is how the prompt flows into the image.
- The attention path is: no-affine RMSNorm → AdaLN modulate → QKV → **per-head
  QK-norm (learned)** → **4-axis RoPE on both streams** → joint attention → out
  proj → gated residual.
- The MLP is a **SwiGLU-style gated** feed-forward (`ff_in`→split→`silu(gate)·up`→
  `ff_out`), per stream.
- **12 linears + 4 norm gains** per block, ×8 blocks; modulation is shared.
- The **zero-gate identity test** proves the wiring bit-exactly. *Verify code over
  comments* — the "no text RoPE" comment is stale.

Chapter 09 covers the single-stream blocks — the other 24 layers — which fuse all
of this into fewer, larger GEMMs.

---

### Exercises

1. **Joint vs separate attention.** Explain what information an image token gains
   from *joint* attention that it could not get if text and image attended
   separately. Where in the diagram does the prompt enter the image stream?
2. **QK-norm placement.** Why is QK-norm applied *before* RoPE rather than after?
   What would rotating *then* normalizing change?
3. **Zero-gate proof.** Argue why zeroing all gates makes a double-stream block the
   exact identity, and why that makes a strong wiring test. What class of bug would
   it *not* catch?
4. **Fused MLP-in.** The MLP's first linear outputs `2·ffn` and is split into
   gate/up. Why is producing both halves in one GEMM better than two separate
   GEMMs on a bandwidth-bound machine? (Foreshadows Chapter 09/13.)

*Next: [Chapter 09 — Single-stream blocks](09-single-stream.md).*
