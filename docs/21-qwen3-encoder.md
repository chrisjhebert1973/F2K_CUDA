# Chapter 21 — The Qwen3 Encoder

> *Goal of this chapter:* implement the text encoder whose *role* we covered in
> Chapter 05 — the 8B Qwen3 LLM run as a single forward pass to produce the
> `[512 × 12288]` conditioning. We cover the Qwen3-8B architecture (grouped-query
> attention, per-head q/k-norm, its own RoPE), the layer and stack implementation,
> the capture-layer mechanism, and the validation against HuggingFace. Anchored to
> `qwen_encoder.{h,cu}`, `qwen_layer.*`, `qwen_attention.*`, `embed_lookup.*`.
>
> *Prerequisites:* Chapters 05 (the role), 06 (RoPE), 08 (attention/QK-norm), 13
> (Linear), 15 (RMSNorm/SwiGLU).

---

## 21.1 Qwen3-8B at a glance

From `text_encoder/config.json`:

| param | value |
|---|---|
| layers | **36** |
| hidden | **4096** |
| query heads | **32** |
| **kv heads** | **8** (grouped-query attention, group size 4) |
| head_dim | **128** |
| FFN | **SwiGLU**, intermediate **12288** |
| RMSNorm eps | **1e-6** |
| **rope_theta** | **1e6** (note: ≠ the transformer's 2000) |
| attention_bias | **false** (no biases on q/k/v/o) |
| vocab | **151936** |

It is a standard decoder-only LLM (the LLaMA/Qwen lineage): token embedding, then 36
identical transformer layers, then a final RMSNorm. We run it as an **encoder** —
one forward pass over the padded prompt, capturing hidden states (Chapter 05), no
generation. Three things distinguish it from the FLUX transformer of Part II and are
worth care: **grouped-query attention**, **per-head q/k-norm**, and its **own RoPE
convention** ($\theta=10^6$, half-rotation).

## 21.2 Grouped-query attention (GQA)

Standard multi-head attention has one K and one V per query head. **GQA** shares each
K/V head across a *group* of query heads: Qwen3 has 32 query heads but only **8 KV
heads**, so each KV head serves **4** query heads (group size $32/8 = 4$).

Why: K and V are the memory-heavy operands (in autoregressive decoding they form the
KV cache; here they are the bandwidth cost of the projections and the attention
reads). Cutting KV heads 4× shrinks the K/V projection weights and the K/V bytes
streamed in attention (Chapter 11 — attention is memory-bound), for a small quality
cost. The query projection stays full-width (32 heads); only K and V are narrowed.

In the kernel (`qwen_attention.cu`), each query head $h$ attends to the K/V of head
$h/4$:

```
 query heads:  0  1  2  3 | 4  5  6  7 | ... | 28 29 30 31
 kv heads:     └──  0  ──┘ └──  1  ──┘  ...  └──   7    ──┘
 head h (query) uses K,V from kv-head (h / 4)
```

The attention is **causal** (decoder-only): query position $i$ attends only to keys
$0..i$ (Chapter 05 — later positions carry the most context, which is why the
capture layers matter). `tests/test_qwen_attention.cu` validates the GQA causal
kernel at the Qwen3 shape ($B{=}1, S{=}128, H_q{=}32, H_{kv}{=}8, D{=}128$) against an
FP32 host reference (max err 0.001).

## 21.3 A Qwen3 layer

`qwen_layer.cu` implements one of the 36 layers (pre-norm transformer with SwiGLU):

```
 # attention sub-layer
 h    = RMSNorm(x, in_norm_gain)                 # eps 1e-6
 q    = Linear(h, Wq) → [S, 32·128]              # no bias
 k    = Linear(h, Wk) → [S,  8·128]              # GQA: 8 kv heads
 v    = Linear(h, Wv) → [S,  8·128]
 q    = RMSNorm(q, q_norm_gain) per head         # per-head q-norm  ┐ BEFORE RoPE
 k    = RMSNorm(k, k_norm_gain) per head         # per-head k-norm  ┘ (Qwen3-specific)
 q,k  = RoPE(q, k)                               # half-rotation, θ=1e6
 a    = GQA_causal_attention(q, k, v)            # head h → kv head h/4
 x    = x + Linear(a, Wo)                        # residual, no bias
 # MLP sub-layer
 h    = RMSNorm(x, post_norm_gain)
 x    = x + SwiGLU_MLP(h)                         # gate/up/down, intermediate 12288
```

Notes:

- **Per-head q/k-norm before RoPE.** Like the FLUX blocks (Chapter 08) Qwen3
  RMS-normalizes each head's Q and K — but with its *own* learned gains, and applied
  **before** RoPE. This ordering (norm, then rotate) is part of the Qwen3 spec; the
  same `rmsnorm_bf16` kernel is reused with `batch_rows = S · n_heads`.
- **RoPE: half-rotation, $\theta=10^6$.** Qwen3 uses the **LLaMA-style $(i, i+D/2)$
  pairing** (`rope.cu`, Chapter 06) — *not* the interleaved $(2k,2k+1)$ of the FLUX
  transformer — and base $10^6$. Using the wrong pairing or base silently corrupts
  the conditioning. (This is exactly why the codebase keeps both RoPE kernels.)
- **No biases.** `attention_bias=false`: the q/k/v/o projections have no bias terms.
- **SwiGLU MLP.** Same gated structure as the FLUX MLP (Chapter 15), intermediate
  12288.

## 21.4 The encoder stack and capture

`qwen_encoder.cu` assembles the full model:

```
 token_ids [512] → embed_lookup(embed_tokens)  → [512, 4096]   (BF16 gather, Chapter 15)
   → Layer 0 → Layer 1 → ... → Layer 35
        with hidden states captured at layers {8, 17, 26}        (Chapter 05)
   → (final RMSNorm — used for the standard LM output, not needed for capture)
 capture: concat( h_8, h_17, h_26 )  → [512, 12288]               → ContextEmbedder
```

The **capture mechanism** is the encoder's whole purpose: `Config::capture_layers`
(`= {8, 17, 26}`) records which layer *outputs* to keep, and the encoder concatenates
those three `[512, 4096]` tensors along the channel axis into the `[512, 12288]`
conditioning (Chapter 05's 3×4096). The capture indices `{8,17,26}` are the corrected
HF off-by-one of diffusers' `{9,18,27}` (Chapter 05). `embed_lookup` is a simple BF16
row-gather from the (BF16-kept) `embed_tokens` table; `lm_head` is unused (we are an
encoder, not a generator).

## 21.5 Quantization and the outliers

The encoder's weights are NVFP4 in the F2K file (`qwen3_f2k/`, 4 shards, 4.8 GiB),
**except** `embed_tokens` which is kept BF16 (a `--keep-bf16` flag in `f2k_convert`,
Chapter 12/14) — embedding tables tolerate quantization poorly and are looked up, not
matmul'd. The NVFP4 quant quality across shards is the same as the transformer (worst
`max_err 0.144`). Recall the **outlier activations** (~22528) from Chapter 05: they
appear in the late layers, and the per-microblock scales handle them (Chapter 12/25)
— the encoder's conditioning is correct despite them.

## 21.6 Validation against HuggingFace

The encoder is checked against the real HF Qwen3-8B, not just internally:

- `tools/qwen3_golden.py` runs HF Qwen3 on a fixed prompt and dumps per-layer hidden
  states; `tests/test_qwen_golden.cu` runs *our* encoder and compares.
- Result: **cosine similarity ≥ 0.974 from layer 0 through 34**, magnitude ratios
  within ~25%. The dominant error source is NVFP4 quantization (worst per-tensor
  `max_err 0.14`), not an implementation bug.
- A subtle HF gotcha was caught here: `output_hidden_states=True` returns
  `n_layers+1` states — `hs[0]` is the embedding, `hs[i]` is the output of layer
  $i-1$, and `hs[N]` is *post-final-norm*. Comparing our layer-$N{-}1$ output against
  `hs[N]` (post-norm) drops cos to ~0.69 from the gamma rotation; comparing against
  `hs[N-1]` gives ≥0.974. (Same off-by-one family as the capture indices.)

Performance: **4 shards load + build in ~10.5 s** (one-time), **~50 ms forward at
seq=128**, ~77 MiB workspace. At seq=512 (the real prompt length) it is the ~0.28 s
text-encode of Chapter 00 — negligible next to the denoise loop, and ~350× faster
than the PyTorch path it replaced.

## 21.7 Where this lives in the code

| Concept | Code |
|---|---|
| token embedding gather | `kernels/embed_lookup.{h,cu}` |
| GQA causal attention | `qwen_attention.{h,cu}` |
| one Qwen3 layer | `qwen_layer.{h,cu}` |
| 36-layer stack + capture | `qwen_encoder.{h,cu}` |
| RoPE (half-rotation, θ=1e6) | `kernels/rope.{h,cu}` (Chapter 06) |
| validation vs HF | `tools/qwen3_golden.py`, `tests/test_qwen_golden.cu` |
| in-pipeline use | `generate.cu` (`--prompt`/`--tokens`, capture {8,17,26}) |

## 21.8 Summary and what to carry forward

- The text encoder is **Qwen3-8B** run as a single forward pass; we implement embed
  → **36 layers** → capture.
- **GQA** (32 query / 8 KV heads, group 4) shrinks the memory-heavy K/V; attention is
  **causal**.
- Each layer: pre-RMSNorm → q/k/v (no bias) → **per-head q/k-norm before RoPE** →
  **half-rotation RoPE, θ=1e6** (its own convention, *not* the FLUX transformer's) →
  GQA → o-proj → residual → RMSNorm → SwiGLU.
- Capture layer outputs **{8, 17, 26}** (corrected HF off-by-one) and concat to
  `[512, 12288]`; `embed_tokens` stays BF16, rest NVFP4.
- Validated vs HF at **cos ≥ 0.974/layer** (NVFP4 noise dominates); ~10.5 s build,
  ~50 ms forward at seq 128. Mind HF's `hidden_states` off-by-one when validating.

Chapter 22 covers the last non-CUDA piece — the native byte-level BPE tokenizer that
feeds this encoder and made the pipeline self-contained.

---

### Exercises

1. **GQA bookkeeping.** With 32 query and 8 KV heads, write the query→kv-head map and
   compute the parameter and K/V-byte savings vs full 32-head MHA at hidden 4096.
2. **Two RoPEs.** Contrast the encoder's RoPE (half-rotation, θ=1e6) with the FLUX
   transformer's (interleaved, θ=2000, 4-axis). Why does the codebase keep both
   kernels, and what breaks if you swap them?
3. **The HF off-by-one (twice).** Explain how the *same* indexing convention causes
   both the capture-layer shift ({9,18,27}→{8,17,26}) and the golden-comparison trap
   (`hs[N]` is post-norm).
4. **Encoder vs generator.** List what we *don't* run (sampling, KV cache, lm_head)
   and why running Qwen3 as an encoder is cheaper than as a chatbot.

*Next: [Chapter 22 — A native byte-level BPE tokenizer](22-tokenizer.md).*
