# Chapter 05 — Text Conditioning with Qwen3

> *Goal of this chapter:* explain how a text prompt becomes the `[512 × 12288]`
> conditioning tensor that steers the diffusion. We cover why FLUX.2-klein uses a
> full **8-billion-parameter LLM (Qwen3-8B)** as its text encoder, how a
> *decoder-only causal* language model is repurposed as an *encoder*, the
> three-layer hidden-state concatenation that produces the 12288 dimension, the
> chat-template wrapping, and the context embedder that brings it into the model.
> The encoder's GPU *implementation* (grouped-query attention, q/k-norm, etc.) is
> Chapter 21; here we cover its *role*.
>
> *Prerequisites:* Chapter 04 (where conditioning enters); transformers at a
> conceptual level.

---

## 5.1 Why an 8B LLM is the text encoder

Earlier text-to-image models used CLIP text encoders (~100M params) or T5-XXL
(~5B). FLUX.2-klein uses **Qwen3-8B** — a frontier-class large language model — as
its text encoder. Why spend 8B parameters just to *read* the prompt?

Because prompt understanding is the bottleneck for instruction-following. A small
text encoder produces embeddings that capture keywords but not composition,
negation, counting, spatial relations, or world knowledge. A large LM has *already
learned* these from language pretraining; its hidden states encode a rich, almost
"thought-through" representation of the prompt. The diffusion transformer then has
something far more informative to attend to. The cost is real (Qwen3-8B is bigger
than some image backbones), but it runs **once** per image (Chapter 00) and its
output is reused across all denoise steps, so it is amortized.

> This is a general trend: as image models got better at *rendering*, the limiting
> factor became *understanding the request*, and the field responded by bolting on
> ever-larger language models as the front end.

## 5.2 A decoder-only LM used as an encoder

Qwen3 is a **decoder-only, causal** language model — the GPT family architecture.
Normally it is run autoregressively to *generate* text: token by token, each
position attending only to earlier positions (the causal mask). FLUX uses it
differently: it feeds the *entire prompt at once* and reads out the **hidden states
at every position** — the internal activations — rather than sampling output
tokens. No generation happens; the LM is run as a single forward pass and its
intermediate representations *are* the conditioning.

A few consequences of using a causal model this way:

- **Causal masking still applies.** Position $i$'s hidden state summarizes tokens
  $0..i$. The later positions therefore carry the most context, which matters for
  the layer-selection in §5.4.
- **A chat template is used** (§5.3), because Qwen3 was instruction-tuned and its
  representations are best-behaved when the input looks like the chat format it was
  trained on.
- **The padding tokens are encoded too.** The prompt is padded to a fixed length
  (512); diffusers feeds all 512 positions, padding included, with no attention
  mask into the *diffusion* transformer (the image tokens simply attend to all 512,
  and the model learned to ignore padding). This is a quirk worth remembering when
  validating (Chapter 25).

## 5.3 The chat template

Qwen3 expects its instruction-tuned chat format. The prompt is wrapped exactly as
`apply_chat_template([{role:user, content:PROMPT}], add_generation_prompt=True,
enable_thinking=False)` produces (this is reproduced bit-for-bit by our native
tokenizer, Chapter 22):

```
<|im_start|>user\n{PROMPT}<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n
```

The special tokens (`<|im_start|>`, `<|im_end|>`, `<think>`, `</think>`) are single
token ids; the prompt and the literal words `user`/`assistant`/newlines are
byte-level BPE'd. The trailing empty `<think>\n\n</think>` blocks (with
`enable_thinking=False`) put the model in its "no chain-of-thought" mode. Feeding
the *raw* prompt without this wrapper shifts the activations off the distribution
Qwen3 was tuned on and measurably degrades conditioning — the template is not
optional decoration.

The wrapped text is tokenized, padded/truncated to **512** tokens with the pad
token (`<|endoftext|>`, id 151643), and fed to the encoder. (The padding side is
right; the wrapper and padding are exactly what `tools/encode_prompt.py` and our
native `BpeTokenizer` produce — Chapter 22.)

## 5.4 The three-layer concatenation → 12288

Here is the detail that explains the mysterious `t5_dim = 12288` from Chapter 04.
Qwen3-8B has hidden size **4096** and **36** layers. FLUX does **not** use just the
final hidden state; it extracts the hidden states at **three specific layers** and
concatenates them along the channel dimension:

$$
3 \times 4096 = 12288 .
$$

So the conditioning is `[512 tokens × 12288]`, where each token's 12288 vector is
`concat(h_layerA, h_layerB, h_layerC)` for three chosen layers. Using multiple
layers gives the diffusion model a mix of representational depths — earlier layers
carry more surface/lexical features, later layers more abstract/semantic ones.

**Which three layers?** This is a genuine gotcha with an off-by-one. diffusers
requests `output.hidden_states[k] for k in (9, 18, 27)`. But HF's
`hidden_states` array is offset: `hidden_states[0]` is the *embedding* output, and
`hidden_states[k]` for $k\ge1$ is the output of *layer $k-1$*. So the actual layer
outputs to capture are **layers {8, 17, 26}** (zero-indexed). This project's
encoder captures exactly those (`capture_layers = {8, 17, 26}` in `generate.cu`).
Getting this wrong (capturing {9,18,27}) silently shifts every conditioning vector
by one layer — the kind of bug that produces *plausible but subtly wrong* images.
(Appendix B; Chapter 21 for the capture mechanism.)

```
 prompt → tokenizer → [512 ids]
                         │ Qwen3-8B forward (36 layers, causal)
                         ▼
   capture hidden states at layers  8, 17, 26   (each [512 × 4096])
                         │ concat along channels
                         ▼
   conditioning  [512 × 12288]   ──▶  ContextEmbedder  ──▶  [512 × 4096]  (into the MMDiT)
```

## 5.5 The outlier activations

A practical wrinkle: Qwen3's late-layer activations contain **large-magnitude
outliers** — diffusers' actual `prompt_embeds` has `abs_max ≈ 22528` at the
real-prompt (non-padding) positions, from activation spikes at chat-template
tokens. These are not errors; they are how the model concentrates information. They
matter for *quantization* (Chapter 12): a per-block scale that has to span a 22528
outlier could crush the small values sharing its block. The project investigated
whether NVFP4 was destroying conditioning via these outliers and found it was
*not* — the per-microblock E4M3 scales handle them correctly (Chapter 25, the
text-attenuation investigation). Worth knowing the outliers exist when you read the
quantization chapters.

## 5.6 The context embedder

The `[512 × 12288]` conditioning is not yet in the transformer's world; it must be
projected to the hidden width 4096. That is the **context embedder**: a single
`Linear` with weight `context_embedder.weight` of shape `[4096, 12288]`. It maps
each text token's 12288-vector to a 4096-vector that lives in the same space as the
image tokens, ready for joint attention (Chapter 08).

```c++
// conceptually, per text token:
txt_hidden[i] = context_embedder( conditioning[i] );   // [12288] → [4096]
```

This is the `ContextEmbedder` in `src/backend/cuda/embeddings.{h,cu}` (Chapter 10),
and it is the entry point on the text side of the Chapter 04 diagram. From here the
text tokens flow through the 8 double-stream blocks (developing their own
representation while the image attends to them) and into the single-stream stack.

## 5.7 Three ways to supply conditioning (and why)

`generate` can inject text at three points, which is invaluable for isolating bugs
(and is how the text path was debugged — Chapter 25):

| Flag | Skips | Use |
|---|---|---|
| `--prompt "<text>"` | nothing | the full native path (tokenizer → Qwen3 → embedder) |
| `--tokens <file>` | tokenizer | feed precomputed token ids (test the encoder) |
| `--embeds <file>` | tokenizer + encoder | feed a precomputed `[512 × 12288]` (test the *transformer* in isolation) |

The `--embeds` path was the key debugging lever: by feeding diffusers' *own*
known-good conditioning into our transformer, the team proved the text path's
residual problems were in the C++ transformer (the modulation bug, Chapter 07/25),
not in the encoder — because even perfect conditioning produced the wrong image
until the modulation fix landed. The native `--prompt` path (Chapter 22) later made
the whole thing self-contained and ~350× faster than the PyTorch `--embeds`
generation (0.28 s vs 98 s).

## 5.8 Where this lives in the code

| Concept | Code |
|---|---|
| Qwen3 encoder (full impl) | `qwen_encoder.{h,cu}`, `qwen_layer.*`, `qwen_attention.*` (Chapter 21) |
| capture layers {8,17,26} | `generate.cu` qwen config |
| chat template + tokenization | `bpe_tokenizer.{h,cpp}`, `encode_prompt.py` (Chapter 22) |
| context embedder (12288→4096) | `embeddings.{h,cu}` `ContextEmbedder` (Chapter 10) |
| validation vs HF Qwen3 | `tools/qwen3_golden.py`, `tests/test_qwen_golden.cu` (cos ≥ 0.974/layer) |
| validation vs diffusers embeds | `tools/diffusers_prompt_embeds.py`, `--embeds` path |

## 5.9 Summary and what to carry forward

- The text encoder is **Qwen3-8B**, a decoder-only causal LLM run as a single
  forward pass; its **hidden states** are the conditioning, not generated text.
- The conditioning is `[512 × 12288]`, where 12288 = **3 × 4096** from
  concatenating hidden states at **layers {8, 17, 26}** (mind the HF off-by-one).
- A **chat template** wraps the prompt; padding is to 512 and is fed through.
- Late-layer **outlier activations** (~22528) exist and matter for quantization but
  are handled correctly by NVFP4's per-microblock scales.
- The **context embedder** (`[4096, 12288]` Linear) projects conditioning into the
  hidden space; from there it joins the image tokens in the double-stream blocks.

Chapter 06 covers how *position* is encoded for these tokens — the 4-axis RoPE that
distinguishes image token $(h,w)$ from text token $\ell$ — before Chapter 07's
modulation and the block chapters.

---

### Exercises

1. **Why hidden states, not logits.** Explain why FLUX reads Qwen3's hidden states
   rather than its output token probabilities. What information would the logits
   throw away?
2. **The off-by-one.** Given HF's convention (`hidden_states[0]`=embedding,
   `hidden_states[k]`=output of layer $k-1$), show that diffusers' `(9,18,27)`
   corresponds to layer outputs `{8,17,26}`. What would capturing `{9,18,27}` do?
3. **12288 decoded.** Connect `t5_dim=12288` (Chapter 04 config),
   `joint_attention_dim=12288`, and `3 × 4096`. Why is "t5_dim" a misnomer?
4. **Isolation logic.** You get a wrong image. Describe how `--embeds`, `--tokens`,
   and `--prompt` let you localize whether the fault is in the tokenizer, the
   encoder, or the transformer.

*Next: [Chapter 06 — Rotary position embeddings](06-rope.md).*
