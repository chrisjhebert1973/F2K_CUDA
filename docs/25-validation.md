# Chapter 25 — Validation Methodology

> *Goal of this chapter:* how correctness was established for a from-scratch port of
> a 9B model, and how the hardest bugs were found. We cover the test hierarchy (FP32
> component references, bit-exact wiring tests, reference-pipeline comparison), the
> diffusers/HF reference tooling, the master debugging technique (inject a known-good
> intermediate), the metrics (cosine similarity and its thresholds), and the two
> defining debugging sagas — the text-attenuation hunt and the VAE bugs.
>
> *Prerequisites:* the whole course; the bugs are cross-referenced to Appendix B.

---

## 25.1 Why validation is the hard part

Porting a model is not mostly about writing kernels — it is about *proving they
compute the same function as the reference*, when the reference is a 9B network whose
correct output is a specific image you cannot eyeball for bit-errors. A single
swapped slice, off-by-one layer, or wrong RoPE pairing produces output that is
**plausible but wrong** — coherent images that ignore the prompt, or subtly degraded
ones. The defenses are a layered test discipline and a debugging method built into
the tool (Chapter 23's isolation flags).

## 25.2 The test hierarchy

Three levels, weakest-to-strongest guarantee:

**1. Component tests vs an FP32 host reference.** Every kernel and module has a
`tests/test_*.cu` that runs it on random data and compares to a CPU FP32 computation
by **cosine similarity** and relative-L2. Thresholds reflect the precision: BF16
elementwise kernels hit `max_err ≈ 0.004` (rounding); the NVFP4 `Linear` hits cos
0.989; attention hits **cos 1.00000** (BF16 noise only). These catch *arithmetic*
errors.

**2. Bit-exact wiring tests (the zero-gate trick).** AdaLN-Zero (Chapter 7) makes a
block the *exact identity* when its gates are zero. `test_double_stream_block` and
`test_single_stream_block` run a block with zeroed gates and assert the output equals
the input **bit-for-bit** (`max_err = 0`). This proves the *entire wiring* — every
norm, projection, RoPE, attention, split, residual connects correctly — because any
miswire would perturb the supposedly-untouched residual. These catch *connection*
errors that tolerance-based tests miss. The same idea validates the stack
(`test_mmdit_stack`: bit-exact identity through 8 blocks).

**3. Comparison to the reference pipeline (diffusers/HF).** The ultimate oracle is
the original model. The project compares at multiple boundaries (§25.3) — this is
what catches *convention* errors (slice order, layer indices, packing order) that are
arithmetically fine but semantically wrong.

> A cautionary note (Appendix B, the CuTe layout bug): **structured-input component
> tests can pass while the model is wrong.** The NVFP4 weight-layout bug passed
> all-ones and K-constant tests but corrupted random weights. Component tests must
> include *random* inputs, and the reference-pipeline comparison is the backstop.

## 25.3 The reference tooling

A suite of Python tools dumps diffusers/HF intermediates, and C++ comparators check
ours against them — at *every* pipeline boundary, which is what makes isolation
possible:

| tool | dumps / does | checks our |
|---|---|---|
| `tools/qwen3_golden.py` + `test_qwen_golden.cu` | HF Qwen3 per-layer hidden states | encoder (cos ≥ 0.974/layer) |
| `tools/diffusers_prompt_embeds.py` (→ `--embeds`) | diffusers `[512×12288]` conditioning | transformer with known-good text |
| `tools/cmp_transformer.cu` + `diffusers_block_dump.py` | one transformer forward vs diffusers | transformer (cos 0.963 NVFP4 / 0.999 MXFP8) |
| `tools/diffusers_latent_dump.py` (→ `--decode_latent`) | diffusers ground-truth latent | VAE decoder in isolation |
| `tools/diffusers_vae_encode_dump.py` + `cmp_vae_encode.cu` | diffusers posterior-mean latent | VAE **encoder** (cos 0.9997, 256–2048px; Ch 26) |
| `tools/diffusers_e2e.py` | full diffusers pipeline image | end-to-end ground truth |
| `tools/probe_ctx_emb.cu` | per-row cos of ContextEmbedder in/out | text-side Linear ablations |
| `tools/quant_sim.py` | PyTorch fake-quant quality study | the FP4-vs-FP8 quality question (Ch 12) |

## 25.4 The master technique: inject a known-good intermediate

The single most valuable debugging move, used repeatedly: **at a stage boundary,
replace our intermediate with the reference's, and see if the bug persists.** This
splits "is stage A wrong?" from "is stage B wrong?" in one experiment. `generate`'s
flags (Chapter 23) are built for exactly this:

- `--embeds <diffusers conditioning>` → if the image is *still* wrong, the fault is in
  the **transformer**, not the encoder/tokenizer.
- `--decode_latent <diffusers latent>` → if the image is *still* mush, the fault is in
  the **VAE**, not the denoise trajectory.
- `--seed` → vary only the noise to measure latent-vs-prompt influence.

Both defining bugs below were localized by this technique in a single decisive run.

## 25.5 Saga 1 — the text-attenuation hunt

**Symptom.** Generated images were coherent but *barely depended on the prompt*: a
cat prompt vs a truck prompt differed by **0.12%** in pixels, while two random seeds
(same prompt) differed by ~10%. The latent dominated the prompt ~40:1 — the text path
was alive but throttled to a whisper.

**Isolation.** Feeding diffusers' *own* known-good conditioning via `--embeds`
produced the *same* broken image. That single run proved the encoder/tokenizer were
**not** the cause — even perfect conditioning failed — localizing the bug to the C++
**transformer's text handling**. (It also ruled out a long list of red herrings:
tokenizer, layer indices, chat template — all already correct.)

**The hunt.** Suspects were investigated and *eliminated with evidence*, not
hand-waving: NVFP4 outlier-collapse (disproved — scaling the 22528 outliers down 100×
didn't change ContextEmbedder cos or the image, via `probe_ctx_emb`); joint-attention
concat order; modulation gates. Several real fixes landed along the way (the {9,18,27}
→ {8,17,26} off-by-one, the chat template, seq 128→512, the patchify channel order,
uniform→Gaussian noise init, 4-axis RoPE incl. the text stream) — each improved
things slightly but none restored prompt control.

**Root cause.** The `ModulationMLP` sliced every `Flux2Modulation` output as
`[scale, shift, gate]`, but diffusers emits **`[shift, scale, gate]`** (Chapter 7).
All 32 blocks computed `(1+shift)·x + scale` instead of `(1+scale)·x + shift`,
corrupting AdaLN — the *only* channel through which the prompt modulates the network.
The fix (swap the slice order; keep `norm_out` scale-first) took prompt influence
from **0.12% to 12.5%** (~100×), making it dominant over the latent. *That one
ordering change* turned "ignores the prompt" into "renders the prompt."

**The lesson.** A swapped slice doesn't crash or NaN — it produces plausible output.
It was findable only by (a) isolating the stage with `--embeds`, and (b) being willing
to diff *conventions* against the reference, not just trust the math.

## 25.6 Saga 2 — the VAE bugs

**The "mush."** Even with the transformer fixed, images were soft/mushy. Decisive
test: feed diffusers' **ground-truth latent** through our VAE (`--decode_latent`) →
still mush. That removed the transformer trajectory from suspicion and localized the
fault to the **VAE**. Cause: the missing **`post_quant_conv`** (a $1\times1$ conv
diffusers runs before the decoder, a *sibling* of the `decoder.` prefix, easily
skipped). Folding it in → instant sharp cat (Chapter 20).

**The patch-grid.** A hard $2\times2$ checkerboard across the image. It was
*structural*, not undercooking — proven by running 28 steps and seeing it unchanged.
Cause: the missing **batch-norm de-normalization** of the patch latent; each patch's
four sub-pixels live in different channels with different stats, so skipping the
de-norm mis-scaled them into a grid (Chapter 3). Loading `bn.running_{mean,var}` and
applying the affine fixed it.

**The lesson (both).** Reference pipelines hide affine/normalization steps in
unobvious places. **Diff at the *latent* boundary, not just the final image** — both
were localized by comparing/injecting latents, and the "more steps?" test
distinguished structural bugs from undercooking.

## 25.7 The metric: cosine similarity (and why)

The recurring metric is **cosine similarity** between our output and the reference,
flattened. Why cosine and not, say, mean-abs-error: low-precision (NVFP4/BF16)
introduces a roughly *uniform* magnitude perturbation, so the error is mostly
**directional**; cosine isolates whether the *structure* matches independent of a
global scale. Thresholds that recur:

- **cos ≈ 1.00000** — bit-exact-up-to-BF16 (attention, conv, kernels). Anything less
  is a real difference.
- **cos ≥ 0.974** — Qwen3 encoder per layer (NVFP4 quant noise).
- **cos 0.963 (NVFP4) / 0.999 (MXFP8)** — full transformer forward; the NVFP4 gap is
  *quantization*, not a bug (proved by the MXFP8 number and `quant_sim.py`).

Crucially, the method **distinguishes quantization noise from bugs**: when the NVFP4
transformer hit 0.963, the question was "bug or FP4?" — answered by switching to MXFP8
(0.999) and the fake-quant study (Chapter 12). A bug would not vanish with precision;
quantization noise does. Likewise, the 1.8% pixel diff between attention kernels
(Chapter 17) was confirmed as reordered-summation BF16 drift — both kernels matched
FP32 at cos 1.0 — not a regression.

## 25.8 Summary and what to carry forward

- Validate in **layers**: FP32 component tests (random inputs!), **bit-exact zero-gate
  wiring tests**, and **comparison to the reference pipeline** at every boundary.
- Build **isolation** into the tool: inject a known-good intermediate (`--embeds`,
  `--decode_latent`, `--seed`) to split which stage is wrong in one run.
- **Convention bugs** (slice order, layer indices, packing, RoPE pairing) are
  plausible-but-wrong and don't crash — find them by **diffing intermediates against
  the reference**, not by eyeballing images.
- Use **cosine similarity** (directional, scale-invariant) with precision-aware
  thresholds, and **distinguish quantization noise from bugs** by varying precision.
- The two defining sagas — modulation scale/shift swap (text attenuation) and the VAE
  post_quant_conv/bn-de-norm (mush/grid) — were each localized by injecting a
  reference intermediate and diffing at the latent boundary.

This closes Part VII — the text-to-image pipeline, proven correct. **Part VIII** then
goes beyond it: the VAE encoder (Chapter 26) that closes the autoencoder loop, the
image-conditioning techniques it unlocks (Chapter 27 — img2img, inpainting,
upscaling), and the serving layer that streams it live (Chapter 28). The appendices
follow: a glossary (A), the bug museum (B), and the build & reproduce guide (C).

---

### Exercises

1. **Design the isolation.** Given a wrong image, write the exact sequence of
   `--embeds`/`--decode_latent`/`--seed` runs (and what each outcome implies) to
   localize the fault to one of: tokenizer, encoder, transformer, VAE.
2. **Why zero-gate.** Explain why a bit-exact zero-gate identity test catches wiring
   bugs that a cos>0.99 tolerance test would pass, and name a bug class it *cannot*
   catch.
3. **Bug or quant?** You see transformer cos 0.95. Describe the two experiments that
   decide whether it is a bug or quantization noise.
4. **The structured-test trap.** Explain how the CuTe layout bug passed all-ones and
   K-constant tests, and write the random-input test that catches it (Chapter 13).

*Next: [Chapter 26 — Assembling the VAE encoder](26-vae-encoder.md), opening Part VIII.*
