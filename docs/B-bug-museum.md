# Appendix B — The Bug Museum

> Every nontrivial bug from the port, collected as **symptom → cause → fix →
> lesson**. These are referenced throughout the course; gathered here, they form a
> field guide to the failure modes of porting a large model to a new GPU. The
> recurring theme: the worst bugs **don't crash** — they produce plausible-but-wrong
> output, and are found only by diffing against a reference.

---

## B.1 The modulation scale/shift swap (the big one)

- **Symptom.** Images were coherent but barely prompt-dependent: cat vs truck differed
  0.12% in pixels; two seeds differed ~10%. The prompt was throttled ~40:1.
- **Cause.** `ModulationMLP` sliced every `Flux2Modulation` output as `[scale, shift,
  gate]`; diffusers emits `[shift, scale, gate]`. All 32 blocks computed
  `(1+shift)·x+shift'` instead of `(1+scale)·x+shift`, corrupting AdaLN — the only
  channel through which the prompt modulates the network.
- **Fix.** Swap the slice order to `[shift, scale, gate]` for img/txt/single; keep
  `norm_out` **scale-first** (it's `AdaLayerNormContinuous`, a different convention).
- **Result.** Prompt influence 0.12% → **12.5%** (~100×), now dominant over the latent.
- **Lesson.** The order of packed sub-tensors is a silent contract. A swap doesn't
  crash or NaN — diff *conventions* against the reference. (Ch 7, 25)

## B.2 VAE missing `post_quant_conv` (the "mush")

- **Symptom.** Soft, mushy images even with a correct transformer.
- **Cause.** diffusers runs `z = post_quant_conv(z)` (a $1\times1$ conv, $32\to32$)
  *before* the decoder. It is a **sibling** of the `decoder.` prefix, so a decoder
  starting at `conv_in` skips it.
- **Fix.** Fold it into `VAEDecoder` (`Config::post_quant_conv_name`), applied at the
  top of `forward`; skip gracefully if absent.
- **Localized by.** Feeding diffusers' ground-truth latent via `--decode_latent` → still
  mush, proving the fault was in the VAE, not the transformer trajectory.
- **Lesson.** Reference pipelines hide steps in unobvious places (a sibling conv).
  Inject a known-good intermediate to localize. (Ch 3, 20, 25)

## B.3 VAE missing batch-norm de-normalization (the patch-grid)

- **Symptom.** A hard $2\times2$ checkerboard across the whole image.
- **Cause.** diffusers standardizes the patch latent with a per-channel BatchNorm and
  inverts it (`z·√(var+ε)+mean`) before decode; this was skipped. Each patch's four
  sub-pixels live in different channels with different stats → mis-scaled → grid.
- **Fix.** Load `bn.running_{mean,var}` and apply the affine to the 128-channel patch
  latent before `unpatchify`.
- **Proved structural by.** Running 28 steps left the grid unchanged (not undercooking).
- **Lesson.** Distinguish *structural* bugs from undercooking with a step-count
  sweep; diff at the latent boundary. (Ch 3, 20, 25)

## B.4 Patchify channel order

- **Symptom.** Coherent but subtly wrong images.
- **Cause.** Packed patch channels as `idx = ph·p·C + pw·C + c` (channel-fastest);
  diffusers uses `idx = c·p·p + ph·p + pw` (channel-slowest). Each token's 128 numbers
  were permuted relative to the weights.
- **Fix.** Use the channel-slowest packing in `patchify.cu`.
- **Lesson.** Layout conventions in data-movement kernels are contracts; a permutation
  degrades rather than destroys, which makes it sneaky. (Ch 3)

## B.5 The NVFP4 CuTe scale-layout trap

- **Symptom.** `Linear` gave correct output on structured inputs (all-ones,
  K-constant) but **uncorrelated** output on random weights.
- **Cause.** Hand-rolled FP4 byte index `byte_idx = k·(N/2)+(n/2)` for column-major B
  assumed flat packing; CUTLASS uses a tiled/swizzled layout. The formula matched only
  column 0, so CUTLASS re-read column-0 bytes for every $k$ — fine for K-constant
  weights, catastrophic for real ones.
- **Fix.** Write through CuTe: `cute::recast_ptr` + `cute::make_tensor(ptr, layout)`,
  assign through the tensor.
- **Lesson.** Never hand-roll sub-byte indexing under a CUTLASS layout. And **component
  tests must use random inputs** — structured tests hid this for a while. (Ch 13, 25)

## B.6 Capture-layer off-by-one ({9,18,27} → {8,17,26})

- **Symptom.** Conditioning subtly off; plausible-but-wrong images.
- **Cause.** diffusers requests `hidden_states[9,18,27]`, but HF's `hidden_states[0]`
  is the embedding and `hidden_states[k]` is the output of layer $k-1$. The real layer
  outputs are `{8,17,26}`.
- **Fix.** `capture_layers = {8,17,26}` in the encoder config.
- **Lesson.** Know your reference's indexing convention exactly. The same off-by-one
  also traps golden comparisons (B.7). (Ch 5, 21)

## B.7 HF `hidden_states` post-norm trap

- **Symptom.** Encoder golden comparison cos dropped to ~0.69 at the last layer.
- **Cause.** Comparing our layer-$N{-}1$ output to HF `hidden_states[N]`, which is the
  *post-final-norm* output, not the layer output — a gamma rotation apart.
- **Fix.** Compare against `hidden_states[N-1]`.
- **Lesson.** Validate against the *right* reference tensor; a low cos can be a
  comparison bug, not a model bug. (Ch 21)

## B.8 Text-stream RoPE missing

- **Symptom.** Image stream worked; prompt conditioning weaker than it should be.
- **Cause.** RoPE was applied only to image Q/K; the text stream's Q/K were left
  unrotated in the double-stream blocks.
- **Fix.** Apply 4-axis RoPE to **both** streams (text positions $(0,0,0,\ell)$).
- **Lesson.** Symmetric components need symmetric treatment; and the header comment
  ("text gets NO RoPE") went **stale** after the fix — *verify code over comments*.
  (Ch 6, 8)

## B.9 Noise init: uniform vs Gaussian

- **Symptom.** Off-manifold, degraded images.
- **Cause.** Initial latent drawn `uniform(-1,1)`; rectified-flow models start from the
  ODE's noise endpoint $\mathcal N(0,\mathbf I)$.
- **Fix.** Draw the initial latent standard-normal.
- **Lesson.** The sampler's start distribution must match what the field was trained to
  transport. (Ch 2, 23)

## B.10 cuDNN 8.9 link + legacy API (the 170× conv)

- **Symptom.** A $128$-ch $1024^2$ $3\times3$ conv took ~510 ms (vs PyTorch's ~3 ms);
  VAE decode 18.7 s.
- **Cause (two).** (1) CMake linked the unversioned `libcudnn.so` → cuDNN **8.9**
  (pre-Blackwell, no sm_121 conv engines). (2) Even on cuDNN 9 the **legacy API** never
  reaches the fast engines — they require the **graph API + NHWC**.
- **Fix.** Link `libcudnn.so.9`; rewrite `conv2d.cu` to the graph API with NHWC/KRSC,
  taking heuristic `config[0]`.
- **Result.** VAE decode 18.7 s → **1.35 s**.
- **Lesson.** Performance is a property of the whole dispatch path. Benchmark new
  dependency paths against a reference — correctness tests won't reveal a wrong-kernel
  dispatch. (Ch 19)

## B.11 CMake CUDA arch auto-probe (sm_75 instead of sm_121)

- **Symptom.** Arch-conditional MMA instructions failed at runtime ("MMA instruction
  used without targeting appropriate compute capability").
- **Cause.** `enable_language(CUDA)` auto-probes and writes `CMAKE_CUDA_ARCHITECTURES`
  to the cache (defaulting to Turing 75 here), so a later `if(NOT DEFINED)` guard never
  takes effect — the binary was built for sm_75.
- **Fix.** `set(CMAKE_CUDA_ARCHITECTURES "121a" CACHE STRING ... FORCE)` **before**
  `enable_language(CUDA)`; wipe `build/` after changing.
- **Lesson.** Pin the GPU arch with FORCE before enabling the CUDA language; a silent
  arch mismatch compiles fine and fails at the MMA. (Ch 13, App C)

## B.12 regO out-of-bounds write (partial tile)

- **Symptom.** Potential corruption / OOB at non-tile-aligned sequence lengths.
- **Cause.** The register-O attention kernel's final normalize/write loop wrote all 16
  rows of a query tile unconditionally; for a partial last tile ($q_0+r \ge S$) that
  writes past the end of $O$.
- **Fix.** Guard with `if (q0 + r < S)` before the write.
- **Lesson.** Tiled kernels need partial-tile guards on *every* global write, not just
  the loads; test non-aligned shapes ($S=130$). (Ch 17)

## B.13 Manual-load `mma` slower than regO

- **Symptom.** A correct, higher-occupancy `mma.sync` attention kernel ran *slower*
  (42.3 ms) than the regO kernel it replaced (34.3 ms).
- **Cause.** Fragments were filled with manual scalar `pack2` reads from shared memory
  — 96 bank-conflict-prone reads per tile, costing more than the rescale tax removed.
- **Fix.** Use `ldmatrix.x2` (conflict-free cooperative load) → 19.7 ms at peak BW.
- **Lesson.** Occupancy is necessary, not sufficient; the smem *access pattern* matters.
  Correctness-first then optimize, but know the optimization (ldmatrix) is required to
  win. (Ch 17, 18)

## B.14 The minor ones

- **`seq_txt` 128 → 512.** Text sequence was 128; diffusers uses 512
  (`max_sequence_length`). (Ch 5)
- **Stale doc comments.** Beyond B.8, the `conv2d.h` and `rope.cu` headers had comments
  that lagged the code. *Verify against the `.cu`.* (Ch 6, 8, 19)
- **Hex-literal typos.** Repeated attempts to spell words in hex test seeds (`0xS1NG...`)
  don't compile — half the alphabet isn't a hex digit. Use real hex (`0xCAFEBABE`) or a
  number.

---

## The patterns across all of them

1. **Plausible-but-wrong is the default failure mode.** Most of these produced
   coherent images, not crashes. Eyeballing the output is not validation.
2. **Convention mismatches dominate.** Slice order (B.1), layer indices (B.6, B.7),
   packing order (B.4), RoPE pairing/streams (B.8), layout (B.5) — porting is mostly
   matching conventions, and conventions are invisible until you diff.
3. **Inject a known-good intermediate.** B.1 (`--embeds`) and B.2 (`--decode_latent`)
   were each localized in one run by replacing our intermediate with the reference's.
4. **Diff at internal boundaries, not just the image.** Latents (B.2, B.3), per-layer
   hidden states (B.6, B.7), per-tensor outputs (B.5) — the closer to the bug you diff,
   the faster you find it.
5. **Distinguish noise from bugs.** Quantization (cos 0.963), reordered-summation (1.8%
   pixel), and BF16 rounding are *expected*; vary precision / compare to FP32 to tell
   them from regressions. (Ch 25)
6. **Performance bugs are silent too.** B.10, B.11 produced *correct* output at terrible
   speed; only a reference *number* revealed them.
7. **Random inputs in tests.** B.5 hid behind structured tests; random data caught it.
