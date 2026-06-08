# Chapter 09 — Single-Stream Blocks

> *Goal of this chapter:* the second block type — the 24 layers that do the bulk of
> the work. Single-stream blocks process the **combined** `[txt ‖ img]` sequence
> with **one** set of weights, and they **fuse** the QKV projection with the MLP
> input, and the attention output with the MLP output, into two big GEMMs computed
> in **parallel**. We walk the forward pass, explain the parallel attention+MLP
> formulation, and show why the fusion is a deliberate bandwidth optimization.
> Anchored to `single_stream_block.{h,cu}` and `kernels/qkv_mlp_split.*`.
>
> *Prerequisites:* Chapters 07–08; the GEMM/`Linear` idea (Chapter 13 for depth).

---

## 9.1 What changes from double to single

After the 8 double-stream blocks, the text and image tokens are concatenated into
one sequence of length $S_{\text{txt}}+S_{\text{img}}$ and handed to **24
single-stream blocks**. Three things change:

1. **One stream, one set of weights.** No more separate text/image projections —
   both modalities flow through the same weights. (They are already "aligned" by
   the double-stream stage; Chapter 04.)
2. **Parallel attention + MLP.** Instead of two sequential sub-layers (attention
   *then* MLP), a single-stream block computes attention and the MLP **from the
   same normalized input, in parallel**, and merges them in a single output
   projection with **one** gated residual.
3. **Fused linears.** The QKV projection and the MLP-input projection are a single
   matrix; the attention-output and MLP-output projections are a single matrix.

This "parallel" transformer block (used by GPT-J, PaLM, and FLUX's single stream)
trades a small amount of expressiveness for a large efficiency win: fewer, bigger
GEMMs and one normalization instead of two.

## 9.2 The forward pass

From `single_stream_block.h`, in-place on the combined sequence `x` (`[B·S, 4096]`):

```
 norm_x  = RMSNorm(x, ones)                       # statistics-only
 mod_x   = modulate(norm_x, scale, shift)         # one AdaLN triple for the block
 fused   = Linear(mod_x, W_qkv_mlp_proj)          # [B·S, 36864]   ← ONE big GEMM
 q,k,v, gate,up = split(fused)                    # 4096,4096,4096, 12288,12288
 q       = RMSNorm(q, norm_q) per head            # QK-norm (learned gains)
 k       = RMSNorm(k, norm_k) per head
 q,k     = RoPE(q,k)                              # 4-axis, combined positions
 attn    = Attention(q, k, v)                     # [B·S, 4096]
 mlp_act = silu(gate) · up                        # [B·S, 12288]   (SwiGLU gate)
 concat  = [attn ‖ mlp_act]                       # [B·S, 16384]
 delta   = Linear(concat, W_out)                  # [B·S, 4096]    ← ONE big GEMM
 x       = gated_residual(x, delta, gate_mod)     # single residual
```

Read the structure: **one** normalize+modulate, **one** input GEMM that produces
*everything* (the attention's Q/K/V *and* the MLP's gate/up), the attention and
SwiGLU computed in parallel from those, **one** output GEMM that consumes *both*
the attention result and the MLP activation, and **one** gated residual. Compared
to a double-stream block's two sub-layers with their two norms and two residuals,
this is markedly leaner — appropriate for 24 repetitions.

## 9.3 The fused projections, decoded

The two weight matrices are large and their column/row splits encode the fusion.
From the architecture (Chapter 04):

**Input projection `to_qkv_mlp_proj` : `[36864, 4096]`.** It maps a 4096-hidden
token to 36864 outputs, sliced as:

$$
36864 = \underbrace{4096}_{Q} + \underbrace{4096}_{K} + \underbrace{4096}_{V}
      + \underbrace{12288}_{\text{gate}} + \underbrace{12288}_{\text{up}}
      = 3\cdot\text{hidden} + 2\cdot\text{ffn} = 9\cdot 4096 .
$$

So a *single* GEMM produces the attention's Q, K, V **and** the MLP's gate and up.
The `split` (in `kernels/qkv_mlp_split.*`) just views the contiguous output as
those five tensors — no copy.

**Output projection `to_out` : `[4096, 16384]`.** It maps a 16384-wide concatenation
back to hidden:

$$
16384 = \underbrace{4096}_{\text{attn out}} + \underbrace{12288}_{\text{mlp act}}
      = \text{hidden} + \text{ffn} = 4\cdot 4096 .
$$

So a *single* GEMM consumes both the attention output and the SwiGLU activation and
produces the block's delta. The `concat` builds the 16384-wide input by placing the
attention result and the MLP activation side by side (again a view/placement, not
real math).

A single-stream block therefore owns just **2 linears** (plus 2 QK-norm gains and
the shared modulation). 24 blocks × 4 tensors = the 96 single-block tensors the
router accounts for (Chapter 14).

## 9.4 Why fuse? The bandwidth argument

This is the first place the Chapter 00 thesis — *everything bends toward
bandwidth* — shows up in the architecture itself, not just the kernels.

On a bandwidth-bound machine, a GEMM's cost is dominated by **reading its weight
matrix from memory**. Two separate projections (say Q-proj and K-proj) read two
weight tensors with two kernel launches, each paying launch overhead and each
streaming its weights. Fusing them into one `[N₁+N₂, K]` matrix reads the *same
total bytes* but in **one** launch, with better memory-access contiguity and one
set of fixed overheads. For the input projection, fusing five projections (Q, K,
V, gate, up) into one GEMM means the **activation** `mod_x` is read **once** from
memory instead of five times, and the output is written as one contiguous block.

The activations are not free either: at $S=4608$, the 4096-wide `mod_x` is ~38 MB
in BF16; reading it once vs five times is ~150 MB of saved traffic per block, per
step. Across 24 blocks × 4 steps that adds up. The model was *trained* with these
fused matrices (the weights are stored fused in the checkpoint), so we are not
choosing to fuse — but the architecture chose to, for exactly this reason, and our
job is to honor it: **don't split the fused weight before using it; feed the whole
matrix to one GEMM.** (Chapter 13 builds the `Linear` that does this; the fused
shapes are why `Linear` must handle large $N$ efficiently.)

## 9.5 Parallel attention + MLP: the subtle part

In a *sequential* block, the MLP sees the residual *after* attention has updated
it. In a *parallel* block, attention and MLP both read the **same** normalized
input `mod_x`, and their results are merged only at the output projection. The two
computations are independent until the final GEMM:

```
              mod_x  (shared)
            ┌────────┴────────┐
            ▼                 ▼
   Q,K,V (from fused)   gate,up (from fused)
   qk-norm, RoPE        silu(gate)·up
   Attention             │
        │                │
        └──── concat ────┘
              │ to_out (fused)
              ▼
            delta → gated_residual
```

This loses the "MLP refines attention's output within the same block" interaction,
but in a deep stack (24 blocks) the next block's attention sees the previous
block's MLP output anyway, so the loss is minor — and the gain (one norm, one
residual, two big GEMMs, more parallelism) is real. It also makes the block a
clean target for the kernel: the attention is exactly the $D=128$ kernel of Part
IV, the SwiGLU is one `silu_mul`, and the rest is two GEMMs and elementwise ops.

## 9.6 Position and the combined sequence

RoPE here uses the **combined** position table (Chapter 06): the single-stream
sequence is text-first then image, so positions are $(0,0,0,\ell)$ for the leading
text tokens and $(0,h,w,0)$ for the trailing image tokens, baked into one
`[seq, head_dim/2]` table. The attention is over the whole combined sequence, so
(as in the double-stream joint attention) image and text tokens continue to mix —
but now through shared weights.

## 9.7 Correctness

Like the double-stream block, the single-stream block has a **zero-gate identity
test**: with `gate_mod = 0` the block is the exact identity, and
`tests/test_single_stream_block.cu` checks `max_err = 0` against the input, proving
the fused split/concat, the parallel attention+MLP wiring, QK-norm, RoPE, and the
residual all connect correctly. A real-weights smoke test loads
`single_transformer_blocks.0` from the F2K file and checks finite, sanely-scaled
output (`mean_abs ≈ 0.77`). This was, historically, the *first* complete MMDiT-block
forward pass to run on real FLUX.2-klein weights on this custom CUDA path.

## 9.8 Where this lives in the code

| Concept | Code |
|---|---|
| the block | `single_stream_block.{h,cu}` (`SingleStreamBlock`) |
| fused QKV+MLP split / attn‖mlp concat | `kernels/qkv_mlp_split.{h,cu}` |
| SwiGLU activation | `kernels/silu_mul.*` / `swiglu.*` |
| attention ($D=128$) | `attention.cu` (Part IV) |
| QK-norm, RoPE | `kernels/rmsnorm.*`, `kernels/rope_4axis.*` |
| tests | `tests/test_single_stream_block.cu` |

## 9.9 Summary and what to carry forward

- Single-stream blocks (24 of them) process the **combined** sequence with **one**
  weight set and a **parallel** attention+MLP structure: one norm, one input GEMM,
  one output GEMM, one gated residual.
- The input GEMM is **fused** `[36864, 4096]` (= Q+K+V+gate+up); the output GEMM is
  **fused** `[4096, 16384]` (= attn‖mlp). `split`/`concat` are views, not copies.
- **Fusion is a bandwidth optimization**: read the shared activation once, stream
  one weight matrix per projection, one launch. Honor it — feed the whole fused
  matrix to one GEMM.
- Same correctness tools as the double-stream block: the **zero-gate identity**
  proves the wiring; a real-weights smoke test checks magnitudes.

Chapter 10 assembles the embedders, the 8+24 block stack, the stream transitions,
and the final projection into the complete `FluxTransformer::forward`, closing Part
II.

---

### Exercises

1. **Decode the shapes.** From `[36864, 4096]` and `[4096, 16384]`, reconstruct the
   five input slices and two output slices, and verify $36864=9\cdot4096$ and
   $16384=4\cdot4096$.
2. **Parallel vs sequential.** State precisely what interaction a parallel block
   gives up versus a sequential one, and argue why 24 stacked blocks make the loss
   small.
3. **Bandwidth math.** Estimate the activation bytes saved per block by fusing the
   five input projections into one GEMM at $S=4608$, hidden 4096, BF16. Multiply by
   24 blocks × 4 steps.
4. **One residual.** A single-stream block has one gated residual but does two
   things (attention + MLP). Where do both contributions enter that single
   residual? (Hint: the output GEMM.)

*Next: [Chapter 10 — Assembling the transformer](10-assembly.md).*
