# Chapter 06 — Rotary Position Embeddings

> *Goal of this chapter:* derive RoPE from first principles — including the
> relative-position property that is its whole point — then extend it to the
> **4-axis** form FLUX uses to give 2D image patches and 1D text tokens distinct
> positions. We pin down the two pairing conventions present in this codebase
> (interleaved for the FLUX transformer, half-rotation for Qwen3), the table
> construction, and why $\theta=2000$. Anchored to
> `kernels/rope_4axis.{h,cu}` and `kernels/rope.{h,cu}`.
>
> *Prerequisites:* Chapter 04; attention's $QK^\top$ (Chapter 16 derives it, but
> "scores are dot products of queries and keys" suffices); 2×2 rotation matrices,
> complex exponentials.

---

## 6.1 The problem: attention is position-blind

Attention computes scores from dot products $q_i\cdot k_j$ of query and key
vectors. Nothing in that operation knows *where* tokens $i$ and $j$ are — permute
the tokens and the set of pairwise scores is unchanged. But position is essential:
"a cat on a dog" ≠ "a dog on a cat", and image patch $(0,0)$ is not patch
$(63,63)$. We must inject position into the tokens.

The classic fix (original Transformer) adds a fixed sinusoidal vector to each
token — *absolute* position, added to the content. RoPE (Su et al., 2021) does
something more elegant: it **rotates** the query and key vectors by an angle
proportional to position, so that the *dot product* of a query and key ends up
depending only on their **relative** position $i-j$. Relative position is what
attention actually wants (how far apart are these tokens), and RoPE bakes it into
the geometry rather than adding it as content.

## 6.2 RoPE in one pair of dimensions

Take a single 2D sub-vector $(x_0, x_1)$ of a query (or key) at sequence position
$m$. RoPE rotates it by angle $m\theta$ for some frequency $\theta$:

$$
\begin{pmatrix} x_0' \\ x_1' \end{pmatrix}
= \underbrace{\begin{pmatrix} \cos m\theta & -\sin m\theta \\ \sin m\theta & \cos m\theta \end{pmatrix}}_{R(m\theta)}
\begin{pmatrix} x_0 \\ x_1 \end{pmatrix}.
$$

It is cleanest in complex notation: identify $(x_0,x_1)$ with $z = x_0 + i x_1$,
and the rotation is just multiplication by a unit complex number,
$z' = e^{i m\theta} z$.

**The relative-position property.** Let a query pair at position $m$ be
$q_m = e^{im\theta} q$ and a key pair at position $n$ be $k_n = e^{in\theta} k$.
The contribution of this pair to the attention score is the real part of
$q_m \overline{k_n}$:

$$
q_m\,\overline{k_n} = e^{im\theta} q \,\overline{e^{in\theta} k}
= e^{i(m-n)\theta}\, q\bar k .
$$

The phase depends only on $m-n$. So **after RoPE, the score from this pair depends
only on the relative offset $m-n$ and the (rotation-invariant) content interaction
$q\bar k$.** No absolute position leaks in; translate both tokens by the same
amount and the score is unchanged. This is exactly the inductive bias attention
wants, achieved with zero added parameters and zero added content — purely a
rotation.

## 6.3 RoPE across a full head: a bank of frequencies

A head has $D$ dimensions, not 2. RoPE splits them into $D/2$ pairs and gives each
pair its own frequency, geometrically spaced (the same idea as sinusoidal
embeddings):

$$
\theta_k = \text{base}^{-2k/D}, \qquad k = 0,1,\dots,\tfrac{D}{2}-1 .
$$

Low-$k$ pairs rotate fast (encode fine, short-range position); high-$k$ pairs
rotate slowly (encode coarse, long-range position). At position $m$, pair $k$ is
rotated by $m\,\theta_k$. The per-position rotation angles are precomputed into
**cos/sin tables** of shape $[\text{seq}, D/2]$ — exactly the tables the kernels
take.

> **`base` / $\theta$ choice.** Qwen3 (text) uses $\text{base}=10^6$; the FLUX
> transformer uses **$\text{base}=2000$** (`rope_theta=2000`, not the usual
> 10000). A smaller base makes all frequencies higher → faster rotation per unit
> position. Image patch grids are small (e.g. 64×64) compared to LLM context
> lengths, so a base tuned for thousands-of-tokens text would barely rotate across
> a 64-wide grid; $\theta=2000$ spreads the rotations sensibly over the patch
> range. This single number must match the trained model exactly — it is wired as
> a parameter, not hard-coded to 10000.

## 6.4 Two pairing conventions (both live in this repo)

"Split $D$ into pairs" hides a convention choice: *which* two coordinates form a
pair. Two are common, and **this codebase contains both**, deliberately:

- **Interleaved `(2k, 2k+1)`** — adjacent coordinates are paired. This is the
  FLUX2 / diffusers convention (`apply_rotary_emb` with `use_real_unbind_dim=-1`).
  Implemented in `rope_4axis.cu`:

  ```
  x'[2k]   = x[2k]·cos - x[2k+1]·sin
  x'[2k+1] = x[2k]·sin + x[2k+1]·cos
  ```

- **Half-rotation `(i, i+D/2)`** — coordinate $i$ is paired with the one $D/2$
  away. This is the LLaMA / Qwen3 convention. Implemented in `rope.cu`:

  ```
  x'[i]      = x[i]·cos - x[i+D/2]·sin
  x'[i+D/2]  = x[i]·sin + x[i+D/2]·cos
  ```

The two are related by a fixed permutation of the channel order and are *not*
interchangeable — applying the wrong one rotates the wrong coordinate pairs and
silently corrupts the geometry. The FLUX transformer's Q/K use the **interleaved**
kernel; the Qwen3 encoder (Chapter 21) uses the **half-rotation** one. (The header
comment in `rope.cu` was once wrong about which it implemented — corrected after
verifying against HF; Appendix B.)

## 6.5 From 1D to 4 axes: positioning a 2D image

Text is 1D (token index $\ell$). An image is 2D (patch row $h$, patch column $w$).
A single scalar position cannot distinguish patch $(3,5)$ from patch $(5,3)$. FLUX
solves this with **multi-axis RoPE**: partition the head's $D=128$ dimensions into
**4 axis-groups** and give each group its own position coordinate. From the config,
`axes_dims_rope = [32, 32, 32, 32]` — four axes labeled $(T, H, W, L)$:

| axis | width | meaning | image token | text token |
|---|---|---|---|---|
| **T** | 32 | temporal/frame | 0 | 0 |
| **H** | 32 | patch row | $h$ | 0 |
| **W** | 32 | patch column | $w$ | 0 |
| **L** | 32 | sequence index | 0 | $\ell$ |

Each axis applies ordinary RoPE within its 32 dims, using that axis's coordinate.
The elegance: **image tokens carry position only in $(H,W)$**, while **text tokens
carry position only in $L$** — they live in *orthogonal* position subspaces. So an
image token at $(0,h,w,0)$ and a text token at $(0,0,0,\ell)$ are positioned
without interfering, yet share the same head and attend jointly (Chapter 08). The
$T$ axis is reserved for video/frames and is 0 here.

```
 head_dim = 128  =  [ T:32 | H:32 | W:32 | L:32 ]
                       │      │      │      │
 image patch (h,w):    0      h      w      0     → rotates in H,W subspaces
 text token  ℓ    :    0      0      0      ℓ     → rotates in L subspace
```

This is built by `build_rope_4axis_tables(axes_dim, token_positions, theta, ...)`:
the host passes, for each token, its $[T,H,W,L]$ coordinates; the helper computes
the per-axis frequencies and bakes the per-token rotation into flat
`[seq, head_dim/2]` cos/sin tables. The device kernel `rope_4axis_inplace_bf16`
then doesn't need to know about axes at all — it just applies the interleaved
rotation using the prebaked tables. **Three tables are built per forward pass**: one
for the image-stream length (image positions), one for the text-stream length (text
positions), and one for the combined single-stream sequence.

## 6.6 Where RoPE is applied in the model

RoPE rotates the **queries and keys** (not values) just before attention, inside
each block:

- **Double-stream blocks (Chapter 08):** RoPE is applied to **both** the image
  Q/K (with image positions $(0,h,w,0)$) **and** the text Q/K (with text positions
  $(0,0,0,\ell)$). An earlier version skipped RoPE on the text stream; adding it
  changed the image and was one of the fixes on the road to working text
  conditioning (Appendix B).
- **Single-stream blocks (Chapter 09):** applied to the combined sequence's Q/K
  with the combined position table (text-first, then image).
- **Qwen3 encoder (Chapter 21):** uses 1D half-rotation RoPE on its Q/K, with its
  own $\theta=10^6$, applied *after* the per-head q/k-norm (a Qwen3 quirk).

RoPE acts on Q and K only because the relative-position phase must enter the score
$q_i\cdot k_j$; values carry content that is summed by the (already
position-aware) attention weights, so they are left unrotated.

## 6.7 Implementation notes

- **Tables are FP32, data is BF16.** The cos/sin tables are kept in FP32 for
  precision (the rotations compose multiplicatively over long sequences); the Q/K
  data is BF16. The kernel reads FP32 angles and writes BF16 rotated values
  in-place.
- **In-place, one CTA per row.** `rope_4axis_inplace_bf16` rotates each
  $[\text{head\_dim}]$ row in place; `batch_rows = batch · seq · n_heads`. It is a
  cheap, memory-bound elementwise kernel — negligible next to the GEMMs and
  attention.
- **Correctness.** `tests/test_rope_4axis.cu` checks the kernel's interleaved
  rotation against an FP32 host reference; `tests/test_rope.cu` does the same for
  the half-rotation variant. The 4-axis math (interleaved $(2k,2k+1)$, per-axis
  concat layout, $\theta^{-2k/d_a}$ frequencies, positions img $=(0,h,w,0)$ with
  flat index $s=h\cdot W+w$, txt $=(0,0,0,\ell)$, single-stream text-first) was
  verified against diffusers during the text-path debugging (Chapter 25).

## 6.8 Summary and what to carry forward

- RoPE injects position by **rotating Q/K pairs by angles proportional to
  position**; the dot-product then depends only on **relative** position — the bias
  attention wants, with no parameters and no added content.
- A head uses a **bank of geometric frequencies** $\theta_k=\text{base}^{-2k/D}$;
  FLUX's base is **2000**, tuned to image-patch ranges (not 10000).
- Two pairing conventions exist here: **interleaved $(2k,2k+1)$** for the FLUX
  transformer, **half-rotation $(i,i+D/2)$** for Qwen3 — not interchangeable.
- **4-axis RoPE** $(T,H,W,L)$ positions 2D image patches in $(H,W)$ and text tokens
  in $L$, in orthogonal subspaces, so both can share joint attention; tables are
  prebaked per token by `build_rope_4axis_tables`.
- Applied to Q/K only, in every block (text stream included), and in the encoder.

Chapter 07 covers the other thing injected into every block — the **timestep**, via
AdaLN modulation — completing the inputs a block needs before we assemble the
blocks themselves in Chapters 08–09.

---

### Exercises

1. **Relative position.** Re-derive $q_m\overline{k_n}=e^{i(m-n)\theta}q\bar k$ and
   state in words why this makes attention scores translation-invariant.
2. **Base intuition.** For $D=32$ and a 64-wide axis, compare the rotation of the
   slowest pair ($k=D/2-1$) across positions $0\to63$ under base 10000 vs 2000.
   Argue why 2000 is better for image grids.
3. **Convention swap.** You apply the half-rotation kernel to FLUX Q/K (which want
   interleaved). Which coordinate pairs get rotated, and why does the image come
   out wrong rather than catastrophically broken?
4. **Orthogonal axes.** Explain why putting image position in $(H,W)$ and text
   position in $(L)$ lets the two modalities share a head without their positions
   interfering. What would break if text tokens also had nonzero $H$?

*Next: [Chapter 07 — Modulation / AdaLN](07-modulation.md).*
