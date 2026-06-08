# Chapter 16 — Attention and the Memory Wall

> *Goal of this chapter:* derive scaled dot-product attention, quantify *why* a
> naïve implementation is a memory-bandwidth disaster on the GB10, and derive the
> **online softmax** that lets attention be computed in a single streaming pass
> without ever materializing the $S\times S$ score matrix. This is the
> numerical foundation for the kernel journey of Chapter 17. Anchored to
> `attention.h` and the fallback kernel in `attention.cu`.
>
> *Prerequisites:* Chapters 04/08–09 (where attention sits), 11 (roofline);
> softmax, basic numerical stability.

---

## 16.1 Scaled dot-product attention

Attention mixes a sequence of tokens by letting each token (the *query*) gather a
weighted average of all tokens' *values*, weighted by query-key similarity. For
queries $Q\in\mathbb R^{S\times D}$, keys $K\in\mathbb R^{S\times D}$, values
$V\in\mathbb R^{S\times D}$ (one head, $S$ tokens, head-dim $D$):

$$
\text{Attn}(Q,K,V) = \underbrace{\text{softmax}\!\Big(\tfrac{QK^\top}{\sqrt D}\Big)}_{P\ \in\ \mathbb R^{S\times S}} V .
$$

Step by step:

1. **Scores** $S_{ij} = \tfrac{1}{\sqrt D}\, q_i\cdot k_j$ — how much query $i$
   attends to key $j$. The $1/\sqrt D$ keeps the dot products from growing with $D$
   (without it, variance scales with $D$ and the softmax saturates).
2. **Weights** $P_{ij} = \text{softmax}_j(S_{ij}) = \dfrac{e^{S_{ij}}}{\sum_{j'} e^{S_{ij'}}}$
   — a probability distribution over keys, per query.
3. **Output** $O_i = \sum_j P_{ij}\, v_j$ — query $i$'s output is the
   $P$-weighted average of all values.

In the model this runs per head ($H=32$) and the head outputs are concatenated.
The $1/\sqrt D$ is the `scale` in `Attention::Config`; QK-norm (Chapter 08) further
stabilizes the scores before this.

## 16.2 The cost, and why it is a wall

The score matrix $P$ is $S\times S$. Two costs scale with $S^2$:

- **Compute:** $QK^\top$ is $2S^2D$ FLOPs, $PV$ another $2S^2D$ — total $4S^2D$ per
  head.
- **Memory (the killer):** materializing $P$ is $S^2$ values. At 1024px the
  transformer's joint attention has $S=4608$, so $P$ is $4608^2 \approx 21$M
  entries **per head**, $\times 32$ heads. In FP32 that is ~2.7 GB *per layer*,
  written to and read back from HBM. The VAE mid-block attention is worse:
  $S=16384 \Rightarrow S^2 \approx 268$M.

A naïve "materialize $P$, then multiply by $V$" attention is therefore dominated by
**writing and re-reading the $S\times S$ scores through HBM** — pure memory
traffic, the exact thing the GB10 cannot afford (Chapter 11: attention measured at
$I\approx63$ FLOP/byte, deeply memory-bound). The arithmetic is cheap; the bytes
are everything.

There is a second, subtler trap. The row kernel that *avoids* materializing $P$ by
recomputing — one CTA per query, looping over all keys — re-reads **all of K and V
from HBM for every query**. That is $O(S^2 D)$ HBM traffic too, just in K/V reads
instead of $P$ writes. Either way, naïve attention re-touches $O(S^2)$ data in HBM.

```
 naïve A: write P [S×S] to HBM, read it back ........  O(S²) HBM traffic
 naïve B (row): re-read all K,V per query ............  O(S²·D) HBM traffic
 GOAL: touch K,V once each, never materialize P ......  O(S·D) HBM traffic
```

The project's `attention_row_kernel` (the fallback in `attention.cu`) is naïve B:
one CTA per $(b,h,q)$, scores for that query held in smem, block-reduced softmax,
then $O=\sum P\cdot V$. It is correct and simple (cos 1.00000 vs FP32) and fine for
short sequences, but its K/V re-reads are the $O(S^2D)$ HBM cost that dominates at
transformer scale. Eliminating those re-reads is what Chapter 17 is about — and it
requires computing the softmax *without* having all the scores at once.

## 16.3 The obstacle: softmax needs the whole row

To tile attention — process keys in blocks, keeping only a block in fast memory —
we hit a wall: **softmax is a global operation over the whole row of scores.** The
denominator $\sum_{j'} e^{S_{ij'}}$ needs every key, and the numerically-safe form

$$
P_{ij} = \frac{e^{S_{ij} - m_i}}{\sum_{j'} e^{S_{ij'} - m_i}}, \qquad m_i = \max_{j'} S_{ij'}
$$

needs the **row max** $m_i$ first (subtracting it prevents $e^{\text{large}}$
overflow — without it, scores ~30 overflow FP32's exp). So the textbook algorithm
is two passes over the keys: one to find $m_i$ and the sum, one to weight $V$. Two
passes means reading K (and V) twice — back to the memory wall.

The breakthrough — **online softmax** (Milakov & Gimelshein 2018; the heart of
FlashAttention, Dao et al. 2022) — computes the softmax-weighted output in a
**single pass** over the keys, maintaining running statistics and *correcting* them
as new blocks arrive.

## 16.4 Online softmax, derived

Process keys in order (or in tiles). Maintain three running quantities for a query:

- $m$ — the max score seen *so far*,
- $\ell$ — the sum $\sum e^{S_j - m}$ over keys seen so far (relative to the current
  $m$),
- $O$ — the running weighted value sum $\sum e^{S_j - m}\, v_j$ (also relative to the
  current $m$).

When a new key $j$ with score $s = S_{ij}$ arrives, the max may increase to
$m' = \max(m, s)$. Everything accumulated so far was scaled by $e^{-m}$; to put it
on the new scale $e^{-m'}$ we multiply by the **correction factor**

$$
\text{corr} = e^{m - m'} \;\le\; 1 .
$$

Then we fold in the new key. The single-key update is:

$$
m' = \max(m, s),\qquad
\ell' = \ell\cdot\text{corr} + e^{s - m'},\qquad
O' = O\cdot\text{corr} + e^{s - m'}\, v_j .
$$

**Why this is exact.** Define $f_j = e^{S_j - m_{\text{final}}}$ for the final max.
The true output is $\frac{\sum_j f_j v_j}{\sum_j f_j}$. By induction, after each
update $O = \sum_{j\le t} e^{S_j - m_t} v_j$ and $\ell = \sum_{j\le t} e^{S_j - m_t}$
with $m_t$ the running max — the correction $e^{m_{t-1}-m_t}$ exactly rebases the
previous partial sums from scale $m_{t-1}$ to $m_t$, because
$e^{S_j-m_{t-1}}\cdot e^{m_{t-1}-m_t} = e^{S_j-m_t}$. At the end, divide:
$O_i = O/\ell$. No score is ever overflowed (always $s-m'\le 0$), and the whole row
of scores is never stored — only $(m,\ell,O)$, which is $O(D)$, not $O(S)$.

```
 for each key tile:
   load K,V tile into shared memory          (read each K,V byte ONCE)
   for each key j in tile:
     s    = scale · (q · k_j)
     m'   = max(m, s);   corr = exp(m - m')
     p    = exp(s - m')
     l    = l·corr + p
     O    = O·corr + p·v_j                    (rescale running output, add new)
     m    = m'
 O_i = O / l                                  (normalize once at the end)
```

This is the algorithmic core of every fast attention kernel in Part IV. It turns
attention from "materialize an $S\times S$ matrix" into "stream K/V once, keep
$O(D)$ state per query." The HBM traffic drops from $O(S^2)$ to $O(S\cdot D)$ — the
goal of §16.2 — and the smem footprint is bounded by the tile, not by $S$ (so even
the VAE's $S=16384$ fits).

## 16.5 Tiling: sharing K/V across queries

Online softmax removes the *re-read across passes*, but a per-query kernel still
loads each K/V tile once **per query**. The second idea is to process a **block of
queries together** so they **share** each K/V tile load from HBM. If $B_M$ queries
share a tile, K/V is read from HBM $B_M\times$ fewer times. This is why the fast
kernels use query tiles ($B_M=64$ in the WMMA/mma kernels) — the K/V tile is
streamed into shared memory once and reused by all 64 queries in the CTA before the
next tile is loaded. Combined with online softmax, K and V are read from HBM
essentially **once each**, which is the bandwidth-optimal lower bound. Chapter 17 is
the story of realizing this efficiently on tensor cores.

## 16.6 The interface

`Attention::Config` is `{batch, seq, n_heads, head_dim, scale}` with tensors laid
out `[batch, seq, n_heads, head_dim]` BF16 (the same layout Q/K/V come out of the
block projections). One `forward(Q,K,V,O,stream)` call. The contract is identical
across all the kernel implementations of Chapter 17 — they are interchangeable
behind this interface, which is exactly what let the project swap row → flash →
WMMA → regO → mma.sync while keeping the rest of the model unchanged and re-running
the same correctness test (`test_attention`, cos vs FP32). The same interface
serves the transformer ($D=128$) and the VAE ($D=512$, $S$ up to 16384).

## 16.7 Summary and what to carry forward

- Attention is $O=\text{softmax}(QK^\top/\sqrt D)\,V$; the score matrix $P$ is
  $S\times S$, so both compute and (especially) memory scale as $S^2$.
- Naïve attention is a **memory wall**: either materialize $P$ ($O(S^2)$ writes) or
  re-read K/V per query ($O(S^2D)$ reads). On the GB10 attention is decisively
  memory-bound ($I\approx63$).
- **Online softmax** computes the softmax-weighted output in **one streaming pass**,
  keeping running $(m,\ell,O)$ and rescaling by $\text{corr}=e^{m-m'}$ as new keys
  arrive — exact, overflow-safe, $O(D)$ state, no $P$ materialization.
- **Query tiling** makes a block of queries share each K/V tile load, driving HBM
  traffic to the $O(S\cdot D)$ lower bound — K/V read once each.
- All kernels sit behind one `Attention::forward` interface, so they are swappable
  and tested identically.

Chapter 17 takes these two ideas (online softmax + query tiling) and walks the five
kernels that implement them with increasing efficiency — from the row fallback to a
`mma.sync`+`ldmatrix` kernel pinned to peak HBM bandwidth.

---

### Exercises

1. **The $S^2$ bill.** Compute the size of $P$ (one head and all 32) at $S=4608$ in
   FP32 and BF16, and the HBM time to write+read it at 273 GB/s. Compare to the
   measured ~19.7 ms full attention call (Chapter 17).
2. **Online softmax exactness.** Prove by induction that after processing keys
   $1..t$, $O=\sum_{j\le t} e^{S_j-m_t}v_j$ and $\ell=\sum_{j\le t}e^{S_j-m_t}$, and
   hence $O/\ell$ is the exact softmax output at the end.
3. **Overflow.** Show that without the running-max subtraction, a score of 90
   overflows FP32 `expf`, and that online softmax never evaluates $e^{>0}$.
4. **Tiling win.** With $B_M$ queries per CTA sharing K/V tiles, write the HBM K/V
   read traffic as a function of $S, D, B_M$ and explain why $B_M=64$ helps but
   $B_M\to\infty$ is limited by smem/registers (foreshadowing Chapter 17).

*Next: [Chapter 17 — The kernel journey](17-attention-journey.md).*
