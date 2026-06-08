# Chapter 17 — The Kernel Journey

> *Goal of this chapter:* tell the full optimization story of the $D=128$ attention
> kernel — five implementations, each fixing the bottleneck the last one exposed —
> from a naïve row loop to a `mma.sync`+`ldmatrix` kernel pinned to **peak HBM
> bandwidth**. Every step has a measured number and a *reason*. The throughline is a
> lesson in finding the real bottleneck before optimizing: the biggest win came from
> diagnosing an **occupancy** limit that "obvious" optimizations would have missed.
> Anchored to `attention.cu` and `tools/bench_attention.cu`.
>
> *Prerequisites:* Chapter 16 (online softmax, tiling), 11 (roofline), 13 (tensor
> cores).

---

## 17.1 The scoreboard

All numbers are the GB10 at the real transformer shape **$S=4608, H=32, D=128$**
(1024px joint attention), measured with `tools/bench_attention.cu`:

| # | kernel | per-call | HBM | vs prev | the idea |
|---|---|---|---|---|---|
| 1 | `attention_row_kernel` | — (baseline B) | $O(S^2D)$ re-reads | — | one CTA per query |
| 2 | `flash_attn_kernel` | (4.1× denoise vs row) | streams K/V | — | warp-per-query + online softmax |
| 3 | `flash_wmma_kernel` | **56.5 ms** | 97.7 GB/s (36%) | 1.45× denoise | tensor-core QK^T/PV |
| 4 | `flash_wmma_regO_kernel` | **34.3 ms** | 160 GB/s | **1.65×** | register-resident O → 2 CTA/SM |
| 5 | `flash_mma_kernel` | **19.7 ms** | **279 GB/s (~peak)** | **1.74×** | `mma.sync` + `ldmatrix` |

End to end, kernel 5 vs the committed kernel 3 took the **denoise loop from 8.41 s
to 6.54 s** at 1024px (the difference is exactly $128$ attention calls × the
per-call saving). Total speedup #3→#5: **2.87×**. The rest of this chapter is *how*
and, more importantly, *why each step*.

## 17.2 Step 1→2: stop re-reading K/V (flash)

The row kernel (Chapter 16, "naïve B") computes one query per CTA and re-reads all
of K and V from HBM for each query — $O(S^2D)$ HBM traffic, the memory wall.

`flash_attn_kernel` applies Chapter 16's two ideas: **online softmax** (one pass,
$O(D)$ state) and **query tiling** (8 warps/CTA, one query per warp, so 8 queries
share each K/V tile loaded into shared memory). K/V is read from HBM ~8× fewer
times. Each lane owns $D/32$ dims of its query in registers; scores are a warp
dot-reduce; the running $(m,\ell,\text{acc})$ live in registers; the smem holds only
the current `TILE_K` tile, so even the VAE's $S=16384$ fits. **Denoise 114 s → 28 s
(4.1×)** at 1024px. This kernel is still the path for $D\ne128$ (e.g. the VAE's
$D=512$), where its register layout fits but the tensor-core tiling below does not.

## 17.3 Step 2→3: put the matmuls on tensor cores (WMMA)

The warp-per-query kernel computes $q\cdot k$ with warp shuffles — CUDA cores, not
tensor cores. For $D=128$ the dot products are exactly what tensor cores are built
for. `flash_wmma_kernel` is FlashAttention-2-style: **$B_M=64$ queries** per CTA (4
warps × 16), **$B_N=48$** keys per K/V tile, $QK^\top$ and $P V$ on tensor cores
(`wmma`, BF16 MMA, FP32 accumulate), online softmax rescaling the FP32 $O$
accumulator held in shared memory. ($B_N=48$ because 64 overflows the 99 KiB smem
cap.) A non-obvious tuning detail: the **softmax epilogue** (the `exp`, the O-rescale,
the PV add-back), not the matmul, was the bottleneck, so it is spread across all 32
lanes via flat indexing rather than left on the 16 row-owning lanes. **Denoise 28 s
→ 19.25 s (1.45×)**; per-call **56.5 ms**. This was the committed baseline at the
start of the session.

## 17.4 The diagnosis that mattered: occupancy, not bandwidth

Here is the pivotal moment, and the methodological lesson of the whole chapter.

At 56.5 ms the WMMA kernel runs at **97.7 GB/s — only ~36 % of the GB10's 273 GB/s
peak — and 6.2 TFLOP/s.** It is nowhere near the memory ceiling *or* a compute
ceiling. So it is neither bandwidth-bound nor compute-bound; it is
**utilization/occupancy-bound**. The "obvious" next optimizations — *bigger query
tiles for more K/V reuse*, *persistent-CTA scheduling* — both assume a
bandwidth/launch bottleneck the kernel **does not have**. They would have been
wasted effort.

The real cause, found by smem accounting + `cudaOccupancyMaxActiveBlocksPerMultiprocessor`:
the kernel uses **93 KiB of shared memory per CTA** (Ks + Vs + Qs + Ps + the FP32
Ss + Os + per-row stats). Against the 99 KiB opt-in cap that allows exactly **1 CTA
per SM** — 4 warps on an SM that can hold ~48. With only one CTA resident there is
**no latency hiding**: when the barrier-heavy softmax epilogue stalls, the SM has
nothing else to run. The kernel is starved of parallelism, not bandwidth.

> **The lesson.** Measure *which* ceiling you are under before optimizing. The
> profiler (GB/s, TFLOP/s, occupancy) said "occupancy," and the lever that follows
> from "occupancy-bound" is **cut shared memory to fit more CTAs** — the opposite of
> "use bigger tiles." Chasing the intuitive-but-wrong lever is the most common way
> to waste optimization effort.

## 17.5 Step 3→4: register-resident O (regO)

To raise occupancy we must shrink smem. The biggest consumers are the FP32 output
accumulator `Os[BM][D]` (32 KiB) and the staged queries `Qs[BM][D]` (16 KiB).
`flash_wmma_regO_kernel` removes both: hold the **O accumulator in registers** (the
`wmma` accumulator fragments) and load **Q straight from global** into its
fragments. Shared memory drops **93 KiB → 43 KiB → 2 CTAs/SM** (8 warps), doubling
occupancy.

Result: **56.5 ms → 34.3 ms (1.65×)**, 97.7 → 160 GB/s, **bit-exact** (cos_vs_prod
1.00000). The win comes purely from latency hiding — two CTAs means when one stalls
on the epilogue the other runs. Productionizing it took one fix: the final
normalize/write loop wrote all 16 rows of a query tile unconditionally, which is an
**out-of-bounds write for a partial last tile** (when $q_0+r \ge S$); guarding with
`if (q0 + r < S)` made it correct for non-tile-aligned $S$ (e.g. $S=130$ in
`test_attention`).

**The rescale tax.** regO pays a price for hiding O in `wmma` fragments: `wmma`
*hides the fragment's row→lane mapping*, so when online softmax needs to multiply
each row of O by its correction factor, the kernel cannot index "row $r$" in
registers. It must **store the fragment to smem, scale per row there, and reload it**
— per K/V tile, per output column-tile. (The *PV add-back* is free — both
accumulators share the hidden layout, so `ofrag.x[i] += acc.x[i]` works without
knowing rows — but the per-row *rescale* does not.) This store/scale/reload is the
"rescale tax," and removing it is the next step.

## 17.6 Step 4→5: `mma.sync` + `ldmatrix` (peak)

`wmma` is a convenience wrapper with opaque fragment layouts. Dropping to **raw
`mma.sync` PTX** (Chapter 18) exposes the *documented* thread→register layout, which
unlocks three things at once:

1. **The rescale tax disappears.** With a known layout, each thread knows which two
   rows of O its accumulator registers hold, so the per-row correction is a plain
   **register multiply** — no smem store/reload.
2. **The softmax leaves smem.** Row max/sum become **register reductions across the
   4 lanes that share a row** (`__shfl_xor`), so the score matrix never touches
   shared memory.
3. **No P round-trip.** Because $P V$'s contraction dim (the keys) equals $QK^\top$'s
   N dim, and the `mma` C-fragment and A-fragment share the same row/column mapping,
   the $QK^\top$ result registers **repack directly** into the $P V$ A-fragments (cast
   to BF16) — the probabilities never go to smem either.

The inner loop now touches shared memory **only** for the unavoidable K/V
global→smem staging. Smem drops to 24 KiB.

**But correctness-first cost speed first.** The first `mma.sync` version loaded the
K/V fragments with *manual scalar `pack2` reads* from smem — correct (bit-exact, all
`test_attention` shapes incl. $S=130$), higher occupancy (3 CTA/SM), but **42.3 ms —
slower than regO's 34.3.** The 96 scalar bank-conflict-prone smem reads per tile
cost more than the rescale tax they removed. The unlock was **`ldmatrix`** — the
warp-cooperative, conflict-free shared-memory→fragment load that `wmma` had been
using internally all along. Swapping the manual loads for `ldmatrix.x2` (with the
address recipes validated in Chapter 18) dropped it to **19.7 ms at 279 GB/s** —
essentially the **memory ceiling** (279 ≳ 273 GB/s). The kernel is now bandwidth-
bound; there is nothing left to win, because the K/V bytes cannot arrive faster.

```
 56.5 ms  WMMA           1 CTA/SM   97.7 GB/s   ── occupancy-bound
 34.3 ms  regO           2 CTA/SM   160  GB/s   ── + register O (rescale tax remains)
 42.3 ms  mma (manual)   3 CTA/SM   130  GB/s   ── correct but scalar loads dominate
 19.7 ms  mma + ldmatrix 2 CTA/SM   279  GB/s   ── BANDWIDTH-BOUND (peak)  ✔ production
```

(Note the final kernel is *back* to 2 CTA/SM, lower than the manual version's 3 —
register pressure rose — but it does not matter: at the memory ceiling, occupancy
beyond what hides latency buys nothing.)

## 17.7 The methodology, distilled

The journey is a template for kernel optimization on a bandwidth-bound machine:

1. **Eliminate redundant HBM traffic first** (row→flash): the algorithm (online
   softmax + tiling) beats any micro-optimization of a wasteful one.
2. **Use the right compute units** (flash→WMMA): tensor cores for the matmuls.
3. **Measure which ceiling you are under** (WMMA diagnosis): GB/s vs peak, TFLOP/s
   vs peak, and `cudaOccupancyMaxActiveBlocks`. *Do not* optimize for a bottleneck
   you have not confirmed.
4. **Pull the right lever** (occupancy → cut smem → regO): the diagnosis dictates
   the lever; here "smaller smem," the opposite of the intuitive "bigger tiles."
5. **Validate every step against an independent oracle and bit-exactly where
   possible** (`test_attention` vs FP32, cos_vs_prod, the end-to-end image A/B): a
   faster wrong kernel is worthless, and reordered-summation BF16 drift (1.8%
   image diff mma-vs-regO) must be distinguished from bugs (both match FP32 at
   cos 1.0).
6. **Know when to stop** (279 GB/s ≈ peak): declare victory at the ceiling, not when
   the code stops being improvable in the abstract.

`bench_attention.cu` is the instrument: it prints per-shape latency, the K/V-reread
HBM model in GB/s, TFLOP/s, and (via the probes) the achieved CTAs/SM — exactly the
numbers each decision above needed.

## 17.8 What didn't make the cut (and why)

- **Bigger query tiles / persistent CTAs** — rejected by the occupancy diagnosis
  (§17.4); they target a bottleneck the kernel doesn't have.
- **`ldmatrix.x4` to halve load instruction count**, **K/V double-buffering** —
  possible micro-opts, but the kernel is already at peak HBM bandwidth, so they
  cannot help a memory-bound kernel meaningfully. Noted, not pursued.
- **A bespoke kernel for $D=512$ (VAE)** — the warp-per-query flash kernel already
  serves it; its smem is too large for the $B_M{=}64$ tensor-core tiling, and the
  VAE decode is dominated by convolutions (Chapter 19), not attention, so it wasn't
  worth a special kernel.

## 17.9 Where this lives in the code

| kernel | symbol in `attention.cu` |
|---|---|
| row fallback | `attention_row_kernel` |
| warp-per-query flash | `flash_attn_kernel<DPT>` (D∈{64,256,512}) |
| WMMA (superseded) | `flash_wmma_kernel<D>` (kept for bench A/B) |
| register-O WMMA (superseded) | `flash_wmma_regO_kernel<D>` (kept for bench A/B) |
| **production $D=128$** | `flash_mma_kernel<D>` (mma.sync + ldmatrix) |
| dispatch | `Attention::forward` (D=128 → mma; D∈{64,256,512} → flash; else row) |
| benchmark/probes | `tools/bench_attention.cu`, `launch_*_probe` |

## 17.10 Summary and what to carry forward

- The kernel went **56.5 → 19.7 ms** (2.87×) by, in order: streaming K/V (flash),
  tensor cores (WMMA), **raising occupancy by cutting smem** (regO), and removing
  smem round-trips with `mma.sync`+`ldmatrix` (peak bandwidth).
- The decisive insight was a **profiling diagnosis**: WMMA was occupancy-bound (1
  CTA/SM from 93 KiB smem), so the lever was *less* shared memory, not bigger tiles.
- The `mma.sync` kernel removes the **rescale tax**, the smem softmax, and the **P
  round-trip** (QK^T registers repack into PV fragments); **`ldmatrix`** made its
  fragment loads cheap enough to hit **279 GB/s ≈ peak**.
- Method: eliminate redundant HBM → right compute units → **measure the ceiling** →
  pull the matching lever → validate bit-exactly → **stop at the ceiling**.

Chapter 18 opens the hood on the `mma.sync` and `ldmatrix` PTX — the exact fragment
layouts and the standalone harness (`mma_unit.cu`) that validated them before they
ever touched the attention kernel.

---

### Exercises

1. **Why 36 % is the tell.** Explain why 97.7 GB/s (≈36 % of peak) and 6.2 TFLOP/s
   *together* imply occupancy-bound rather than bandwidth- or compute-bound.
2. **smem → occupancy.** From 93 KiB/CTA and the 99 KiB cap, derive 1 CTA/SM; from
   43 KiB derive 2. Why does 2 CTA/SM hide the epilogue latency that 1 cannot?
3. **The rescale tax.** Explain why holding O in `wmma` fragments forces a
   smem store/scale/reload for the per-row correction, while the PV add-back is
   free. What property of `mma.sync` removes the tax?
4. **Manual vs ldmatrix.** The manual-load `mma` kernel had *higher* occupancy yet
   was *slower* than regO. Reconcile this with §17.4's "raise occupancy" lesson.
   (Hint: occupancy is necessary, not sufficient; what dominated?)

*Next: [Chapter 18 — `mma.sync` & `ldmatrix` up close](18-mma-ldmatrix.md).*
