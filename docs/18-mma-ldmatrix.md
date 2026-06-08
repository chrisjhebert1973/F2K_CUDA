# Chapter 18 — `mma.sync` & `ldmatrix` Up Close

> *Goal of this chapter:* open the hood on the two PTX instructions that made the
> peak-bandwidth attention kernel possible — the `mma.sync` tensor-core
> multiply-accumulate and the `ldmatrix` cooperative shared-memory load. We give the
> exact thread→register fragment layouts, show how knowing them removes the rescale
> tax and the P round-trip (Chapter 17), and walk the **validation methodology**
> (`tools/mma_unit.cu`) that locked every layout bit-exactly *before* it touched the
> attention kernel. This is the most hardware-level chapter; it is also a reusable
> recipe for hand-writing tensor-core kernels. Anchored to `attention.cu` and
> `tools/mma_unit.cu`.
>
> *Prerequisites:* Chapter 17; CUDA warps/lanes, inline PTX, shared memory.

---

## 18.1 Why drop below `wmma`

`wmma` (the `nvcuda::wmma` C++ API) is convenient but treats fragments as **opaque**
— you cannot know which register of which lane holds matrix element $(r,c)$. That
opacity is exactly what forced regO's *rescale tax* (Chapter 17): to scale "row $r$"
of the O accumulator you had to round-trip it through shared memory. The raw PTX
instruction `mma.sync` has a **documented, fixed** thread→register mapping. Knowing
it lets you operate on fragment elements *in registers* — rescale by row, repack one
matmul's output into the next's input — eliminating the smem round-trips. The price
is that you manage the layouts yourself, which is error-prone — hence §18.5's
validation harness.

## 18.2 The instruction

We use the BF16 tensor-core MMA with FP32 accumulation, shape **m16n8k16**:

```
mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32
   {d0,d1,d2,d3},        // D : 16×8 FP32 accumulator (also C, in-place)
   {a0,a1,a2,a3},        // A : 16×16 BF16, row-major  (4 regs × 2 bf16)
   {b0,b1},              // B : 16×8  BF16, col-major  (2 regs × 2 bf16)
   {d0,d1,d2,d3};        // C : accumulate into D
```

One warp (32 lanes) cooperatively computes $D = A\cdot B + C$ for a $16\times8$ tile
contracting over $K=16$. `.row.col` means A is row-major, B is column-major. The
operands and result are *distributed across the warp's registers* per a fixed map.
The project's wrapper (`attention.cu`):

```c++
__device__ void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
      "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
      "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
      : "+f"(d[0]),"+f"(d[1]),"+f"(d[2]),"+f"(d[3])
      : "r"(a[0]),"r"(a[1]),"r"(a[2]),"r"(a[3]),"r"(b[0]),"r"(b[1]));
}
```

The `"+f"` ties C and D to the same registers (accumulate in place). `a`/`b` are
`uint32_t` each packing two BF16 values.

## 18.3 The fragment layouts (the crucial table)

Let `gid = lane/4` (0–7) and `t = lane%4` (0–3). The PTX ISA fixes:

**A** (16×16, row-major) — 4 regs, each 2 BF16:
```
 a0 = A[gid  ][2t], A[gid  ][2t+1]      (k = 0..7   region)
 a1 = A[gid+8][2t], A[gid+8][2t+1]
 a2 = A[gid  ][2t+8], A[gid  ][2t+9]    (k = 8..15  region)
 a3 = A[gid+8][2t+8], A[gid+8][2t+9]
```

**B** (16×8, col-major) — 2 regs:
```
 b0 = B[2t  ][gid], B[2t+1][gid]        (k = 0..7)
 b1 = B[2t+8][gid], B[2t+9][gid]        (k = 8..15)
```

**C / D** (16×8, FP32) — 4 regs:
```
 d0 = C[gid  ][2t]      d1 = C[gid  ][2t+1]
 d2 = C[gid+8][2t]      d3 = C[gid+8][2t+1]
```

Two facts in this table do all the work in the attention kernel:

- **C and A share the row map.** Both put row `gid` in their first pair of regs and
  row `gid+8` in the second. So a thread that computes C-row `gid` *also owns* A-row
  `gid` — which is why O (a C-fragment) can be rescaled in registers per row (§18.4)
  and why a QK^T result (C) repacks into a PV input (A) for free (§18.4).
- **The column index is `2t`/`2t+1`** for C — so the 4 lanes of a `gid`-group
  (t=0..3) collectively own all 8 columns of a row, letting a softmax row-reduction
  be a `__shfl_xor` over those 4 lanes (Chapter 17).

## 18.4 How the kernel exploits the layout

In `flash_mma_kernel` (Chapter 17), with $D=128$, $B_N=48$ keys/tile,
$B_M=64$ queries/CTA (16 per warp):

**O-rescale, in registers.** O is a set of C-fragments `of[nt][4]`. After computing
the per-row corrections `corr_lo` (row `gid`) and `corr_hi` (row `gid+8`), the
rescale is just:

```c++
of[nt][0]*=corr_lo; of[nt][1]*=corr_lo;   // d0,d1 are row gid
of[nt][2]*=corr_hi; of[nt][3]*=corr_hi;   // d2,d3 are row gid+8
```

No smem, no reload — *the tax is gone*, because the layout tells each thread which
rows its 4 accumulator regs hold.

**Softmax in registers.** The scores from QK^T are C-fragments. Row max/sum reduce
over the 4 lanes sharing a row via `__shfl_xor_sync(mask, v, 1)` then `..., 2)`
(which stay within the aligned group of 4). Scores never touch smem.

**QK^T → PV repack, for free.** PV's contraction dim is the keys = QK^T's N dim, and
C's row map equals A's row map with the same `2t` column map. So the QK^T result
registers (the probabilities `pcb`, cast to BF16) **are** the PV A-fragment, just
packed two-per-uint32:

```c++
const uint32_t af[4] = {
   pack2(pcb[2*kt][0], pcb[2*kt][1]),    // A-row gid,   k=2t..  ← C d0,d1
   pack2(pcb[2*kt][2], pcb[2*kt][3]),    // A-row gid+8        ← C d2,d3
   pack2(pcb[2*kt+1][0], pcb[2*kt+1][1]),
   pack2(pcb[2*kt+1][2], pcb[2*kt+1][3]) };
```

No store of P to smem and back — the P round-trip regO/WMMA paid is eliminated. The
inner loop's only smem traffic is the K/V global→smem staging.

## 18.5 `ldmatrix`: the cooperative fragment load

The one thing that *must* come from shared memory is the K/V tile (it arrives from
global). Loading it into the `mma` B-fragment layout with manual scalar reads was
*correct but slow* (Chapter 17: 42.3 ms — bank conflicts, 96 scalar reads/tile).
`ldmatrix` is the instruction `wmma` uses internally: one warp-cooperative,
conflict-free instruction loads four (or two) **8×8 BF16** tiles from shared memory
*directly into the fragment layout* `mma` expects.

```c++
__device__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    uint32_t a = __cvta_generic_to_shared(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]),"=r"(r[1]) : "r"(a));
}
// + an x2.trans variant that transposes each 8×8 on load.
```

For `x2`, lanes 0–15 each supply the **address of one row** of the two 8×8 tiles;
the hardware distributes the 128 loaded BF16 values across all 32 lanes' result
registers in exactly the `mma` B layout. The `.trans` variant transposes each 8×8 —
needed when the source's row/column roles are swapped relative to what `mma` wants.

## 18.6 The address recipes — and how they were found

The hard part of `ldmatrix` is the **address each lane supplies**, which depends on
the source's memory layout. The project pinned the two recipes it needs
*empirically*, by sweeping `(sel, trans)` in `mma_unit.cu` until the result matched
a reference (§18.7). The winners:

**QK^T — K as the B operand.** K is in smem as `Ks[key][hd]` (row=key, col=hd). For
$QK^\top$, B is $K^\top$, which in this storage is **column-major** B — so
`ldmatrix` **non-trans**, with each lane addressing its key row and hd half:

```c++
// nt = key 8-tile, kt = hd 16-tile, D = head dim
ldmatrix_x2(bf, &Ks[(nt*8 + (lane&7))*D + kt*16 + ((lane>>3)&1)*8]);
```

**PV — V as the B operand.** V is in smem as `Vs[key][hd]` (row=key, col=hd). For
$PV$, B is V with key as the contraction (row) and hd as N (col) — **row-major** B —
so `ldmatrix` **trans**:

```c++
// nt = hd 8-tile, kt = key 16-tile
ldmatrix_x2_trans(bf, &Vs[(kt*16 + (lane&15))*D + nt*8]);
```

(Q stays a manual register load — it is loaded once, outside the K loop, so it is
not on the hot path; PV's A operand comes from registers via the §18.4 repack.)
Swapping these in for the scalar loads is what took the kernel from 42.3 ms to
**19.7 ms at 279 GB/s** (Chapter 17).

## 18.7 The validation harness: `mma_unit.cu`

Hand-written PTX layouts are a minefield — a wrong index does not crash, it produces
*plausible wrong numbers*. The project's defense was a **standalone, fast-iterating
harness** that validated each layout against a CPU reference *before* it went into
the kernel. `tools/mma_unit.cu` (built with a one-line `nvcc -arch=sm_121a`) does
two things:

1. **Validate the `mma` primitive.** Load a known 16×16 A and 16×8 B with *manual*
   index-by-index fills (using the §18.3 table), run `mma_m16n8k16`, write C back via
   the C-layout, and compare to a CPU `A·B`. Result: **max-abs-error 0.00000** — the
   layout table is exactly right.

2. **Sweep the `ldmatrix` recipes.** For each B operand (QK^T's col-major K source,
   PV's row-major V source), try `sel ∈ {0,1}` × `trans ∈ {0,1}` address recipes,
   run the `mma`, and print which combination reproduces the reference. Output:

   ```
   QK  sel=0 trans=0   max_err=0.0000   <== MATCH
   PV  sel=1 trans=1   max_err=0.0000   <== MATCH
   ```

This is *why the attention kernel was bit-exact on the first compile*: the matmul
primitives and load recipes were proven in isolation, so the only thing left to get
right in the kernel was the online-softmax bookkeeping (which the `test_attention`
oracle then confirmed). Keep `mma_unit.cu` for any future tensor-core work — it is
the cheapest possible way to settle a layout question.

> **The general technique.** When hand-writing tensor-core PTX: (1) write the layout
> table from the ISA, (2) validate the bare `mma` with manual fills vs a CPU
> reference, (3) *sweep* the `ldmatrix` address/transpose options against the same
> reference rather than reasoning them out, (4) only then build the real kernel, and
> (5) validate it against an independent oracle. Steps 2–3 take minutes and save
> hours of debugging a "plausible but wrong" full kernel.

## 18.8 Where this lives in the code

| Concept | Code |
|---|---|
| `mma.sync` wrapper + layout comment | `attention.cu` `mma_m16n8k16`, `pack2` |
| `ldmatrix` wrappers | `attention.cu` `ldmatrix_x2`, `ldmatrix_x2_trans` |
| the production kernel using them | `attention.cu` `flash_mma_kernel<128>` |
| layout + recipe validation | `tools/mma_unit.cu` |
| CUTLASS reference for the PTX | `third_party/cutlass/include/cute/arch/{mma_sm80,copy_sm75}.hpp` |

## 18.9 Summary and what to carry forward

- `mma.sync.m16n8k16.f32.bf16` computes a $16\times8$ tile per warp with a **fixed,
  documented** thread→register layout (the §18.3 table). Dropping below `wmma`
  exposes it.
- That layout has two gifts: **C and A share the row map** (so O rescales in
  registers and QK^T results repack into PV inputs for free), and **columns are
  `2t`** (so softmax reduces via `__shfl_xor` over 4 lanes). Together they remove the
  rescale tax, the smem softmax, and the P round-trip.
- **`ldmatrix`** is the conflict-free cooperative smem→fragment load; its address
  recipe depends on the source layout (non-trans for QK^T's col-major K, trans for
  PV's row-major V). Manual scalar loads were correct but 2× slower.
- **`mma_unit.cu`** validated the `mma` layout (max-err 0.0) and *swept* the
  `ldmatrix` recipes to a match before any kernel code — which is why the kernel was
  correct on the first compile. Reuse this harness for any tensor-core PTX work.

That closes Part IV — attention from the math (16) through the optimization journey
(17) to the metal (18). Part V turns to the VAE decoder and the very different
performance story of **convolutions** on Blackwell.

---

### Exercises

1. **Read the layout.** For lane 5 (gid=1, t=1), name exactly which A, B, and C
   elements its registers hold, using §18.3.
2. **Free repack.** Show that the QK^T C-fragment register `d0` for a thread equals
   the PV A-fragment element that thread needs, and explain why this requires C and
   A to share the row map *and* the column convention.
3. **trans or not.** Given K stored `[key][hd]` and V stored `[key][hd]`, explain
   why QK^T's K-load is non-trans but PV's V-load is trans. (Hint: which axis is the
   contraction for each matmul?)
4. **The harness pays off.** A full attention kernel returns plausible-but-wrong
   numbers. Argue why isolating the `mma`+`ldmatrix` in `mma_unit.cu` localizes the
   bug faster than debugging the kernel directly. What class of bug does each step
   catch?

*Next: [Chapter 19 — Convolutions with the cuDNN 9 graph API](19-cudnn-conv.md), opening Part V.*
