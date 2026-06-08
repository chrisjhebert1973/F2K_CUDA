# Chapter 14 — Weights on Disk: the F2K Format, Loader & Router

> *Goal of this chapter:* follow the weights from a Black Forest Labs
> `.safetensors` download to device-ready tensors. We design **F2K**, a custom
> mmap-and-go container built for block-scaled quantization and unified memory;
> walk the multi-shard **loader**; and explain the **tensor router** that turns 233
> opaque tensor names into the structured per-block roles the model consumes.
> Anchored to `src/common/f2k_format.*`, `f2k_model_loader.*`, `tensor_router.*`,
> and `tools/f2k_convert.cu`.
>
> *Prerequisites:* Chapters 12–13 (the formats and why scale layout matters);
> Chapter 11 (unified memory).

---

## 14.1 Why not safetensors or GGUF?

The weights ship as `.safetensors` (HF's format). Two existing formats were
candidates for the runtime container, and both were rejected:

- **safetensors** has no first-class **quantization metadata**. There is no concept
  of per-microblock scales; stuffing the NVFP4 packed values + E4M3 block scales +
  per-tensor FP32 scale into its JSON header would lose the alignment guarantees the
  tensor cores need (Chapter 13's tiled scale layout).
- **GGUF** (llama.cpp) was the original plan, but it is tied to llama.cpp's k-quant
  types (Q4_K, Q5_K). On Blackwell we target **NVFP4/MXFP8 + tensor cores**, not
  k-quants, so GGUF's quant zoo is the wrong vocabulary, and its design assumes
  VRAM-constrained consumer GPUs — irrelevant with 128 GB unified (Chapter 11).

So the project defines **F2K**, optimized for exactly this stack: store tensors in
**GEMM-ready storage order**, 64-byte aligned, with native block-scale metadata, so
the runtime can **mmap and go** — no parsing, no copy, no dequant on the load path.

## 14.2 The F2K v1 container

```
 ┌───────────────────────────── file ─────────────────────────────┐
 │ 64-byte FileHeader                                              │
 │   magic 'F2K1' (0x314B3246), version, manifest_off, manifest_sz,│
 │   data_off, data_sz, total_tensors, 16 reserved bytes          │
 ├────────────────────────────────────────────────────────────────┤
 │ DATA REGION  (written first; tensor blobs, each 64-byte aligned)│
 │   tensor0 values | tensor0 scales | tensor1 values | ...        │
 ├────────────────────────────────────────────────────────────────┤
 │ MANIFEST  (written after data; one entry per tensor)            │
 │   TensorEntryFixed (48 B) + int64 shape[rank] + UTF-8 name      │
 └────────────────────────────────────────────────────────────────┘
```

Design decisions, each with a reason:

- **Data first, manifest second, header patched last.** The converter can *stream*
  tensors to disk without knowing the total count up front; it appends the manifest
  and back-patches the header offsets on commit. Good for a one-pass converter.
- **64-byte alignment** (`F2K_DATA_ALIGN`) on every blob — matches cache lines and
  the tensor-core load alignment, so a tensor can be fed to a kernel directly from
  its mmap'd address.
- **Quantized tensors carry a separate scale blob.** A quantized tensor stores its
  packed values *and* a separate, 64-byte-aligned per-microblock scale blob with its
  own offset/size; the per-tensor FP32 scale lives in the fixed entry; the
  microblock size (16 for NVFP4, 32 for MXFP8) is recorded. This is the on-disk
  shape of Chapter 12's formats.
- **dtype codes are frozen.** `0 F16, 1 BF16, 2 F32, 3 F8_E4M3, 4 F8_E5M2, 5 NVFP4,
  6 U8, 7 I8, 8 I32, 255 Unknown`. *Never renumber a shipped code*; append new
  dtypes at the next free integer, and bump `F2K_VERSION` only when changing the
  semantics of an existing field (gate new reader paths on the version). The router
  in `Linear` (Chapter 13) switches on these codes to pick the weight path
  (F8_E4M3 → MXFP8 pre-quant copy; BF16 → quantize-at-ctor; NVFP4 → FP4 pre-quant).

A round-trip test (`tests/test_round_trip.cpp`) writes and re-reads tensors to prove
the container is self-consistent.

## 14.3 Conversion: `f2k_convert`

`tools/f2k_convert.cu` is the offline pipeline (Chapter 12 covered the quant math;
here the I/O):

1. Read each tensor from the source `.safetensors` (via `src/common/safetensors.*`).
2. Classify: rank-2 `.weight` with $N\%128=0, K\%64=0$ → quantize (NVFP4 or MXFP8
   per `--quant`); everything else → BF16 passthrough. A `--keep-bf16 <substr>` flag
   forces specific tensors (e.g. Qwen3's `embed_tokens`) to stay BF16.
3. Write packed values + scale blob (NVFP4/MXFP8) or raw BF16 into the data region;
   record the manifest entry.

Real numbers: transformer shard 1 converts 9.13 GiB → **2.57 GiB (3.55× )** in ~77 s
on the ARM cores (NVFP4), 113 tensors quantized + 42 passthrough, worst per-tensor
`max_err 0.106`. The MXFP8 transformer is ~8.7 GiB (`--quant mxfp8`), and a BF16
passthrough F2K (~18 GiB) also exists for construction-time quantization. The model
is split across **2 transformer shards**, **4 Qwen3 shards**, and **1 VAE file**.

> **Why an MXFP8 on-disk form matters.** Quantizing BF16→MXFP8 *at model build*
> took ~76 s every launch. Storing the result on disk (`--quant mxfp8`) lets the
> runtime *copy* pre-quantized FP8 instead of computing it, cutting transformer
> build to **5.3 s** (14×) with bit-identical output. On-disk quant is the norm;
> the BF16 file is only needed if you want ctor-time quant.

## 14.4 The loader: mmap and go

`f2k_model_loader.*` is a thin multi-shard layer over the raw reader:

- **mmap, not read.** Each shard is memory-mapped. Because memory is **unified and
  coherent** (Chapter 11), the GPU can read weights at their mmap'd host addresses;
  there is no `cudaMemcpy` of the model, no device copy, no staging. The OS pages
  data in on demand.
- **Unified name→view.** It presents one `name → TensorView` lookup across all
  shards, with duplicate detection and per-tensor shard location. A `TensorView` is
  just `(ptr, dtype, shape, scale ptr, …)` into the mmap.
- **Fast.** The real model (233 tensors, 4.76 GiB on disk) "loads" in
  *milliseconds* — mmap is lazy; the cost is just mapping, not reading. The actual
  byte traffic happens when `Linear` construction touches each weight to repack it
  (Chapter 13).

This is the unified-memory payoff made concrete: "loading a 9B model" is an `mmap`
and a name table, not a multi-second device transfer.

## 14.5 The router: names → structure

The loader gives `name → bytes`. But the model needs *structure*: "give me block
3's image Q-projection weight," not a string lookup. `tensor_router.*` parses the
HF naming convention into roles:

```
 transformer_blocks.3.attn.to_q.weight        → DoubleBlock[3].img_q
 transformer_blocks.3.attn.add_q_proj.weight  → DoubleBlock[3].txt_q
 single_transformer_blocks.7.attn.to_qkv_mlp_proj.weight → SingleBlock[7].qkv_mlp
 context_embedder.weight                       → Globals.context_embedder
 double_stream_modulation_img.linear.weight    → Globals.mod_double_img   (SHARED)
 ...
```

It produces a `GlobalTensors` struct + a `vector<DoubleBlockTensors>` (8) +
`vector<SingleBlockTensors>` (24). On the real file **every required slot is filled
and zero tensors are unrouted** — a strong structural check that the naming
convention is fully understood. The decomposition confirms Chapter 04's accounting:

$$
\underbrace{8\times 16}_{\text{double}} + \underbrace{24\times 4}_{\text{single}} + \underbrace{9}_{\text{globals}} = 233\ \text{tensors}.
$$

The router is also where the **shared-modulation** discovery lives (Chapter 04/07):
there are no per-block modulation linears in the file — `double_stream_modulation_{img,txt}`
and `single_stream_modulation` are single tensors shared across all blocks of their
type. The router maps them into the globals, and every block reads the same
modulation pointers. `tests/test_tensor_router.cu` asserts the full routing on the
real model.

## 14.6 The path from disk to a running model

Putting Chapters 12–14 together, the lifecycle of one weight:

```
 safetensors (BF16, on HF)
    │ f2k_convert (offline): classify → quantize (NVFP4/MXFP8) → pack + scales
    ▼
 F2K shard on disk (packed values + block scales, 64-B aligned, dtype-coded)
    │ F2KModelLoader: mmap (ms), name→TensorView (unified memory, no copy)
    ▼
 TensorRouter: name → GlobalTensors / DoubleBlock[i] / SingleBlock[i]
    │ FluxTransformer ctor: Linear repacks each view into CUTLASS SFB layout (ch 13)
    ▼
 device-resident, tensor-core-ready Linear  (5.3 s for all 124, on-disk MXFP8)
```

No step parses JSON on the hot path, none copies the model across a bus, and the
quant work is either offline (`f2k_convert`) or a one-time copy at construction.
That is the F2K design goal — *mmap and go* — realized.

## 14.7 Where this lives in the code

| Concept | Code |
|---|---|
| container format | `src/common/f2k_format.{h,cpp}` (`FileHeader`, `TensorEntryFixed`, `F2KReader`) |
| conversion | `tools/f2k_convert.cu` |
| multi-shard mmap loader | `src/common/f2k_model_loader.{h,cpp}` |
| name → role routing | `src/common/tensor_router.{h,cpp}` |
| safetensors reader (convert-time) | `src/common/safetensors.{h,cpp}` |
| tests | `tests/test_round_trip.cpp`, `test_f2k_model_loader.cpp`, `test_tensor_router.cpp` |

## 14.8 Summary and what to carry forward

- **F2K** is a custom container because safetensors lacks block-scale metadata and
  GGUF assumes k-quants/VRAM limits. It stores **GEMM-ready, 64-B-aligned,
  block-scale-aware** tensors for **mmap-and-go**.
- Layout: 64-B header → data region (values + separate scale blobs) → manifest.
  **dtype codes are frozen**; bump `F2K_VERSION` for semantic changes.
- `f2k_convert` quantizes offline (3.55× shrink, NVFP4); an **on-disk MXFP8** form
  cuts model build 76 s → 5.3 s.
- The **loader mmaps** (ms; unified memory → no device copy); the **router** turns
  233 names into `Globals + 8 DoubleBlock + 24 SingleBlock`, with **shared
  modulation** and zero unrouted tensors.

Chapter 15 finishes Part III with the small kernel library — the norms, activations,
and data-movement ops that glue the GEMMs and attention together.

---

### Exercises

1. **Why data-first.** Explain how writing the data region before the manifest lets
   the converter stream tensors without knowing the count, and what the header
   back-patch does.
2. **Frozen codes.** Why must dtype code 5 (NVFP4) never be reused for a different
   format, even in v2? What is the right way to add a new dtype?
3. **mmap economics.** Explain why "load the 9B model" is milliseconds here but the
   *first forward* touches real bandwidth. Where does the on-disk→device byte
   traffic actually happen?
4. **Routing as a check.** Why is "zero unrouted tensors + every slot filled" strong
   evidence the architecture is fully understood? What would a missing slot mean?

*Next: [Chapter 15 — The core kernel library](15-kernels.md).*
