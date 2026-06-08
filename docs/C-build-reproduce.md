# Appendix C — Build & Reproduce

> Everything needed to build the project, run the tests, and reproduce the numbers
> and images cited in the course. Commands assume the repo root and the GB10
> environment described in Chapter 11.

---

## C.1 Toolchain

| component | version | notes |
|---|---|---|
| GPU | NVIDIA GB10 (Blackwell), **sm_121** | the only supported target here |
| CUDA | **13.0** (sbsa-linux), nvcc V13.0 | aarch64 SBSA |
| cuDNN | **9** (`libcudnn.so.9`) | **must** link 9, not the `.so`→8.x symlink (App B.10) |
| CUTLASS | vendored submodule | `third_party/cutlass` |
| compiler | g++ 13 | aarch64 |
| CMake | ≥ 3.28 | |
| Python (tools only) | `.venv` with torch 2.12+cu130, transformers 5.10, diffusers 0.38 | reference dumpers + (legacy) tokenization |

The runtime is fully native C++/CUDA; Python is only for the offline reference tools
of Chapter 25 (`.venv/bin/python tools/...`).

## C.2 Submodules

The vendored dependencies are git submodules (`third_party/`): `cutlass`, `glfw`,
`imgui`, `stb`, `json`. After cloning:

```bash
git submodule update --init --recursive
```

(The submodule *contents* are not in the repo — only references — so this step is
required before building. See the push note in the project history.)

## C.3 The CMake gotchas (read before building)

Two settings are load-bearing (both are war stories — App B):

1. **Pin the CUDA arch with FORCE before `enable_language(CUDA)`** (App B.11):
   ```cmake
   set(CMAKE_CUDA_ARCHITECTURES "121a" CACHE STRING "" FORCE)
   enable_language(CUDA)
   ```
   Without this, the auto-probe defaults to Turing (sm_75) and the block-scaled MMA
   fails at runtime. After changing it, **wipe `build/`** (the cache value sticks).
2. **Link cuDNN 9 explicitly** (App B.10): link
   `/usr/lib/aarch64-linux-gnu/libcudnn.so.9`, not the unversioned `libcudnn.so` dev
   symlink (which may point at a pre-Blackwell 8.x).

## C.4 Build

```bash
cmake -S . -B build
cmake --build build -j
```

Targets of interest: `generate` (the CLI), `bench_attention`, `f2k_convert`, and the
`test_*` suite. `tools/mma_unit.cu` builds standalone for fast PTX iteration:

```bash
nvcc -arch=sm_121a -o /tmp/mma_unit tools/mma_unit.cu && /tmp/mma_unit
```

## C.5 Models on disk

The runtime expects converted F2K weights under `~/models/flux2-klein-9B/`:

```
 transformer_f2k/   shard-0000{1,2}.f2k1     # NVFP4 transformer (--precision nvfp4)
 transformer_mxfp8/ shard-0000{1,2}.f2k1     # MXFP8 transformer (--precision fp8)
 qwen3_f2k/         shard-{00001..00004}.f2k1# NVFP4 Qwen3 encoder
 vae_f2k/           vae.f2k1                  # VAE decoder weights
 tokenizer/         vocab.json, merges.txt, added_tokens.json, chat_template.jinja, ...
```

Produce them from a HF download with `f2k_convert` (Chapter 12/14):

```bash
./build/f2k_convert <safetensors...> --quant nvfp4 -o .../transformer_f2k/...
./build/f2k_convert <safetensors...> --quant mxfp8 -o .../transformer_mxfp8/...
# Qwen3: --keep-bf16 embed_tokens ; VAE: passthrough/quant as configured
```

## C.6 Generate an image

```bash
# native, single call (no Python):
./build/generate --prompt "a cat on a skateboard" \
                 --res 1024 --precision fp8 --steps 4 --seed 0xCAFEBABE \
                 --out cat.png

# precomputed token ids (skip the native tokenizer):
.venv/bin/python tools/encode_prompt.py --seq 512 "a cat on a skateboard" /tmp/tok.bin
./build/generate --tokens /tmp/tok.bin --res 1024 --precision fp8 --out cat.png

# precomputed diffusers conditioning (skip encoder too — for transformer isolation):
.venv/bin/python tools/diffusers_prompt_embeds.py --prompt "a cat..." --out /tmp/emb.bin
./build/generate --embeds /tmp/emb.bin --res 1024 --precision fp8 --out cat.png
```

PPM output is the default; an `.png` extension writes PNG (Chapter 22). PPM→PNG if
needed: `.venv/bin/python -c "from PIL import Image; Image.open('out.ppm').save('out.png')"`.

## C.7 Run the tests

```bash
cd build && ctest                    # the whole suite
ctest -R 'attention|transformer|tokenizer|denoise'   # a subset
```

Each component has a golden test vs an FP32 (or HF/diffusers) reference (Chapter 25).
Expected: all green; `test_tokenizer` prints `ALL OK`; `test_attention` shows
`cos=1.00000` across shapes including the non-aligned `S=130`.

## C.8 Reproduce the course's numbers

| number (chapter) | command |
|---|---|
| GEMM peaks 252/161 TFLOP/s (11, 13) | `./build/test_fp4_gemm` / `./build/test_fp8_gemm` |
| attention 56.5→34.3→19.7 ms, 279 GB/s, CTAs/SM (11, 17) | `./build/bench_attention` |
| `mma`/`ldmatrix` layout validation (18) | `/tmp/mma_unit` (see C.4) |
| transformer cos 0.963/0.999 vs diffusers (10, 12, 25) | `./build/cmp_transformer <embeds> <sigma> {nvfp4,fp8}` |
| Qwen3 encoder cos ≥0.974 vs HF (21, 25) | `tools/qwen3_golden.py` + `./build/test_qwen_golden` |
| FP4-vs-FP8 quality 12.95%/2.12% (12) | `.venv/bin/python tools/quant_sim.py` |
| VAE decode timing, decode diffusers latent (19, 20, 25) | `./build/test_vae_decoder`; `generate --decode_latent` |
| full timing ledger (0, 23) | `./build/generate --prompt ... --res 1024 --precision fp8` (reads the stderr timings) |

`bench_attention` prints the device properties (SMs, opt-in smem, peak) at startup,
so it doubles as the environment check.

## C.9 The optimization knobs, recapped

| flag | effect | chapter |
|---|---|---|
| `--precision {nvfp4,fp8}` | speed (NVFP4) ↔ quality (MXFP8) | 12 |
| `--res N` | output resolution (N%16==0, seq_img%128==0) | 3, 13 |
| `--steps N` | sampler depth (4 default; more = sharper detail) | 2 |
| `--seed N` | reproducible noise; reroll composition | 23 |
| `--out *.png` | PNG via stb; else PPM | 22 |
| `--prompt`/`--tokens`/`--embeds`/`--decode_latent` | conditioning path / isolation | 23, 25 |

## C.10 Where to go next

- To **understand** the pipeline: read Parts I–II (math + architecture).
- To **modify a kernel**: Parts III–IV, with `bench_attention`/`mma_unit` as your
  instruments and `test_*` as your safety net.
- To **add a feature** (new precision, resolution, sampler): the `Linear::Precision`
  enum (Ch 13), the `--res` derivation (Ch 23), and `FlowMatchScheduler` (Ch 2) are
  the seams.
- To **debug a wrong image**: Chapter 25's isolation method + Appendix B's field guide.

That is the whole system — from rectified-flow theory to a `mma.sync`+`ldmatrix`
kernel at peak HBM bandwidth, building and reproducible end to end.
