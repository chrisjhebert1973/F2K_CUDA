# F2KTest

Custom CUDA/cuDNN inference engine and Dear ImGui frontend for the
[FLUX.2-klein](https://hf.co/black-forest-labs/FLUX.2-klein-9B) image
generator/editor, targeted at NVIDIA DGX Spark (GB10, Blackwell sm_121).

## Status

Working end-to-end prompt → image on real FLUX.2-klein-9B weights, entirely on
our own CUDA/CUTLASS/cuDNN path:

- **Text encoder** — native Qwen3-8B, verified vs HF (cos ≥ 0.974).
- **Transformer** — full MMDiT (8 double + 24 single stream blocks), 4-axis
  RoPE, joint attention; matches diffusers (cos 0.999 in FP8).
- **VAE decoder** — cuDNN conv/groupnorm/attention, incl. `post_quant_conv`.
- **Sampler** — flow-matching Euler with the Flux2 dynamic-shift schedule.
- **Quantization** — Blackwell block-scaled GEMMs: **NVFP4** (E2M1, fastest) and
  **MXFP8** (E4M3, near-BF16 quality). Pick with `generate --precision fp8`.

32 ctest cases green. See `tools/generate.cu` for the end-to-end driver.

## Building (Spark / aarch64-linux, CUDA 13)

```
git submodule update --init --recursive
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
ctest --test-dir build
```

## Weight conversion

```
# safetensors → F2K (NVFP4 or MXFP8); see tools/f2k_convert.cu
./build/f2k_convert model.safetensors out.f2k1 --quant nvfp4   # or --quant mxfp8
```

## Layout

```
src/backend/cuda/   CUDA/CUTLASS kernels + modules (GEMMs, blocks, VAE, sampler)
src/common/         F2K weight format, model loader, tensor router
tests/              ctest suite (one per module)
tools/              generate, f2k_convert, diffusers reference/compare scripts
third_party/        Vendored deps as git submodules (CUTLASS, ImGui, GLFW, ...)
scripts/            Convenience scripts (model download, headless run)
```
