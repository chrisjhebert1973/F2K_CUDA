# Patches

Local fixes applied on top of pinned third-party submodules. Re-apply after any
`git submodule update`.

## cutlass-sm120-tma-align.patch

Fixes a **misaligned-shared-address crash in the SM120 (consumer Blackwell /
RTX 50-series) NVFP4 + MXFP8 block-scaled GEMM** — CUTLASS upstream
[issue #2906](https://github.com/NVIDIA/cutlass/issues/2906) / [#2905](https://github.com/NVIDIA/cutlass/issues/2905).

The TMA tensormap requires 64-byte alignment but the device-side `Params`
structs lacked `alignas(64)`, so on sm_120a the kernel faults inside
`copy_sm90_desc.hpp` during the epilogue's TMA store. (sm_121a / GB10 tolerates
the misalignment; sm_120a / GB202 does not.) Adds `alignas(64)` to the `Params`
structs and their TMA members in:
- `gemm/collective/sm120_blockscaled_mma_tma.hpp`
- `epilogue/collective/sm90_epilogue_tma_warpspecialized.hpp`

Verified: no change on sm_121a (test_fp4_gemm still PASS, ~291 TFLOPS).

### Apply
```sh
cd third_party/cutlass
git apply ../../patches/cutlass-sm120-tma-align.patch
# verify it took:  git diff --stat
```
To revert: `git -C third_party/cutlass checkout -- include/`.
