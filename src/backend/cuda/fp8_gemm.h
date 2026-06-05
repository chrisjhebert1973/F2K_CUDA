// Blackwell MXFP8 → BF16 GEMM, packaged as a reusable building block.
//
// Mirrors fp4_gemm.h but with E4M3 data (1 byte/element) and per-32 UE8M0
// block scale factors (MXFP8) instead of NVFP4's E2M1 + per-16 E4M3 scales.
// MXFP8 keeps 3 mantissa bits vs FP4's 1, recovering most of the BF16 quality
// (see project-fp8-vs-fp4 memory). Type config mirrors CUTLASS example 79c
// (blackwell_geforce mixed mxfp8), with BOTH operands MXFP8.
//
// All CUTLASS template machinery lives in fp8_gemm.cu; this header stays
// CUTLASS-free.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class FP8Gemm {
public:
    // Configures the kernel for an M × N × K MXFP8 → BF16 GEMM.
    // Requires M, N, K multiples of 128 (K must be a multiple of the 32-elem
    // scale-factor vector; 128 is the threadblock K tile).
    FP8Gemm(int m, int n, int k);
    ~FP8Gemm();

    FP8Gemm(const FP8Gemm&)            = delete;
    FP8Gemm& operator=(const FP8Gemm&) = delete;
    FP8Gemm(FP8Gemm&&) noexcept;
    FP8Gemm& operator=(FP8Gemm&&) noexcept;

    bool ok() const;
    const char* last_error() const;

    int m() const;
    int n() const;
    int k() const;

    // ---- caller-side allocation queries (in bytes) ----
    // FP8 operands are 1 byte/element (E4M3); row strides match the matrix
    // dimensions (no padding).
    size_t a_size_bytes() const;       // M × K
    size_t b_size_bytes() const;       // N × K
    size_t c_size_bytes() const;       // M × N × sizeof(bf16)
    size_t d_size_bytes() const;       // M × N × sizeof(bf16)
    size_t sfa_size_bytes() const;     // UE8M0 scale factors, per CUTLASS layoutSFA
    size_t sfb_size_bytes() const;
    size_t workspace_size_bytes() const;

    // Execute one GEMM. All pointers are device pointers.
    //   A:   E4M3 row-major     [M, K]
    //   SFA: UE8M0 block scales for A (per 32 along K)
    //   B:   E4M3 column-major  [N, K]
    //   SFB: UE8M0 block scales for B
    //   C:   BF16 input  [M, N] — pass nullptr if beta == 0
    //   D:   BF16 output [M, N]
    //   workspace: device buffer of workspace_size_bytes()
    bool run(const void* A,    const void* SFA,
             const void* B,    const void* SFB,
             const void* C,    void*       D,
             void*       workspace,
             float alpha = 1.0f, float beta = 0.0f,
             cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// Smoke / benchmark helper — mirrors run_fp4_gemm_smoke.
// ---------------------------------------------------------------------------

struct FP8GemmResult {
    bool   ok             = false;
    bool   verify_passed  = false;
    double max_abs_err    = 0.0;
    double mean_abs_err   = 0.0;
    double avg_runtime_ms = 0.0;
    double tflops         = 0.0;
    const char* error_msg = nullptr;
};

FP8GemmResult run_fp8_gemm_smoke(int m, int n, int k, int iterations = 10);

} // namespace f2k::cuda
