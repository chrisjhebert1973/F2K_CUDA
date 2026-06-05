// Blackwell NVFP4 → BF16 GEMM, packaged as a reusable building block.
//
// The class FP4Gemm is the persistent handle: configure once with the
// problem shape, then call run() repeatedly across iterations. All the
// CUTLASS template machinery lives in fp4_gemm.cu; this header stays
// CUTLASS-free so callers (UI, backend, MMDiT block) don't drag CUTLASS
// includes into every TU.
//
// The standalone smoke test entry — run_fp4_gemm_smoke — remains for
// validation/benchmarking purposes and is implemented on top of FP4Gemm.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class FP4Gemm {
public:
    // Configures the kernel for an M × N × K NVFP4 → BF16 GEMM.
    // Requires M, N multiples of 128 and K a multiple of 64.
    FP4Gemm(int m, int n, int k);
    ~FP4Gemm();

    FP4Gemm(const FP4Gemm&)            = delete;
    FP4Gemm& operator=(const FP4Gemm&) = delete;
    FP4Gemm(FP4Gemm&&) noexcept;
    FP4Gemm& operator=(FP4Gemm&&) noexcept;

    bool ok() const;
    const char* last_error() const;

    int m() const;
    int n() const;
    int k() const;

    // ---- caller-side allocation queries (in bytes) ----
    // FP4 operands are packed two-per-byte; row strides match the matrix
    // dimensions (no extra padding).
    size_t a_size_bytes() const;       // M × K / 2
    size_t b_size_bytes() const;       // N × K / 2
    size_t c_size_bytes() const;       // M × N × sizeof(bf16)
    size_t d_size_bytes() const;       // M × N × sizeof(bf16)
    size_t sfa_size_bytes() const;     // microblock scale factors, laid out per CUTLASS layoutSFA
    size_t sfb_size_bytes() const;
    size_t workspace_size_bytes() const;

    // Execute one GEMM. All pointers are device pointers.
    //   A:   FP4 packed, row-major     [M, K]
    //   SFA: E4M3 microblock scales for A
    //   B:   FP4 packed, column-major  [N, K] (i.e. K × N in row-major)
    //   SFB: E4M3 microblock scales for B
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
// Smoke / benchmark helper. Allocates everything internally, runs the kernel,
// validates against the CUTLASS host reference. Useful for verifying the
// toolchain and benchmarking; not used on the hot path.
// ---------------------------------------------------------------------------

struct FP4GemmResult {
    bool   ok             = false;
    bool   verify_passed  = false;
    double max_abs_err    = 0.0;
    double mean_abs_err   = 0.0;
    double avg_runtime_ms = 0.0;
    double tflops         = 0.0;
    const char* error_msg = nullptr;
};

FP4GemmResult run_fp4_gemm_smoke(int m, int n, int k, int iterations = 10);

} // namespace f2k::cuda
