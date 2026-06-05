// Linear: y = activation(BF16) @ W^T(NVFP4) + bias(BF16, optional)
//
// On construction, the BF16 weight is converted once into:
//   - packed NVFP4 ([out, in], two values per byte)
//   - per-microblock E4M3 scales laid out in CUTLASS's SFB tiling
//   - per-tensor FP32 scale (set to 1.0 in this v1 — dynamic per-block
//     scaling subsumes it for our use)
//
// On every forward(), the BF16 activation is dynamically quantized into a
// matching FP4+SFA pair, the FP4Gemm runs, and bias is added if present.
//
// Sizing constraints (CUTLASS Blackwell GeForce NVFP4 path):
//   batch_rows % 128 == 0
//   out_features % 128 == 0
//   in_features  % 64  == 0
// We may relax these later for unaligned batch sizes via padding workspace.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// Quantization mode for the GEMM operands.
//   NVFP4: E2M1 data + per-16 E4M3 block scales (fastest, ~13% image err).
//   MXFP8: E4M3 data + per-32 UE8M0 block scales (~2× slower, near-BF16).
// See project-fp8-vs-fp4 memory.
enum class Precision { NVFP4, MXFP8 };

// Weight descriptor passed to blocks. Despite the name it carries either path:
//   - NVFP4: pre-quantized bits (packed/scales) from an NVFP4 F2K file.
//   - MXFP8: a BF16 source weight (`bf16`) that Linear quantizes at construction.
//   packed: [N, K/2] row-major bytes (low nibble = even k, high = odd)
//   scales: [N, K/16] row-major bytes (E4M3 raw)
// All pointers are HOST pointers (typically into an mmapped F2K file).
struct PreQuantNVFP4 {
    const void* packed       = nullptr;   // [N, K/2]  (NVFP4)
    const void* scales       = nullptr;   // [N, K/16] (NVFP4)
    int         N            = 0;
    int         K            = 0;
    int         microblock_size = 16;
    float       tensor_scale   = 1.0f;
    // Appended after the original fields so positional initializers
    // `{packed, scales, N, K, mb, scale}` keep working.
    const void* bf16         = nullptr;   // [N, K] BF16 source (MXFP8 ctor-quant)
    Precision   precision    = Precision::NVFP4;
};

class Linear {
public:
    // Alias so existing call sites can keep using Linear::Precision.
    using Precision = f2k::cuda::Precision;

    struct Config {
        int batch_rows   = 0;   // M  (product of leading activation dims)
        int in_features  = 0;   // K
        int out_features = 0;   // N

        Precision precision = Precision::NVFP4;

        // Exactly ONE of these two paths must be populated:
        //   1. W_bf16:  BF16 host weight [out, in]; Linear quantizes internally.
        //   2. W_preq:  Pre-quantized NVFP4 from disk; Linear repacks directly
        //               into CUTLASS SFB layout (no requant noise). NVFP4 only.
        const void*           W_bf16  = nullptr;
        const PreQuantNVFP4*  W_preq  = nullptr;

        const void* bias_bf16 = nullptr;   // [out_features], optional
    };

    explicit Linear(const Config& cfg);
    ~Linear();
    Linear(const Linear&)            = delete;
    Linear& operator=(const Linear&) = delete;
    Linear(Linear&&) noexcept;
    Linear& operator=(Linear&&) noexcept;

    bool        ok()          const;
    const char* last_error()  const;

    // Workspace required for one forward() call: holds the activation's FP4
    // packed buffer, the SFA scale table, and the CUTLASS GEMM workspace.
    size_t workspace_size_bytes() const;

    // y = x @ W^T (+ bias). All pointers are device pointers.
    //   x_bf16: [batch_rows, in_features]
    //   y_bf16: [batch_rows, out_features]
    //   workspace: device buffer of workspace_size_bytes()
    bool forward(const void* x_bf16, void* y_bf16,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Populate a Linear::Config's weight + precision from a weight descriptor.
// MXFP8 routes the BF16 source through W_bf16; NVFP4 uses the pre-quant bits.
inline void set_linear_weight(Linear::Config& c, const PreQuantNVFP4* w) {
    c.precision = w->precision;
    if (w->precision == Precision::MXFP8 && w->bf16) c.W_bf16 = w->bf16;  // quantize at ctor
    else                                             c.W_preq = w;        // pre-quantized bits
}

} // namespace f2k::cuda
