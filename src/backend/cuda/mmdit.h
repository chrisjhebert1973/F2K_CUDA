// MMDiT (Multi-Modal Diffusion Transformer) block skeleton.
//
// This file declares — but does not yet implement — the shape of an MMDiT
// block as it will exist in F2KTest. The intent is to make the composition
// concrete in code so that incremental work has a destination.
//
// What exists today:
//   - FP4Gemm  (real, validated, see fp4_gemm.h)
//   - rmsnorm_bf16  (real, validated, see kernels/rmsnorm.h)
//
// What is still ahead, in rough dependency order:
//
//   1. ActivationQuantize  — BF16 → packed NVFP4 + per-microblock E4M3
//                            scales, computed on the fly per call. This is
//                            the standard NVFP4 inference pattern: weights
//                            are statically quantized at conversion time,
//                            activations get dynamic per-token scales.
//
//   2. Linear              — wraps {ActivationQuantize, FP4Gemm}. Takes
//                            BF16 in, BF16 out, with stored NVFP4 weight
//                            and E4M3 scale tables.
//
//   3. RoPE                — applies rotary positional embedding to Q, K.
//                            Precomputed cos/sin tables. Per-element fused
//                            into the attention QKV path.
//
//   4. JointAttention      — the MMDiT-specific step: concatenate text +
//                            image tokens into one sequence, attend, split
//                            back. Flash-style fused kernel.
//
//   5. SwiGLU              — gated-linear MLP. Two Linears in parallel,
//                            element-wise SiLU(gate) * up, one Linear out.
//                            For ambitious perf, fuse the gate/up into one
//                            wide GEMM call.
//
//   6. Modulation          — scale / shift / gate parameters derived from
//                            timestep + global cond embedding, applied
//                            inside the residual stream.
//
//   7. MMDiTBlock          — assembles all of the above. The full MMDiT
//                            transformer is just N of these stacked.

#pragma once

#include "backend/cuda/fp4_gemm.h"
#include "backend/cuda/kernels/rmsnorm.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace f2k::cuda {

// ---------------------------------------------------------------------------
// Tensor descriptors for the runtime layer (NOT the on-disk format — see
// f2k::TensorView in common/tensor.h for that). DeviceTensor is just a
// pointer + shape, no ownership.
// ---------------------------------------------------------------------------

struct DeviceTensor {
    void*               data    = nullptr;
    std::vector<int>    shape;          // logical shape, e.g. {batch, seq, dim}
    int                 dtype   = 0;    // f2k::DType code
};

// ---------------------------------------------------------------------------
// Linear (placeholder — implementation pending)
// ---------------------------------------------------------------------------
//
// Behavior:
//   y[b, n] = sum_k(x[b, k] * W[n, k]) + (optional bias)
// where W is stored as NVFP4 with E4M3 microblock scales and a per-tensor
// FP32 scale. x is BF16; y is BF16.
//
// Construction takes ownership of the weight pointers (which live in unified
// memory so CPU and GPU see the same allocation).

class Linear {
public:
    struct Config {
        int in_features  = 0;
        int out_features = 0;
        // Static weights (device pointers, owned externally — e.g. mmapped F2K).
        const void* W_fp4         = nullptr;
        const void* W_scales_e4m3 = nullptr;
        float       W_tensor_scale = 1.0f;
        const void* bias_bf16     = nullptr;  // optional, may be null
    };

    explicit Linear(const Config& cfg);
    ~Linear();

    // y = this(x). batch_rows = product of leading dims.
    // y must be allocated by caller, size batch_rows * out_features BF16.
    bool forward(const void* x_bf16, void* y_bf16,
                 int batch_rows, void* workspace, size_t workspace_bytes,
                 cudaStream_t stream);

    size_t workspace_size_bytes(int batch_rows) const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// ---------------------------------------------------------------------------
// MMDiT block (skeleton)
// ---------------------------------------------------------------------------
//
// Holds the per-block weights and runs the block. Designed for sequential
// inference (no batching across requests). Implementation will land
// incrementally as Linear / Attention / etc. come online.

class MMDiTBlock {
public:
    struct Config {
        int hidden_dim   = 3072;   // FLUX.2 klein, approximate
        int num_heads    = 24;
        int head_dim     = 128;
        int mlp_hidden   = 12288;
        float rms_eps    = 1e-6f;
    };

    explicit MMDiTBlock(const Config& cfg);
    ~MMDiTBlock();

    // Stubs that will be filled in as the underlying kernels land. For now
    // they assert false at runtime.
    bool forward_image_only(const void* x_bf16, void* y_bf16,
                            int batch_rows,
                            cudaStream_t stream);
    bool forward_joint     (const void* img_bf16, int img_rows,
                            const void* txt_bf16, int txt_rows,
                            void* img_out_bf16, void* txt_out_bf16,
                            cudaStream_t stream);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
