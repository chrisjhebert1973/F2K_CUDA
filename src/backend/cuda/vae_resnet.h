// VAE ResnetBlock — the conv-resnet primitive that fills the mid_block and
// up_blocks of the FLUX.2-klein VAE decoder.
//
//   residual = norm1(x)                            # GroupNorm 32 groups
//   residual = SiLU(residual)
//   residual = conv1(residual)                     # 3x3, optional channel change
//   residual = norm2(residual)
//   residual = SiLU(residual)
//   residual = conv2(residual)                     # 3x3, channels stay
//   skip     = x  OR  conv_shortcut(x)             # 1x1 conv when C_in ≠ C_out
//   y        = skip + residual
//
// All weights (gains, biases, conv weights) are BF16; compute is BF16 with
// FP32 accumulation inside GroupNorm and cuDNN's GEMM.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class ResnetBlock {
public:
    struct Config {
        int N = 0, C_in = 0, H = 0, W = 0;
        int C_out = 0;
        int num_groups = 32;
        float eps = 1e-6f;

        // Host pointers, BF16:
        const void* norm1_gain;   // [C_in]
        const void* norm1_bias;   // [C_in]
        const void* conv1_W;      // [C_out, C_in, 3, 3]
        const void* conv1_bias;   // [C_out]
        const void* norm2_gain;   // [C_out]
        const void* norm2_bias;   // [C_out]
        const void* conv2_W;      // [C_out, C_out, 3, 3]
        const void* conv2_bias;   // [C_out]
        // Optional shortcut weights (only when C_in != C_out):
        const void* shortcut_W;     // [C_out, C_in, 1, 1]; nullable
        const void* shortcut_bias;  // [C_out];               nullable
    };

    explicit ResnetBlock(const Config& cfg);
    ~ResnetBlock();
    ResnetBlock(const ResnetBlock&)            = delete;
    ResnetBlock& operator=(const ResnetBlock&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    size_t workspace_size_bytes() const;

    // x: [N, C_in, H, W]  BF16, device
    // y: [N, C_out, H, W] BF16, device
    bool forward(const void* x_bf16, void* y_bf16,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
