// GroupNorm — NCHW BF16 with FP32 reduction.
//
//   For each (batch n, group g):
//     μ  = mean over channels_in_group × H × W
//     σ² = var  over channels_in_group × H × W
//     y[n, c, h, w] = (x[n, c, h, w] − μ) / √(σ² + ε) · γ[c] + β[c]
//
// Where g(c) = c / (C / num_groups). γ (gain) and β (bias) are per-channel,
// not per-group. This matches PyTorch's nn.GroupNorm with affine=True.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

bool groupnorm_bf16(const void* x_bf16, void* y_bf16,
                    const void* gain_bf16, const void* bias_bf16,
                    int N, int C, int H, int W, int num_groups,
                    float eps = 1e-6f,
                    cudaStream_t stream = nullptr);

} // namespace f2k::cuda
