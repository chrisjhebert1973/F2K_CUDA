// Affine-free LayerNorm: y = (x - mean(x)) / sqrt(var(x) + eps),
// per-row over the trailing `dim`. Used by FLUX2's pre-attention and pre-MLP
// norms (`nn.LayerNorm(dim, elementwise_affine=False, eps=eps)`). Distinct
// from RMSNorm: LayerNorm subtracts the row mean BEFORE the divide, which
// matters once the residual stream drifts off zero.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// x: [batch_rows, dim] BF16, y: same. Aliasing allowed (in-place OK).
bool layernorm_bf16(const void* x_bf16, void* y_bf16,
                    int batch_rows, int dim, float eps,
                    cudaStream_t stream = nullptr);

} // namespace f2k::cuda
