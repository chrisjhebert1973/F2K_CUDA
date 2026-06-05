// Upsample 2× nearest-neighbor on NCHW BF16.
//
//   y[n, c, 2h+a, 2w+b] = x[n, c, h, w]  for a,b ∈ {0,1}

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

bool upsample2x_nearest_bf16(const void* x_bf16, void* y_bf16,
                             int N, int C, int H, int W,
                             cudaStream_t stream = nullptr);

} // namespace f2k::cuda
