// NCHW ↔ NSC (= [N, S=H*W, C]) transpose for BF16 tensors.
//
// Used by VAE attention to feed Linear projections (which expect rows of
// [S, C]) and to splat results back into spatial layout for convs.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// x:[N, C, H, W] → y:[N, H*W, C]
bool nchw_to_nsc_bf16(const void* x_bf16, void* y_bf16,
                      int N, int C, int H, int W,
                      cudaStream_t stream = nullptr);

// x:[N, H*W, C] → y:[N, C, H, W]
bool nsc_to_nchw_bf16(const void* x_bf16, void* y_bf16,
                      int N, int C, int H, int W,
                      cudaStream_t stream = nullptr);

} // namespace f2k::cuda
