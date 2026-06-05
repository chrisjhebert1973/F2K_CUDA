// RMSNorm: y[b,d] = (x[b,d] / rms(x[b,:])) * gamma[d],
// where rms(x) = sqrt(mean(x^2) + eps).
//
// All tensors are BF16. Reduction happens in FP32 for stability. One CTA
// handles one row of x; reductions are warp-shuffle based.
//
// Shape convention: x is (batch_rows, dim), gamma is (dim,).
// "batch_rows" can be (B * S) for a (batch, seq, dim) input — the kernel
// doesn't care, it just normalizes along the trailing dim.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// Launches the RMSNorm kernel. `dim` must be a multiple of 8 and ≤ 8192 in
// this first version (we pick a block size that handles the row in chunks).
// Returns true if the kernel was launched without error.
bool rmsnorm_bf16(const void* x_bf16,
                  const void* gamma_bf16,
                  void*       y_bf16,
                  int batch_rows,
                  int dim,
                  float eps,
                  cudaStream_t stream = nullptr);

} // namespace f2k::cuda
