// Sequence-level helpers used to bridge the dual-stream and single-stream
// halves of the FLUX.2-klein transformer.
//
// concat_two_streams_bf16:
//   A[B, S_a, D] || B[B, S_b, D]  →  out[B, S_a + S_b, D]
//
// take_tail_bf16:
//   src[B, S_a + S_b, D]  →  dst[B, S_b, D]   (the last S_b rows per batch)
//
// Both are BF16 element copies; D includes the full hidden dim, no per-head
// concept here.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

bool concat_two_streams_bf16(const void* A, const void* B, void* out,
                             int batch, int S_a, int S_b, int D,
                             cudaStream_t stream = nullptr);

bool take_tail_bf16(const void* src, void* dst,
                    int batch, int S_a, int S_b, int D,
                    cudaStream_t stream = nullptr);

} // namespace f2k::cuda
