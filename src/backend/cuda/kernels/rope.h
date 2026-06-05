// Rotary Position Embedding (1D, in-place), "half-rotation" / LLaMA pairing.
//
// Treats the input as a flat sequence of "rows" each of length head_dim and
// pairs `x[i]` with `x[i+D/2]` (NOT the interleaved (2i, 2i+1) layout):
//   x'[i]      = x[i]     * cos[pos, i] - x[i+D/2] * sin[pos, i]
//   x'[i+D/2]  = x[i]     * sin[pos, i] + x[i+D/2] * cos[pos, i]
//
// This matches the LLaMA / Qwen3 HF convention. The cos/sin tables are
// indexed only by the lower half (`half_dim = D/2`); the upper half reuses
// the same row entries because cos/sin are even/odd-symmetric across the pair.
//
// Shape contract:
//   x:   [batch_rows, head_dim] BF16, contiguous along head_dim
//        batch_rows = batch * seq * num_heads (flattened)
//        The caller supplies pos_ids so the kernel knows the row->pos mapping.
//   cos: [seq_len, head_dim/2] FP32
//   sin: [seq_len, head_dim/2] FP32
//   pos_ids: [batch_rows] int — sequence position for each row.
//
// head_dim must be even and ≤ 256 in this v1 (we keep one CTA per row).

#pragma once

#include <cstdint>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

bool rope_inplace_bf16(void*       x_bf16,
                       const float* cos_table,
                       const float* sin_table,
                       const int32_t* pos_ids,
                       int batch_rows,
                       int head_dim,
                       int seq_len,
                       cudaStream_t stream = nullptr);

} // namespace f2k::cuda
