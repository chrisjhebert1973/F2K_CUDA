// FLUX2-style RoPE: 4-axis (T, H, W, L) interleaved-pair rotation, in-place.
//
// Convention notes:
//   - Pair convention is INTERLEAVED `(2i, 2i+1)` (FLUX2 / diffusers
//     apply_rotary_emb with use_real_unbind_dim=-1), NOT the LLaMA-style
//     `(i, i+D/2)` we already have in rope.cu (used by Qwen3).
//   - head_dim is partitioned into K axes of axes_dim[a]; for FLUX2 it's
//     [32, 32, 32, 32] = T, H, W, L. The cos/sin tables are pre-built per
//     token, baking in each axis's position so the kernel doesn't need to
//     know about axes.
//
// Math per row:
//   pos = pos_ids[row]                             // index into seq_max
//   for k in [0, head_dim/2):
//     c = cos[pos, k];   s = sin[pos, k]
//     x'[2k]   = x[2k] * c - x[2k+1] * s
//     x'[2k+1] = x[2k] * s + x[2k+1] * c

#pragma once

#include <cstdint>
#include <vector>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// Apply the rotation. cos/sin must be FP32 device tables of shape
// [seq_max, head_dim/2]; x is BF16, in-place.
bool rope_4axis_inplace_bf16(void* x_bf16,
                              const float* cos_table,
                              const float* sin_table,
                              const int32_t* pos_ids,
                              int batch_rows,
                              int head_dim,
                              int seq_max,
                              cudaStream_t stream = nullptr);

// Build per-token cos/sin tables on the host.
//   axes_dim:        the K per-axis widths (must sum to head_dim);  for FLUX2 = {32,32,32,32}
//   token_positions: [seq, K]  per-token positions for each axis
//   theta:           RoPE base (FLUX2 = 2000.0)
// On return, cos_out and sin_out hold flat [seq, head_dim/2] FP32 tables ready
// for cudaMemcpy.
void build_rope_4axis_tables(const std::vector<int>& axes_dim,
                              const std::vector<std::vector<int>>& token_positions,
                              float theta,
                              std::vector<float>& cos_out,
                              std::vector<float>& sin_out);

} // namespace f2k::cuda
