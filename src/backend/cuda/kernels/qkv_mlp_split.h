// Helper kernels for the FLUX.2-klein single-stream block, whose fused
// `to_qkv_mlp_proj` produces all of Q, K, V plus the MLP gate and up in one
// [B*S, 36864] tensor. We split it before attention/MLP, then concatenate
// (attn_out || silu(gate)*up) before the to_out projection.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// fused [rows, hidden + hidden + hidden + 2*ffn]  (= 9*hidden when ffn=3*hidden)
//   q[rows, hidden]     ← fused[:, 0 .. hidden)
//   k[rows, hidden]     ← fused[:, hidden .. 2*hidden)
//   v[rows, hidden]     ← fused[:, 2*hidden .. 3*hidden)
//   gate[rows, ffn]     ← fused[:, 3*hidden .. 3*hidden + ffn)
//   up[rows, ffn]       ← fused[:, 3*hidden + ffn .. 3*hidden + 2*ffn)
bool split_qkv_mlp_bf16(const void* fused, void* q, void* k, void* v,
                        void* gate, void* up,
                        int rows, int hidden, int ffn,
                        cudaStream_t stream = nullptr);

// concat[rows, hidden + ffn] = [attn_out[rows, hidden] || mlp_act[rows, ffn]]
bool concat_attn_mlp_bf16(const void* attn_out, const void* mlp_act,
                          void* concat_out,
                          int rows, int hidden, int ffn,
                          cudaStream_t stream = nullptr);

// GeGLU gate||up split. fused [rows, 2*half] → gate [rows, half] + up [rows, half].
// gate := fused[:, 0..half),  up := fused[:, half..2*half).
bool split_half_bf16(const void* fused, void* gate, void* up,
                     int rows, int half,
                     cudaStream_t stream = nullptr);

} // namespace f2k::cuda
