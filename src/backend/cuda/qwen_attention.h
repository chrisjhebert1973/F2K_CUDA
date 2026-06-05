// Qwen3-style Grouped-Query Causal Attention.
//
//   For each (b, q_pos, q_head):
//     kv_head = q_head / group_size   (group_size = n_heads / n_kv_heads)
//     score[k] = (Q[b, q_pos, q_head] · K[b, k, kv_head]) / sqrt(D)   for k ≤ q_pos
//                                                    -inf            for k >  q_pos
//     p[k] = softmax_k(score[k])
//     O[b, q_pos, q_head, d] = sum_k p[k] * V[b, k, kv_head, d]
//
// Q is already q_norm + RoPE'd; K is already k_norm + RoPE'd.
// All tensors are BF16; scores accumulate in FP32.
//
// One CTA = one (b, q_pos, q_head). Per-block smem holds the Q row, scores,
// and the warp-reduce scratch.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

struct QwenGQAAttention {
    struct Config {
        int batch       = 0;
        int seq         = 0;
        int n_heads     = 0;
        int n_kv_heads  = 0;
        int head_dim    = 0;
        float scale     = 0.0f;   // 0 ⇒ default to 1/sqrt(head_dim)
    };

    // Q, K, V, O all device pointers, BF16.
    //   Q: [batch, seq, n_heads,    head_dim]
    //   K: [batch, seq, n_kv_heads, head_dim]
    //   V: [batch, seq, n_kv_heads, head_dim]
    //   O: [batch, seq, n_heads,    head_dim]
    static bool forward(const Config& cfg,
                        const void* Q, const void* K, const void* V,
                        void* O, cudaStream_t stream = nullptr);
};

} // namespace f2k::cuda
