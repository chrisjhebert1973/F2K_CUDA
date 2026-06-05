// Self-attention (Q, K, V same length) — v1.
//
//   S[b, h, q, k] = scale * sum_d Q[b, q, h, d] * K[b, k, h, d]
//   P[b, h, q, k] = softmax_k(S[b, h, q, k])
//   O[b, q, h, d] = sum_k P[b, h, q, k] * V[b, k, h, d]
//
// Shape contract — all tensors row-major BF16 in (B, S, H, D) order:
//   Q, K, V, O: [batch, seq, n_heads, head_dim]
//
// One CTA computes one row of O. Attention scores for that row live in
// shared memory through the whole softmax → WV accumulation, so the
// per-block shared-mem requirement is:
//   smem = D * sizeof(bf16) + S * sizeof(float)
// At D=128, S=1024 → ~4.4 KiB. At S=4096 → ~16.4 KiB. At S=16384 → ~65 KiB.
//
// This kernel is correctness-first; a flash-attention-style tiled version
// is the natural upgrade once we exceed shared-mem limits or need more
// throughput.

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class Attention {
public:
    struct Config {
        int   batch    = 0;
        int   seq      = 0;
        int   n_heads  = 0;
        int   head_dim = 0;
        float scale    = 0.0f;   // 0 ⇒ default to 1/sqrt(head_dim)
    };

    explicit Attention(const Config& cfg);
    ~Attention();

    bool        ok()         const;
    const char* last_error() const;

    // All device pointers.
    //   Q, K, V, O: [batch, seq, n_heads, head_dim] BF16
    bool forward(const void* Q, const void* K, const void* V,
                 void* O, cudaStream_t stream = nullptr);

private:
    Config cfg_{};
    bool   valid_ = false;
    const char* err_ = "";
};

} // namespace f2k::cuda
