// VAE spatial self-attention block (the one inside mid_block).
//
//   h = group_norm(x)                           # NCHW
//   h = NCHW → NSC  (S = H*W tokens of C)
//   q = h @ Wq^T + bq
//   k = h @ Wk^T + bk
//   v = h @ Wv^T + bv
//   o = attention(q, k, v)                       # n_heads=1, head_dim=C
//   o = o @ Wo^T + bo
//   o = NSC → NCHW
//   y = x + o
//
// Constraints (cascade from our Linear): H*W % 128 == 0, C % 128 == 0.
// For FLUX VAE mid_block at the typical 16×16 spatial resolution: H*W=256, C=512 — fine.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class VAEAttention {
public:
    struct Config {
        int N = 0, C = 0, H = 0, W = 0;
        int num_groups = 32;
        float eps = 1e-6f;

        // BF16 host pointers:
        const void* norm_gain;     // [C]
        const void* norm_bias;     // [C]
        const void* to_q_W;        // [C, C]
        const void* to_q_b;        // [C]
        const void* to_k_W;        // [C, C]
        const void* to_k_b;        // [C]
        const void* to_v_W;        // [C, C]
        const void* to_v_b;        // [C]
        const void* to_out_W;      // [C, C]
        const void* to_out_b;      // [C]
    };

    explicit VAEAttention(const Config& cfg);
    ~VAEAttention();
    VAEAttention(const VAEAttention&)            = delete;
    VAEAttention& operator=(const VAEAttention&) = delete;

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    // x, y: [N, C, H, W] BF16 device pointers.
    bool forward(const void* x_bf16, void* y_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
