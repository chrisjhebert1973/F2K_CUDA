// MMDiTBlock — one transformer block of the MMDiT stack.
//
// Forward pass (in-place on x):
//
//   norm1   = RMSNorm(x, W_norm1)
//   mod1    = modulate(norm1, scale_attn, shift_attn)
//   q,k,v   = Linear_q/k/v(mod1)
//   q,k     = RoPE(q,k, cos, sin)
//   attn    = Attention(q, k, v)
//   proj    = Linear_proj(attn)
//   x       = gated_residual(x, proj, gate_attn)
//
//   norm2   = RMSNorm(x, W_norm2)
//   mod2    = modulate(norm2, scale_mlp, shift_mlp)
//   mlp     = SwiGLU(mod2)
//   x       = gated_residual(x, mlp, gate_mlp)
//
// Shape contract:
//   hidden_dim = n_heads * head_dim
//   x:          [batch * seq, hidden_dim] BF16 (modified in-place)
//   modulation: each [hidden_dim] BF16, broadcast across all rows
//   rope_cos/sin: [seq, head_dim/2] FP32
//
// All weights are BF16 on host (copied/quantized into the contained Linear
// instances). Modulation tensors and RoPE tables are device pointers.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class MMDiTBlock {
public:
    struct Config {
        int batch     = 0;   // outer batch
        int seq       = 0;
        int n_heads   = 0;
        int head_dim  = 0;
        int ffn_dim   = 0;
        float rms_eps = 1e-6f;

        // Weights (host pointers; copied/quantized at construction).
        const void* W_norm1 = nullptr;   // BF16 [hidden_dim]
        const void* W_q     = nullptr;   // BF16 [n_heads*head_dim, hidden_dim]
        const void* W_k     = nullptr;
        const void* W_v     = nullptr;
        const void* W_proj  = nullptr;   // BF16 [hidden_dim, n_heads*head_dim]
        const void* W_norm2 = nullptr;
        const void* W_gate  = nullptr;   // BF16 [ffn_dim, hidden_dim]
        const void* W_up    = nullptr;
        const void* W_down  = nullptr;   // BF16 [hidden_dim, ffn_dim]
    };

    struct Modulation {
        // All device pointers, BF16 [hidden_dim], broadcast across rows.
        const void* scale_attn = nullptr;
        const void* shift_attn = nullptr;
        const void* gate_attn  = nullptr;
        const void* scale_mlp  = nullptr;
        const void* shift_mlp  = nullptr;
        const void* gate_mlp   = nullptr;
    };

    explicit MMDiTBlock(const Config& cfg);
    ~MMDiTBlock();
    MMDiTBlock(const MMDiTBlock&)            = delete;
    MMDiTBlock& operator=(const MMDiTBlock&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    size_t workspace_size_bytes() const;

    bool forward(void* x_bf16,
                 const Modulation& mod,
                 const float* rope_cos,
                 const float* rope_sin,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
