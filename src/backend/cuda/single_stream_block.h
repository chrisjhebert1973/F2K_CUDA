// SingleStreamBlock — FLUX.2-klein single-stream MMDiT block.
//
// Forward pass (in-place on x):
//
//   norm_x  = RMSNorm(x, ones_gain)              // statistics-only, no affine
//   mod_x   = modulate(norm_x, scale, shift)
//   fused   = Linear(mod_x, W_qkv_mlp_proj)      // [B*S, 36864]
//   q,k,v,gate,up = split(fused)                 // [B*S, 4096]^3 + [B*S, 12288]^2
//   q       = RMSNorm(q, norm_q_gain) per head
//   k       = RMSNorm(k, norm_k_gain) per head
//   q,k     = RoPE(q,k)
//   attn    = Attention(q, k, v)                 // [B*S, 4096]
//   mlp_act = silu_mul(gate, up)                 // [B*S, 12288]
//   concat  = [attn || mlp_act]                  // [B*S, 16384]
//   delta   = Linear(concat, W_out)              // [B*S, 4096]
//   x       = gated_residual(x, delta, gate_mod)
//
// Hidden = n_heads * head_dim. For FLUX.2-klein: n_heads=32, head_dim=128 →
// hidden=4096, ffn=12288 (mlp_ratio=3).
//
// Modulation (scale, shift, gate_mod) is BF16 [hidden] device pointers,
// broadcast across all rows. For single-stream blocks in FLUX.2-klein the
// modulation is shared across ALL single blocks (one set per inference step).
//
// Weights are pre-quantized NVFP4 from the F2K file (PreQuantNVFP4 views).
// Norm gains are BF16 host pointers (copied to device at construction).

#pragma once

#include "backend/cuda/linear.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class SingleStreamBlock {
public:
    struct Config {
        int batch    = 0;
        int seq      = 0;
        int n_heads  = 0;
        int head_dim = 0;
        int ffn_dim  = 0;
        float rms_eps = 1e-6f;

        // Weight tensors. Pre-quantized NVFP4 for the linears; BF16 host
        // pointers for the per-head norm gains.
        const PreQuantNVFP4* W_qkv_mlp_proj = nullptr;   // [9*hidden, hidden]
        const PreQuantNVFP4* W_out          = nullptr;   // [hidden, hidden + ffn]
        const void*          norm_q_bf16    = nullptr;   // [head_dim]
        const void*          norm_k_bf16    = nullptr;   // [head_dim]
    };

    struct Modulation {
        const void* scale = nullptr;   // BF16 [hidden]
        const void* shift = nullptr;
        const void* gate  = nullptr;
    };

    explicit SingleStreamBlock(const Config& cfg);
    ~SingleStreamBlock();
    SingleStreamBlock(const SingleStreamBlock&)            = delete;
    SingleStreamBlock& operator=(const SingleStreamBlock&) = delete;

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
