// DoubleStreamBlock — FLUX.2-klein dual-stream (image + text) MMDiT block.
//
// Forward pass (in-place on img + txt):
//
//   # Attention sub-block (per stream)
//   img_norm = RMSNorm(img, ones)                  # no-affine
//   img_mod  = modulate(img_norm, img_scale_attn, img_shift_attn)
//   txt_norm = RMSNorm(txt, ones)
//   txt_mod  = modulate(txt_norm, txt_scale_attn, txt_shift_attn)
//
//   img_q,k,v = Linear(img_mod, W_q/k/v)           # image QKV
//   txt_q,k,v = Linear(txt_mod, W_add_q/k/v_proj)  # text  QKV
//
//   img_q,k = RMSNorm per-head with norm_q/norm_k gains
//   txt_q,k = RMSNorm per-head with norm_added_q/norm_added_k
//   img_q,k = RoPE(...)                            # text gets NO RoPE
//
//   (img_attn, txt_attn) = JointAttention(txt_q,k,v, img_q,k,v)
//
//   img_proj = Linear(img_attn, W_out)             # image output projection
//   txt_proj = Linear(txt_attn, W_to_add_out)
//
//   img = gated_residual(img, img_proj, img_gate_attn)
//   txt = gated_residual(txt, txt_proj, txt_gate_attn)
//
//   # MLP sub-block (per stream, GeGLU)
//   img_mod2 = modulate(RMSNorm(img, ones), img_scale_mlp, img_shift_mlp)
//   img_fused = Linear(img_mod2, W_ff_in)          # [2*ffn]
//   gate,up  = split_half(img_fused)               # [ffn] each
//   img_mlp  = Linear(silu(gate)*up, W_ff_out)     # [hidden]
//   img      = gated_residual(img, img_mlp, img_gate_mlp)
//
//   (same for txt with W_ff_ctx_in / W_ff_ctx_out)
//
// hidden = n_heads * head_dim. For FLUX.2-klein: n_heads=32, head_dim=128 →
// hidden=4096, ffn=12288 (mlp_ratio=3).

#pragma once

#include "backend/cuda/linear.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class DoubleStreamBlock {
public:
    struct Config {
        int batch    = 0;
        int seq_img  = 0;
        int seq_txt  = 0;
        int n_heads  = 0;
        int head_dim = 0;
        int ffn_dim  = 0;
        float rms_eps = 1e-6f;

        // ---- Image stream weights ----
        const PreQuantNVFP4* W_q       = nullptr;   // [hidden, hidden]
        const PreQuantNVFP4* W_k       = nullptr;
        const PreQuantNVFP4* W_v       = nullptr;
        const PreQuantNVFP4* W_out     = nullptr;   // [hidden, hidden]
        const PreQuantNVFP4* W_ff_in   = nullptr;   // [2*ffn, hidden]
        const PreQuantNVFP4* W_ff_out  = nullptr;   // [hidden, ffn]
        const void*          norm_q_bf16 = nullptr; // [head_dim]
        const void*          norm_k_bf16 = nullptr;

        // ---- Text stream weights ----
        const PreQuantNVFP4* W_add_q     = nullptr;
        const PreQuantNVFP4* W_add_k     = nullptr;
        const PreQuantNVFP4* W_add_v     = nullptr;
        const PreQuantNVFP4* W_add_out   = nullptr;
        const PreQuantNVFP4* W_ff_ctx_in  = nullptr;
        const PreQuantNVFP4* W_ff_ctx_out = nullptr;
        const void*          norm_added_q_bf16 = nullptr;
        const void*          norm_added_k_bf16 = nullptr;
    };

    struct Modulation {
        // Image
        const void* img_scale_attn = nullptr;
        const void* img_shift_attn = nullptr;
        const void* img_gate_attn  = nullptr;
        const void* img_scale_mlp  = nullptr;
        const void* img_shift_mlp  = nullptr;
        const void* img_gate_mlp   = nullptr;
        // Text
        const void* txt_scale_attn = nullptr;
        const void* txt_shift_attn = nullptr;
        const void* txt_gate_attn  = nullptr;
        const void* txt_scale_mlp  = nullptr;
        const void* txt_shift_mlp  = nullptr;
        const void* txt_gate_mlp   = nullptr;
    };

    explicit DoubleStreamBlock(const Config& cfg);
    ~DoubleStreamBlock();
    DoubleStreamBlock(const DoubleStreamBlock&)            = delete;
    DoubleStreamBlock& operator=(const DoubleStreamBlock&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    size_t workspace_size_bytes() const;

    // rope_cos_img / rope_sin_img: [seq_img, head_dim/2] FP32 — 4-axis RoPE
    //   tables for image tokens (positions (0, h_idx, w_idx, 0)).
    // rope_cos_txt / rope_sin_txt: [seq_txt, head_dim/2] FP32 — text tokens
    //   (positions (0, 0, 0, l_idx)).
    bool forward(void* img_bf16, void* txt_bf16,
                 const Modulation& mod,
                 const float* rope_cos_img, const float* rope_sin_img,
                 const float* rope_cos_txt, const float* rope_sin_txt,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
