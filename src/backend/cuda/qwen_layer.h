// Single Qwen3 decoder block.
//
//   residual = x
//   x = input_layernorm(x)
//   q = x @ Wq^T   reshape → [S, Hq,  D]
//   k = x @ Wk^T   reshape → [S, Hkv, D]
//   v = x @ Wv^T   reshape → [S, Hkv, D]
//   q = rmsnorm(q row-by-row, q_norm_gamma)
//   k = rmsnorm(k row-by-row, k_norm_gamma)
//   q = rope(q)
//   k = rope(k)
//   o = causal_gqa(q, k, v)
//   o = o.flat() @ Wo^T
//   x = residual + o
//
//   residual = x
//   x = post_attention_layernorm(x)
//   gate = x @ Wg^T  ;  up = x @ Wu^T
//   h    = silu(gate) * up
//   h    = h @ Wd^T
//   x = residual + h

#pragma once

#include "backend/cuda/linear.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class QwenLayer {
public:
    struct Config {
        int batch = 1;
        int seq = 0;
        int hidden = 0;
        int n_heads = 0;
        int n_kv_heads = 0;
        int head_dim = 0;
        int ffn_dim = 0;
        float rms_eps = 1e-6f;

        // BF16 host pointers for gains:
        const void* input_norm_gamma   = nullptr;   // [hidden]
        const void* post_norm_gamma    = nullptr;   // [hidden]
        const void* q_norm_gamma       = nullptr;   // [head_dim]
        const void* k_norm_gamma       = nullptr;   // [head_dim]

        // Pre-quantized NVFP4 weights (host pointers, typically into F2K mmap):
        const PreQuantNVFP4* q_proj   = nullptr;
        const PreQuantNVFP4* k_proj   = nullptr;
        const PreQuantNVFP4* v_proj   = nullptr;
        const PreQuantNVFP4* o_proj   = nullptr;
        const PreQuantNVFP4* gate_proj = nullptr;
        const PreQuantNVFP4* up_proj   = nullptr;
        const PreQuantNVFP4* down_proj = nullptr;
    };

    explicit QwenLayer(const Config& cfg);
    ~QwenLayer();
    QwenLayer(const QwenLayer&)            = delete;
    QwenLayer& operator=(const QwenLayer&) = delete;

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    // x, y: [batch*seq, hidden] BF16 device pointers (y can equal x for in-place).
    // workspace: shared scratch sized at workspace_size_bytes().
    // rope_cos, rope_sin: shared device pointers [seq, head_dim/2] FP32.
    // pos_ids_q: device int32 [batch*seq*n_heads]    (= row / n_heads)
    // pos_ids_k: device int32 [batch*seq*n_kv_heads] (= row / n_kv_heads)
    bool forward(const void* x_bf16, void* y_bf16,
                 const float* rope_cos, const float* rope_sin,
                 const int32_t* pos_ids_q, const int32_t* pos_ids_k,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
