// Qwen3-8B text encoder: token_ids → 36-layer forward → [seq, 3*hidden]
// conditioning for FLUX.2.
//
//   1. embed_tokens[id]    → x : [seq, hidden]
//   2. for i in 0..35:
//        x = QwenLayer[i].forward(x)
//        if i in capture_layers: stash a copy
//   3. x = final_norm(x)    (RMSNorm)
//   4. output[s, k] = capture_layers[k][s]    concatenated along feature dim
//
// The encoder owns its own RoPE tables, pos_ids, and a single shared workspace
// reused across all 36 layers.

#pragma once

#include "common/f2k_model_loader.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class QwenEncoder {
public:
    struct Config {
        // Architectural constants for Qwen3-8B.
        int seq          = 0;       // prompt length (padded)
        int hidden       = 4096;
        int n_heads      = 32;
        int n_kv_heads   = 8;
        int head_dim     = 128;
        int ffn_dim      = 12288;
        int n_layers     = 36;
        int vocab_size   = 151936;
        float rms_eps    = 1e-6f;
        float rope_theta = 1e6f;
        // Which 3 layer indices' hidden states to concatenate as output.
        // Defaults to (n_layers - 3, n_layers - 2, n_layers - 1).
        // Set to (-1, -1, -1) to skip capture and just return the final state x3.
        std::array<int, 3> capture_layers = { -1, -1, -1 };
        // Source for weights (must outlive the encoder).
        const f2k::F2KModelLoader* loader = nullptr;
        std::string prefix = "model";   // tensors are "model.layers.{i}..." etc.
    };

    explicit QwenEncoder(const Config& cfg);
    ~QwenEncoder();
    QwenEncoder(const QwenEncoder&)            = delete;
    QwenEncoder& operator=(const QwenEncoder&) = delete;

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    int output_hidden() const;   // 3 * hidden (= 12288 for Qwen3)

    // token_ids: device int32 [seq]
    // out:       device BF16  [seq, 3*hidden]
    bool forward(const int32_t* token_ids, void* out_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
