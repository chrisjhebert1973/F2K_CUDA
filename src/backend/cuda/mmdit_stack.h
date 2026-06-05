// MMDiTStack — a stack of MMDiTBlock layers.
//
// Owns N MMDiTBlock instances and runs them sequentially in-place on x. All
// blocks share the same shape config; only their weights differ.
//
// Workspace lifetime: each block's internal buffers (Q/K/V/attn/proj/...) die
// at the end of its forward, so a single workspace (sized to the per-block
// requirement) is reused across all N blocks.
//
// Modulation: one set of (scale, shift, gate) per block per sub-residual,
// passed as flat arrays of device pointers (num_blocks entries each).

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class MMDiTStack {
public:
    struct Config {
        int batch    = 0;
        int seq      = 0;
        int n_heads  = 0;
        int head_dim = 0;
        int ffn_dim  = 0;
        int num_blocks = 0;
        float rms_eps  = 1e-6f;

        // Each array has num_blocks entries; per-block host weight pointers.
        const void* const* W_norm1 = nullptr;
        const void* const* W_q     = nullptr;
        const void* const* W_k     = nullptr;
        const void* const* W_v     = nullptr;
        const void* const* W_proj  = nullptr;
        const void* const* W_norm2 = nullptr;
        const void* const* W_gate  = nullptr;
        const void* const* W_up    = nullptr;
        const void* const* W_down  = nullptr;
    };

    struct Modulation {
        // Each array has num_blocks entries; per-block device pointers to
        // BF16 [hidden_dim] vectors.
        const void* const* scale_attn = nullptr;
        const void* const* shift_attn = nullptr;
        const void* const* gate_attn  = nullptr;
        const void* const* scale_mlp  = nullptr;
        const void* const* shift_mlp  = nullptr;
        const void* const* gate_mlp   = nullptr;
    };

    explicit MMDiTStack(const Config& cfg);
    ~MMDiTStack();
    MMDiTStack(const MMDiTStack&)            = delete;
    MMDiTStack& operator=(const MMDiTStack&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    int    num_blocks()             const;
    size_t workspace_size_bytes()   const;

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
