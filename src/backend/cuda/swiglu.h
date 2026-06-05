// SwiGLU MLP block:
//
//   y = (silu(x @ W_gate^T) * (x @ W_up^T)) @ W_down^T
//
// Three NVFP4 Linears + one fused silu_mul. Owns the three Linear instances
// internally; the caller hands in BF16 weights at construction.
//
// Shape contract:
//   x        : [batch_rows, hidden_dim] BF16
//   W_gate   : [ffn_dim,    hidden_dim] BF16
//   W_up     : [ffn_dim,    hidden_dim] BF16
//   W_down   : [hidden_dim, ffn_dim   ] BF16
//   y        : [batch_rows, hidden_dim] BF16
//
// Dimensions must satisfy each underlying Linear's constraints
// (batch_rows % 128 == 0, hidden_dim % 64 == 0, ffn_dim % 128 == 0).

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class SwiGLU {
public:
    struct Config {
        int batch_rows  = 0;
        int hidden_dim  = 0;
        int ffn_dim     = 0;
        const void* W_gate_bf16 = nullptr;
        const void* W_up_bf16   = nullptr;
        const void* W_down_bf16 = nullptr;
    };

    explicit SwiGLU(const Config& cfg);
    ~SwiGLU();
    SwiGLU(const SwiGLU&)            = delete;
    SwiGLU& operator=(const SwiGLU&) = delete;

    bool        ok()          const;
    const char* last_error()  const;

    size_t workspace_size_bytes() const;

    bool forward(const void* x_bf16, void* y_bf16,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
