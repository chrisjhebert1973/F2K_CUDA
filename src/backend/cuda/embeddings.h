// Embedding / un-embedding modules for the FLUX.2-klein MMDiT.
//
//   ImageEmbedder    [B*S, in_channels=128]  →  [B*S, hidden=4096]
//   ContextEmbedder  [B*S_txt, t5_dim=12288] →  [B*S_txt, hidden=4096]
//   FinalProjection  [B*S, hidden=4096]      →  [B*S, in_channels=128]
//                    (applies RMSNorm(ones) + modulate(norm_out_scale/shift)
//                     before the projection — this is the "norm_out" head.)
//
// All linears are pre-quantized NVFP4 from the F2K model file.

#pragma once

#include "backend/cuda/linear.h"

#include <cstddef>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class ImageEmbedder {
public:
    struct Config {
        int batch_rows   = 0;
        int in_channels  = 0;
        int hidden_dim   = 0;
        const PreQuantNVFP4* W = nullptr;
    };

    explicit ImageEmbedder(const Config& cfg);
    ~ImageEmbedder();

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    bool forward(const void* x_latent_bf16, void* x_hidden_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class ContextEmbedder {
public:
    struct Config {
        int batch_rows = 0;
        int t5_dim     = 0;
        int hidden_dim = 0;
        const PreQuantNVFP4* W = nullptr;
    };

    explicit ContextEmbedder(const Config& cfg);
    ~ContextEmbedder();

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    bool forward(const void* txt_emb_bf16, void* txt_hidden_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

class FinalProjection {
public:
    struct Config {
        int batch_rows   = 0;
        int hidden_dim   = 0;
        int out_channels = 0;
        float rms_eps    = 1e-6f;
        const PreQuantNVFP4* W_proj_out = nullptr;
    };

    explicit FinalProjection(const Config& cfg);
    ~FinalProjection();

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    // norm_out_scale/shift are device pointers BF16 [hidden_dim].
    bool forward(const void* x_hidden_bf16,
                 const void* norm_out_scale,
                 const void* norm_out_shift,
                 void* x_latent_out_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
