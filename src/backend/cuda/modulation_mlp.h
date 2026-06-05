// ModulationMLP — the FLUX.2-klein timestep → per-block (scale, shift, gate)
// projection. Runs ONCE per inference step (per timestep). Outputs are then
// reused across all blocks of the corresponding type — see
// reference-flux2-klein-arch memory for why these are shared not per-block.
//
// Pipeline (all linears NVFP4 from F2K):
//
//   t = timestep_emb_bf16                             # [time_dim] e.g. 256
//   t = SiLU(Linear_1(t))                             # [hidden]   4096
//   t = SiLU(Linear_2(t))                             # [hidden]
//   double_img_out = Linear_double_img(t)             # [6*hidden] = 24576
//   double_txt_out = Linear_double_txt(t)             # [6*hidden]
//   single_out     = Linear_single(t)                 # [3*hidden] = 12288
//   norm_out_out   = Linear_norm_out(t)               # [2*hidden] = 8192
//
// Then each of those four projections is split into individual [hidden]
// vectors that the blocks consume via the modulate / gated_residual kernels.
//
// Internal padding: Linear requires M (batch_rows) % 128 == 0. The MLP
// processes batch=1, so we pad the input to M=128 internally and read row 0
// of every output. Output pointers in the Output struct point at row 0 of
// the internal buffers; they remain valid until the next forward() call (or
// the MLP is destroyed).

#pragma once

#include "backend/cuda/linear.h"

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class ModulationMLP {
public:
    struct Config {
        int hidden_dim   = 0;   // 4096
        int time_dim     = 0;   // 256

        // All weights NVFP4 from F2K.
        const PreQuantNVFP4* W_time_1     = nullptr;  // [hidden, time_dim]
        const PreQuantNVFP4* W_time_2     = nullptr;  // [hidden, hidden]
        const PreQuantNVFP4* W_double_img = nullptr;  // [6*hidden, hidden]
        const PreQuantNVFP4* W_double_txt = nullptr;  // [6*hidden, hidden]
        const PreQuantNVFP4* W_single     = nullptr;  // [3*hidden, hidden]
        const PreQuantNVFP4* W_norm_out   = nullptr;  // [2*hidden, hidden]
    };

    // Pointers into the internal output buffers, each [hidden_dim] BF16.
    // Field order matches DoubleStreamBlock::Modulation +
    // SingleStreamBlock::Modulation + the final norm_out.
    struct Output {
        // Double-stream image
        const void* img_scale_attn = nullptr;
        const void* img_shift_attn = nullptr;
        const void* img_gate_attn  = nullptr;
        const void* img_scale_mlp  = nullptr;
        const void* img_shift_mlp  = nullptr;
        const void* img_gate_mlp   = nullptr;
        // Double-stream text
        const void* txt_scale_attn = nullptr;
        const void* txt_shift_attn = nullptr;
        const void* txt_gate_attn  = nullptr;
        const void* txt_scale_mlp  = nullptr;
        const void* txt_shift_mlp  = nullptr;
        const void* txt_gate_mlp   = nullptr;
        // Single-stream
        const void* single_scale   = nullptr;
        const void* single_shift   = nullptr;
        const void* single_gate    = nullptr;
        // Final norm-out (no gate)
        const void* norm_out_scale = nullptr;
        const void* norm_out_shift = nullptr;
    };

    explicit ModulationMLP(const Config& cfg);
    ~ModulationMLP();
    ModulationMLP(const ModulationMLP&)            = delete;
    ModulationMLP& operator=(const ModulationMLP&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    size_t workspace_size_bytes() const;

    // timestep_emb_bf16: host or device pointer to BF16 [time_dim].
    //   (The MLP copies it onto the staging buffer either way.)
    bool forward(const void* timestep_emb_bf16,
                 Output& out,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
