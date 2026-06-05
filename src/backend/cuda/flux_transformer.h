// FluxTransformer — top-level orchestrator wiring every component of the
// FLUX.2-klein MMDiT into a single forward() call.
//
// Pipeline:
//
//   img_latent [B*S_img, in_channels=128]
//        │ ImageEmbedder
//        ▼
//   img_hidden [B*S_img, hidden=4096]      txt_emb [B*S_txt, t5_dim=12288]
//        │                                     │ ContextEmbedder
//        │                                     ▼
//        │                              txt_hidden [B*S_txt, hidden]
//        │                                     │
//        │   timestep_emb [time_dim=256]       │
//        │        │ ModulationMLP              │
//        │        ▼                            │
//        │   17 modulation vectors             │
//        │                                     │
//        ├──── × 8 DoubleStreamBlock ──────────┤
//        │                                     │
//        │  (concat [txt_hidden || img_hidden])│
//        │           ▼                         │
//        │   combined [B*(S_img+S_txt), hidden]│
//        │           │                         │
//        │      × 24 SingleStreamBlock         │
//        │           │                         │
//        │  (take tail S_img rows = image)     │
//        ▼           │                         │
//   img_hidden_final ◄                          │
//        │ FinalProjection                     │
//        ▼                                     │
//   img_latent_out [B*S_img, in_channels=128]
//
// All weights come from a constructed TensorRouter over the F2K shards.

#pragma once

#include "common/tensor_router.h"
#include "backend/cuda/linear.h"   // f2k::cuda::Precision

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class FluxTransformer {
public:
    struct Config {
        int   batch         = 1;
        int   seq_img       = 0;       // image token sequence length (= H_patches * W_patches)
        int   seq_txt       = 0;       // text token sequence length
        // Patch grid dims for the image-stream RoPE. If left 0, defaults to
        // sqrt(seq_img) × sqrt(seq_img) (square grid). Must satisfy
        // H_patches * W_patches == seq_img.
        int   H_patches     = 0;
        int   W_patches     = 0;
        int   in_channels   = 128;     // image latent channels
        int   t5_dim        = 12288;
        int   time_dim      = 256;
        int   n_heads       = 32;
        int   head_dim      = 128;
        int   ffn_dim       = 12288;
        int   num_double_blocks = 8;
        int   num_single_blocks = 24;
        float rms_eps       = 1e-6f;
        float rope_theta    = 2000.0f;

        // Quantization precision for all internal Linears. MXFP8 requires the
        // router to be built over a BF16 F2K (Linear quantizes to FP8 at
        // construction); NVFP4 uses the pre-quantized F2K. See project-fp8-vs-fp4.
        Precision precision = Precision::NVFP4;

        // Weights, owned externally — typically the TensorRouter built over
        // your F2K shards.
        const TensorRouter* router = nullptr;
    };

    explicit FluxTransformer(const Config& cfg);
    ~FluxTransformer();
    FluxTransformer(const FluxTransformer&)            = delete;
    FluxTransformer& operator=(const FluxTransformer&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    size_t workspace_size_bytes() const;

    // image_latent_bf16:    [batch * seq_img, in_channels] BF16, device
    // text_emb_bf16:        [batch * seq_txt, t5_dim] BF16, device
    // timestep_emb_bf16:    [time_dim] BF16, host or device
    // image_latent_out_bf16:[batch * seq_img, in_channels] BF16, device
    bool forward(const void* image_latent_bf16,
                 const void* text_emb_bf16,
                 const void* timestep_emb_bf16,
                 void*       image_latent_out_bf16,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
