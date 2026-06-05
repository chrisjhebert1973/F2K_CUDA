// FLUX.2-klein VAE decoder — turns a [B, 32, H, W] latent into a [B, 3, 8H, 8W]
// image in BF16. Reads its weights from an F2KReader and lays out an internal
// workspace shared across all stages.
//
// Architecture:
//   conv_in: 32→512 (3x3)
//   mid_block: resnet → attention → resnet (all 512 ch)
//   up_blocks[0]: 3× resnet 512→512 + upsample 2× + conv 512→512
//   up_blocks[1]: 3× resnet 512→512 + upsample 2× + conv 512→512
//   up_blocks[2]: resnet 512→256, 2× resnet 256→256 + upsample 2× + conv 256→256
//   up_blocks[3]: resnet 256→128, 2× resnet 128→128                  (no upsample)
//   conv_norm_out + SiLU + conv_out: 128→3 (3x3)

#pragma once

#include "common/f2k_format.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class VAEDecoder {
public:
    struct Config {
        int N = 1;        // batch
        int H_lat = 0;    // latent spatial (output is 8× this)
        int W_lat = 0;
        // Prefix into the F2K file (typically "decoder").
        std::string prefix = "decoder";
        // Name of the pre-decoder 1×1 conv (sibling of `prefix`, NOT under it).
        // diffusers' Flux2 VAE runs `z = post_quant_conv(z)` before the decoder
        // (autoencoder_kl_flux2.py _decode). If the tensor is absent it's
        // skipped (identity). Set empty to force-skip.
        std::string post_quant_conv_name = "post_quant_conv";
        // Source of weights (mmapped F2K file). Pointer is borrowed; must
        // outlive this decoder.
        const f2k::F2KReader* reader = nullptr;
    };

    explicit VAEDecoder(const Config& cfg);
    ~VAEDecoder();
    VAEDecoder(const VAEDecoder&)            = delete;
    VAEDecoder& operator=(const VAEDecoder&) = delete;

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    int output_H() const;     // 8 × H_lat
    int output_W() const;     // 8 × W_lat

    // x:       [N, 32, H_lat,    W_lat]      BF16, device
    // pixels:  [N,  3, 8*H_lat,  8*W_lat]    BF16, device
    bool forward(const void* x_bf16, void* pixels_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
