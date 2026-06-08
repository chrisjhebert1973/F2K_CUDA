// FLUX.2-klein VAE encoder — turns a [B, 3, 8H, 8W] image (BF16, model space
// [-1, 1]) into [B, 64, H, W] latent "moments" (mean ‖ logvar concatenated on
// the channel axis). The deterministic latent used for img2img is the mean,
// i.e. the first 32 channels of the output. Mirror of VAEDecoder.
//
// Architecture (block_out_channels = [128, 256, 512, 512], layers_per_block=2):
//   conv_in: 3→128 (3x3)
//   down_blocks[0]: 2× resnet 128→128 + downsample 128→128            (½)
//   down_blocks[1]: resnet 128→256, resnet 256→256 + downsample 256   (¼)
//   down_blocks[2]: resnet 256→512, resnet 512→512 + downsample 512   (⅛)
//   down_blocks[3]: 2× resnet 512→512                       (no downsample)
//   mid_block: resnet → attention → resnet (all 512 ch)
//   conv_norm_out + SiLU + conv_out: 512→64 (3x3)
//   quant_conv: 64→64 (1x1)
//
// Downsampling matches diffusers' Downsample2D with padding=0: the input is
// asymmetrically padded by (0,1,0,1) (right + bottom) before a 3x3 stride-2
// conv with no padding — NOT a symmetric pad-1 conv.

#pragma once

#include "common/f2k_format.h"

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class VAEEncoder {
public:
    struct Config {
        int N = 1;        // batch
        int H_in = 0;     // image height (latent output is H_in/8)
        int W_in = 0;
        // Prefix into the F2K file (typically "encoder").
        std::string prefix = "encoder";
        // 1×1 post-encoder conv producing the moments (sibling of `prefix`).
        // Skipped (identity) if the tensor is absent or this is empty.
        std::string quant_conv_name = "quant_conv";
        // Source of weights (mmapped F2K file). Borrowed; must outlive this.
        const f2k::F2KReader* reader = nullptr;
    };

    explicit VAEEncoder(const Config& cfg);
    ~VAEEncoder();
    VAEEncoder(const VAEEncoder&)            = delete;
    VAEEncoder& operator=(const VAEEncoder&) = delete;

    bool        ok()         const;
    const char* last_error() const;
    size_t      workspace_size_bytes() const;

    int latent_H()  const;    // H_in / 8
    int latent_W()  const;    // W_in / 8
    int output_C()  const;    // 64 (moments); mean = channels [0,32)

    // image:    [N, 3,  H_in,    W_in]      BF16, device, model space [-1,1]
    // moments:  [N, 64, H_in/8,  W_in/8]    BF16, device (mean ‖ logvar)
    bool forward(const void* image_bf16, void* moments_bf16,
                 void* workspace, size_t ws_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
