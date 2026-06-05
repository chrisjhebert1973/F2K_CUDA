// FLUX2-style 2D patchify / unpatchify (BF16, p=2 fixed for now).
//
// Matches diffusers Flux2Pipeline._patchify_latents + _pack_latents:
//   tokens[b, h_idx*W_p + w_idx, c*p*p + ph*p + pw] = x[b, c, h_idx*p+ph, w_idx*p+pw]
//
// (c is the SLOWEST-varying axis in the packed channel dim; pw the FASTEST.)
// W_p = W/p, H_p = H/p, C is the latent channel count (32 for FLUX VAE),
// and the packed dim is C_pkt = p*p*C (= 128 for p=2, C=32).

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// x:[N, C, H, W]   →   tokens:[N, (H/p)*(W/p), p*p*C]
bool patchify_bf16(const void* x_bf16, void* tokens_bf16,
                   int N, int C, int H, int W, int p,
                   cudaStream_t stream = nullptr);

// tokens:[N, (H/p)*(W/p), p*p*C]   →   x:[N, C, H, W]
bool unpatchify_bf16(const void* tokens_bf16, void* x_bf16,
                     int N, int C, int H, int W, int p,
                     cudaStream_t stream = nullptr);

} // namespace f2k::cuda
