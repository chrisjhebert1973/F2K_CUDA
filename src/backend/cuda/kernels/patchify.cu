#include "backend/cuda/kernels/patchify.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// Diffusers Flux2 channel-packing convention:
//   tokens[b, h_idx*W_p + w_idx, c*p*p + ph*p + pw] = x[b, c, h_idx*p+ph, w_idx*p+pw]
//
// Decompose idx_in_token = c*(p*p) + ph*p + pw  →  c slowest, pw fastest.
__device__ __forceinline__ void decompose_idx(int idx, int p,
                                              int& c, int& ph, int& pw) {
    const int pp = p * p;
    c  = idx / pp;
    const int rem = idx - c * pp;
    ph = rem / p;
    pw = rem - ph * p;
}

__global__ void patchify_kernel(const __nv_bfloat16* __restrict__ x,
                                __nv_bfloat16* __restrict__ tokens,
                                int N, int C, int H, int W, int p) {
    const int Hp = H / p;
    const int Wp = W / p;
    const int C_pkt = p * p * C;

    const int hw_idx = blockIdx.x;  // 0..Hp*Wp-1
    const int n      = blockIdx.y;
    const int idx_in_token = threadIdx.x + blockIdx.z * blockDim.x;
    if (n >= N || hw_idx >= Hp * Wp || idx_in_token >= C_pkt) return;

    const int h_idx = hw_idx / Wp;
    const int w_idx = hw_idx - h_idx * Wp;

    int c, ph, pw;
    decompose_idx(idx_in_token, p, c, ph, pw);

    const int h = h_idx * p + ph;
    const int w = w_idx * p + pw;

    const size_t x_off     = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
    const size_t tokens_off = (static_cast<size_t>(n) * Hp * Wp + hw_idx) * C_pkt + idx_in_token;
    tokens[tokens_off] = x[x_off];
}

__global__ void unpatchify_kernel(const __nv_bfloat16* __restrict__ tokens,
                                  __nv_bfloat16* __restrict__ x,
                                  int N, int C, int H, int W, int p) {
    const int Hp = H / p;
    const int Wp = W / p;
    const int C_pkt = p * p * C;

    const int hw_idx = blockIdx.x;
    const int n      = blockIdx.y;
    const int idx_in_token = threadIdx.x + blockIdx.z * blockDim.x;
    if (n >= N || hw_idx >= Hp * Wp || idx_in_token >= C_pkt) return;

    const int h_idx = hw_idx / Wp;
    const int w_idx = hw_idx - h_idx * Wp;

    int c, ph, pw;
    decompose_idx(idx_in_token, p, c, ph, pw);

    const int h = h_idx * p + ph;
    const int w = w_idx * p + pw;

    const size_t x_off     = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
    const size_t tokens_off = (static_cast<size_t>(n) * Hp * Wp + hw_idx) * C_pkt + idx_in_token;
    x[x_off] = tokens[tokens_off];
}

} // anonymous namespace

namespace f2k::cuda {

bool patchify_bf16(const void* x, void* tokens,
                   int N, int C, int H, int W, int p, cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || p <= 0) return false;
    if (H % p != 0 || W % p != 0) return false;
    const int C_pkt = p * p * C;
    const int BLOCK = 128;
    dim3 grid(static_cast<unsigned>((H / p) * (W / p)),
              static_cast<unsigned>(N),
              static_cast<unsigned>((C_pkt + BLOCK - 1) / BLOCK));
    patchify_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(tokens),
        N, C, H, W, p);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool unpatchify_bf16(const void* tokens, void* x,
                     int N, int C, int H, int W, int p, cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || p <= 0) return false;
    if (H % p != 0 || W % p != 0) return false;
    const int C_pkt = p * p * C;
    const int BLOCK = 128;
    dim3 grid(static_cast<unsigned>((H / p) * (W / p)),
              static_cast<unsigned>(N),
              static_cast<unsigned>((C_pkt + BLOCK - 1) / BLOCK));
    unpatchify_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(tokens),
        static_cast<__nv_bfloat16*>(x),
        N, C, H, W, p);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
