#include "backend/cuda/kernels/transpose_chw.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

// One block per (n, hw_tile, c_tile). Each thread copies one element.
// We launch with grid.x = ceil(S/32), grid.y = ceil(C/32), grid.z = N, threads = (32,32).
// Each block reads a 32×32 tile from one layout and writes to the other.

__global__ void nchw_to_nsc_kernel(const __nv_bfloat16* __restrict__ x,
                                   __nv_bfloat16* __restrict__ y,
                                   int N, int C, int H, int W) {
    const int S = H * W;
    const int s_base = blockIdx.x * 32;
    const int c_base = blockIdx.y * 32;
    const int n      = blockIdx.z;
    const int s = s_base + threadIdx.x;
    const int c = c_base + threadIdx.y;
    if (n >= N || s >= S || c >= C) return;
    const int h = s / W;
    const int w = s - h * W;
    const size_t x_idx = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
    const size_t y_idx = (static_cast<size_t>(n) * S + s) * C + c;
    y[y_idx] = x[x_idx];
}

__global__ void nsc_to_nchw_kernel(const __nv_bfloat16* __restrict__ x,
                                   __nv_bfloat16* __restrict__ y,
                                   int N, int C, int H, int W) {
    const int S = H * W;
    const int s_base = blockIdx.x * 32;
    const int c_base = blockIdx.y * 32;
    const int n      = blockIdx.z;
    const int s = s_base + threadIdx.x;
    const int c = c_base + threadIdx.y;
    if (n >= N || s >= S || c >= C) return;
    const int h = s / W;
    const int w = s - h * W;
    const size_t x_idx = (static_cast<size_t>(n) * S + s) * C + c;
    const size_t y_idx = ((static_cast<size_t>(n) * C + c) * H + h) * W + w;
    y[y_idx] = x[x_idx];
}

} // anonymous namespace

namespace f2k::cuda {

bool nchw_to_nsc_bf16(const void* x, void* y,
                      int N, int C, int H, int W, cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return false;
    const int S = H * W;
    dim3 grid((S + 31) / 32, (C + 31) / 32, N);
    dim3 block(32, 32);
    nchw_to_nsc_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(y),
        N, C, H, W);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool nsc_to_nchw_bf16(const void* x, void* y,
                      int N, int C, int H, int W, cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return false;
    const int S = H * W;
    dim3 grid((S + 31) / 32, (C + 31) / 32, N);
    dim3 block(32, 32);
    nsc_to_nchw_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(y),
        N, C, H, W);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
