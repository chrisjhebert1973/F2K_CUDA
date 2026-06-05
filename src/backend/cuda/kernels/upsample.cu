#include "backend/cuda/kernels/upsample.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int BLOCK = 256;

__global__ void upsample2x_kernel(const __nv_bfloat16* __restrict__ x,
                                  __nv_bfloat16* __restrict__ y,
                                  int N, int C, int H, int W) {
    const int Wy = W * 2;
    const int Hy = H * 2;
    const int n  = blockIdx.z;
    const int c  = blockIdx.y;
    const int yh = blockIdx.x;
    if (n >= N || c >= C || yh >= Hy) return;

    const int xh = yh >> 1;
    const size_t x_row = ((static_cast<size_t>(n) * C + c) * H + xh) * W;
    const size_t y_row = ((static_cast<size_t>(n) * C + c) * Hy + yh) * Wy;

    for (int yw = threadIdx.x; yw < Wy; yw += BLOCK) {
        const int xw = yw >> 1;
        y[y_row + yw] = x[x_row + xw];
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool upsample2x_nearest_bf16(const void* x, void* y,
                              int N, int C, int H, int W, cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0) return false;
    dim3 grid(H * 2, C, N);
    upsample2x_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(y),
        N, C, H, W);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
