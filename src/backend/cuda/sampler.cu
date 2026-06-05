#include "backend/cuda/sampler.h"

#include <cuda_runtime.h>

namespace {

constexpr int BLOCK = 256;

__global__ void axpy_kernel(__nv_bfloat16* __restrict__ y,
                            const __nv_bfloat16* __restrict__ x,
                            float alpha, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * BLOCK + threadIdx.x;
    if (i >= n) return;
    const float xf = __bfloat162float(x[i]);
    const float yf = __bfloat162float(y[i]);
    y[i] = __float2bfloat16(yf + alpha * xf);
}

} // anonymous namespace

namespace f2k::cuda {

bool axpy_bf16(void* y, const void* x, float alpha, size_t n, cudaStream_t stream) {
    if (n == 0) return true;
    const size_t grid = (n + BLOCK - 1) / BLOCK;
    axpy_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(x),
        alpha, n);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
