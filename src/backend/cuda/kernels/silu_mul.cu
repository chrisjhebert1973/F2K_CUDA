// SiLU(gate) * up elementwise BF16 kernel.

#include "backend/cuda/kernels/silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_SIZE = 256;

__global__ void silu_mul_kernel(const __nv_bfloat16* __restrict__ gate,
                                const __nv_bfloat16* __restrict__ up,
                                __nv_bfloat16* __restrict__ out,
                                size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    const float g = __bfloat162float(gate[i]);
    const float u = __bfloat162float(up[i]);
    // SiLU(x) = x * sigmoid(x) = x / (1 + exp(-x))
    const float silu_g = g / (1.0f + __expf(-g));
    out[i] = __float2bfloat16(silu_g * u);
}

__global__ void silu_inplace_kernel(__nv_bfloat16* __restrict__ x, size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * BLOCK_SIZE + threadIdx.x;
    if (i >= n) return;
    const float v = __bfloat162float(x[i]);
    x[i] = __float2bfloat16(v / (1.0f + __expf(-v)));
}

} // anonymous namespace

namespace f2k::cuda {

bool silu_mul_bf16(const void* gate, const void* up, void* out,
                   size_t n_elements, cudaStream_t stream) {
    if (n_elements == 0) return true;
    const size_t grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_mul_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(up),
        static_cast<      __nv_bfloat16*>(out),
        n_elements);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool silu_inplace_bf16(void* x, size_t n_elements, cudaStream_t stream) {
    if (n_elements == 0) return true;
    const size_t grid = (n_elements + BLOCK_SIZE - 1) / BLOCK_SIZE;
    silu_inplace_kernel<<<grid, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x), n_elements);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
