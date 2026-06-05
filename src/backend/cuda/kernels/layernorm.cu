#include "backend/cuda/kernels/layernorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>

namespace {

constexpr int BLOCK = 256;

__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  1);
    return v;
}

__device__ float block_reduce_sum(float v, float* scratch /*[32]*/) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_reduce_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const int n_warps = (BLOCK + 31) >> 5;
        v = (threadIdx.x < n_warps) ? scratch[lane] : 0.0f;
        v = warp_reduce_sum(v);
    }
    __shared__ float r;
    if (threadIdx.x == 0) r = v;
    __syncthreads();
    return r;
}

__global__ void layernorm_kernel(const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y,
                                 int dim, float eps) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const size_t base = (size_t)row * dim;
    __shared__ float scratch[32];

    // Pass 1: mean.
    float local_sum = 0.0f;
    for (int d = tid; d < dim; d += BLOCK)
        local_sum += __bfloat162float(x[base + d]);
    const float mean = block_reduce_sum(local_sum, scratch) / (float)dim;

    // Pass 2: variance.
    float local_sq = 0.0f;
    for (int d = tid; d < dim; d += BLOCK) {
        const float v = __bfloat162float(x[base + d]) - mean;
        local_sq += v * v;
    }
    const float var = block_reduce_sum(local_sq, scratch) / (float)dim;
    const float inv_std = rsqrtf(var + eps);

    // Pass 3: normalize.
    for (int d = tid; d < dim; d += BLOCK) {
        const float v = (__bfloat162float(x[base + d]) - mean) * inv_std;
        y[base + d] = __float2bfloat16(v);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool layernorm_bf16(const void* x, void* y,
                    int batch_rows, int dim, float eps,
                    cudaStream_t stream) {
    if (batch_rows <= 0 || dim <= 0) return false;
    layernorm_kernel<<<batch_rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x), static_cast<__nv_bfloat16*>(y),
        dim, eps);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
