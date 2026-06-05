// RMSNorm BF16 kernel — one CTA per row, warp-shuffle reduction, FP32
// accumulation. Tuned for "small enough" dims (≤ 8192) where one CTA can
// hold a row in registers across iterations.

#include "backend/cuda/kernels/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_SIZE = 256;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_xor_sync(0xFFFFFFFFu, v, offset);
    }
    return v;
}

__device__ __forceinline__ float block_sum(float v) {
    __shared__ float shared[BLOCK_SIZE / 32];
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;

    v = warp_sum(v);
    if (lane == 0) shared[warp] = v;
    __syncthreads();

    if (warp == 0) {
        v = (threadIdx.x < (BLOCK_SIZE / 32)) ? shared[lane] : 0.0f;
        v = warp_sum(v);
    }
    // Broadcast result from warp 0 lane 0.
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

__global__ void rmsnorm_bf16_kernel(const __nv_bfloat16* __restrict__ x,
                                    const __nv_bfloat16* __restrict__ gamma,
                                    __nv_bfloat16* __restrict__ y,
                                    int dim,
                                    float inv_dim,
                                    float eps) {
    const int row = blockIdx.x;
    const __nv_bfloat16* xr = x + static_cast<size_t>(row) * dim;
          __nv_bfloat16* yr = y + static_cast<size_t>(row) * dim;

    // Pass 1: sum of squares (FP32 accumulate)
    float partial = 0.0f;
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        const float v = __bfloat162float(xr[i]);
        partial += v * v;
    }
    const float sum_sq = block_sum(partial);
    const float rrms   = rsqrtf(sum_sq * inv_dim + eps);

    // Pass 2: scale + gamma
    for (int i = threadIdx.x; i < dim; i += BLOCK_SIZE) {
        const float v = __bfloat162float(xr[i]);
        const float g = __bfloat162float(gamma[i]);
        const float out = v * rrms * g;
        yr[i] = __float2bfloat16(out);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool rmsnorm_bf16(const void* x, const void* gamma, void* y,
                  int batch_rows, int dim, float eps, cudaStream_t stream) {
    if (batch_rows <= 0 || dim <= 0) return false;
    if (dim % 8 != 0) return false;

    const float inv_dim = 1.0f / static_cast<float>(dim);
    dim3 grid(batch_rows), block(BLOCK_SIZE);
    rmsnorm_bf16_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gamma),
        static_cast<      __nv_bfloat16*>(y),
        dim, inv_dim, eps);

    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
