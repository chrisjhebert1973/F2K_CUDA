#include "backend/cuda/kernels/groupnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int BLOCK = 256;

__device__ __forceinline__ float warp_sum(float v) {
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        v += __shfl_xor_sync(0xFFFFFFFFu, v, off);
    return v;
}

__device__ float block_reduce_sum(float v, float* scratch /*[32]*/) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        v = (threadIdx.x < (BLOCK / 32)) ? scratch[lane] : 0.0f;
        v = warp_sum(v);
    }
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

// One CTA per (n, g). Two-pass: compute (sum, sum_sq) for mean/var, then normalize.
__global__ void groupnorm_kernel(const __nv_bfloat16* __restrict__ x,
                                 __nv_bfloat16* __restrict__ y,
                                 const __nv_bfloat16* __restrict__ gain,
                                 const __nv_bfloat16* __restrict__ bias,
                                 int N, int C, int H, int W,
                                 int num_groups, int cpg,   // channels per group
                                 float eps) {
    const int g = blockIdx.x;
    const int n = blockIdx.y;
    const int hw = H * W;
    const int group_size = cpg * hw;
    const size_t group_base = (static_cast<size_t>(n) * C + static_cast<size_t>(g) * cpg) * hw;

    __shared__ float scratch[32];

    // Pass 1: compute sum and sum of squares.
    float s = 0, ss = 0;
    for (int i = threadIdx.x; i < group_size; i += BLOCK) {
        const int local_c = i / hw;
        const int local_p = i % hw;
        const float v = __bfloat162float(x[group_base + local_c * hw + local_p]);
        s  += v;
        ss += v * v;
    }
    const float sum    = block_reduce_sum(s,  scratch);
    const float sum_sq = block_reduce_sum(ss, scratch);

    const float inv_n = 1.0f / static_cast<float>(group_size);
    const float mean = sum * inv_n;
    const float var  = sum_sq * inv_n - mean * mean;
    const float rstd = rsqrtf(var + eps);

    // Pass 2: normalize + affine.
    for (int i = threadIdx.x; i < group_size; i += BLOCK) {
        const int local_c = i / hw;
        const int local_p = i % hw;
        const int c_abs   = g * cpg + local_c;
        const float v = __bfloat162float(x[group_base + local_c * hw + local_p]);
        const float gv = __bfloat162float(gain[c_abs]);
        const float bv = __bfloat162float(bias[c_abs]);
        const float yv = (v - mean) * rstd * gv + bv;
        y[group_base + local_c * hw + local_p] = __float2bfloat16(yv);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool groupnorm_bf16(const void* x, void* y, const void* gain, const void* bias,
                    int N, int C, int H, int W, int num_groups, float eps,
                    cudaStream_t stream) {
    if (N <= 0 || C <= 0 || H <= 0 || W <= 0 || num_groups <= 0) return false;
    if (C % num_groups != 0) return false;
    const int cpg = C / num_groups;
    dim3 grid(num_groups, N);
    groupnorm_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<__nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(gain),
        static_cast<const __nv_bfloat16*>(bias),
        N, C, H, W, num_groups, cpg, eps);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
