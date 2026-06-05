// Qwen3 GQA causal attention v1 — row-at-a-time fused kernel.

#include "backend/cuda/qwen_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

__device__ __forceinline__ float warp_reduce_max(float v) {
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v, 16));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v,  8));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v,  4));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v,  2));
    v = fmaxf(v, __shfl_xor_sync(0xFFFFFFFFu, v,  1));
    return v;
}
__device__ __forceinline__ float warp_reduce_sum(float v) {
    v += __shfl_xor_sync(0xFFFFFFFFu, v, 16);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  8);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  4);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  2);
    v += __shfl_xor_sync(0xFFFFFFFFu, v,  1);
    return v;
}

template <bool IsMax>
__device__ float block_reduce(float v, float* scratch /*[32]*/, int block_size) {
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    v = IsMax ? warp_reduce_max(v) : warp_reduce_sum(v);
    if (lane == 0) scratch[warp] = v;
    __syncthreads();
    if (warp == 0) {
        const int n_warps = (block_size + 31) >> 5;
        v = (threadIdx.x < n_warps) ? scratch[lane]
                                     : (IsMax ? -INFINITY : 0.0f);
        v = IsMax ? warp_reduce_max(v) : warp_reduce_sum(v);
    }
    __shared__ float result;
    if (threadIdx.x == 0) result = v;
    __syncthreads();
    return result;
}

__global__ void qwen_gqa_causal_kernel(const __nv_bfloat16* __restrict__ Q,
                                       const __nv_bfloat16* __restrict__ K,
                                       const __nv_bfloat16* __restrict__ V,
                                       __nv_bfloat16* __restrict__ O,
                                       int B, int S, int Hq, int Hkv, int D,
                                       float scale) {
    const int q   = blockIdx.x;   // query position
    const int qh  = blockIdx.y;   // query head
    const int b   = blockIdx.z;   // batch
    const int tid = threadIdx.x;
    const int kvh = qh * Hkv / Hq;  // group_size = Hq/Hkv; kvh = qh / group_size

    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* sQ        = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float*         sAttn     = reinterpret_cast<float*>(smem_raw + D * sizeof(__nv_bfloat16));
    float*         sScratch  = sAttn + S;

    const size_t base_q = (((size_t)b * S + q)  * Hq  + qh ) * D;

    // 1. Load Q row into smem.
    for (int d = tid; d < D; d += blockDim.x) sQ[d] = Q[base_q + d];
    __syncthreads();

    // 2. Scores for k ∈ [0, q]; -inf for k > q.
    for (int k = tid; k < S; k += blockDim.x) {
        if (k > q) { sAttn[k] = -INFINITY; continue; }
        const size_t base_k = (((size_t)b * S + k) * Hkv + kvh) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += __bfloat162float(sQ[d]) * __bfloat162float(K[base_k + d]);
        }
        sAttn[k] = dot * scale;
    }
    __syncthreads();

    // 3. Row max.
    float local_max = -INFINITY;
    for (int k = tid; k < S; k += blockDim.x) local_max = fmaxf(local_max, sAttn[k]);
    const float row_max = block_reduce<true>(local_max, sScratch, blockDim.x);

    // 4. exp(x - max), sum.
    float local_sum = 0.0f;
    for (int k = tid; k < S; k += blockDim.x) {
        const float e = (sAttn[k] == -INFINITY) ? 0.0f : __expf(sAttn[k] - row_max);
        sAttn[k] = e;
        local_sum += e;
    }
    const float total = block_reduce<false>(local_sum, sScratch, blockDim.x);
    const float inv_total = 1.0f / (total + 1e-30f);

    // 5. O[d] = sum_k p[k] * V[k, kvh, d]
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k <= q; ++k) {
            const size_t base_k = (((size_t)b * S + k) * Hkv + kvh) * D;
            acc += sAttn[k] * __bfloat162float(V[base_k + d]);
        }
        const size_t out_off = (((size_t)b * S + q) * Hq + qh) * D + d;
        O[out_off] = __float2bfloat16(acc * inv_total);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool QwenGQAAttention::forward(const Config& cfg,
                                const void* Q, const void* K, const void* V,
                                void* O, cudaStream_t stream) {
    if (cfg.batch <= 0 || cfg.seq <= 0 || cfg.n_heads <= 0 || cfg.n_kv_heads <= 0 || cfg.head_dim <= 0)
        return false;
    if (cfg.n_heads % cfg.n_kv_heads != 0) return false;
    if (cfg.head_dim > 1024) return false;

    const float scale = (cfg.scale > 0.0f) ? cfg.scale
                                            : (1.0f / std::sqrt((float)cfg.head_dim));
    const int block = (cfg.head_dim > 128) ? cfg.head_dim : 128;
    const size_t smem =
        (size_t)cfg.head_dim * sizeof(__nv_bfloat16) +
        (size_t)cfg.seq     * sizeof(float) +
        32 * sizeof(float);

    dim3 grid(cfg.seq, cfg.n_heads, cfg.batch);
    qwen_gqa_causal_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<      __nv_bfloat16*>(O),
        cfg.batch, cfg.seq, cfg.n_heads, cfg.n_kv_heads, cfg.head_dim, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
