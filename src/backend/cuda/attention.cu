// Self-attention v1 — row-at-a-time fused kernel.

#include "backend/cuda/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>

namespace {

constexpr int MAX_BLOCK = 256;

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

// Block-wide max/sum, BLOCK threads. Uses smem[32] scratch.
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

// One CTA = one (b, h, q) — computes one row of O.
// Grid: (seq, n_heads, batch).
__global__ void attention_row_kernel(const __nv_bfloat16* __restrict__ Q,
                                     const __nv_bfloat16* __restrict__ K,
                                     const __nv_bfloat16* __restrict__ V,
                                     __nv_bfloat16* __restrict__ O,
                                     int B, int S, int H, int D,
                                     float scale) {
    const int q = blockIdx.x;
    const int h = blockIdx.y;
    const int b = blockIdx.z;
    const int tid = threadIdx.x;

    extern __shared__ uint8_t smem_raw[];
    __nv_bfloat16* sQ    = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    float*         sAttn = reinterpret_cast<float*>(smem_raw + D * sizeof(__nv_bfloat16));
    // 32-slot scratch for block reductions, placed after sAttn.
    float*         sScratch = sAttn + S;

    const size_t bhsd = static_cast<size_t>(B) * S * H * D;  // unused but explicit
    (void)bhsd;
    const size_t base_q = ((static_cast<size_t>(b) * S + q) * H + h) * D;

    // 1. Load Q row into shared memory.
    for (int d = tid; d < D; d += blockDim.x) {
        sQ[d] = Q[base_q + d];
    }
    __syncthreads();

    // 2. Compute attention scores S[k] = scale * dot(Q, K[k]).
    for (int k = tid; k < S; k += blockDim.x) {
        const size_t base_k = ((static_cast<size_t>(b) * S + k) * H + h) * D;
        float dot = 0.0f;
        for (int d = 0; d < D; ++d) {
            dot += __bfloat162float(sQ[d]) * __bfloat162float(K[base_k + d]);
        }
        sAttn[k] = dot * scale;
    }
    __syncthreads();

    // 3. Block-wide max.
    float local_max = -INFINITY;
    for (int k = tid; k < S; k += blockDim.x) {
        local_max = fmaxf(local_max, sAttn[k]);
    }
    const float row_max = block_reduce<true>(local_max, sScratch, blockDim.x);

    // 4. exp(x - max) into smem; local sum.
    float local_sum = 0.0f;
    for (int k = tid; k < S; k += blockDim.x) {
        const float e = __expf(sAttn[k] - row_max);
        sAttn[k] = e;
        local_sum += e;
    }
    const float total = block_reduce<false>(local_sum, sScratch, blockDim.x);
    const float inv_total = 1.0f / (total + 1e-30f);

    // 5. O[d] = sum_k (sAttn[k] / total) * V[k, d]
    for (int d = tid; d < D; d += blockDim.x) {
        float acc = 0.0f;
        for (int k = 0; k < S; ++k) {
            const size_t base_k = ((static_cast<size_t>(b) * S + k) * H + h) * D;
            acc += sAttn[k] * __bfloat162float(V[base_k + d]);
        }
        const size_t out_off = ((static_cast<size_t>(b) * S + q) * H + h) * D + d;
        O[out_off] = __float2bfloat16(acc * inv_total);
    }
}

} // anonymous namespace

namespace f2k::cuda {

Attention::Attention(const Config& cfg) : cfg_(cfg) {
    if (cfg.batch <= 0 || cfg.seq <= 0 || cfg.n_heads <= 0 || cfg.head_dim <= 0) {
        err_ = "Attention: zero/negative dim";
        return;
    }
    if (cfg.head_dim > 1024) {
        err_ = "Attention v1: head_dim must be <= 1024";
        return;
    }
    if (cfg_.scale <= 0.0f) {
        cfg_.scale = 1.0f / std::sqrt(static_cast<float>(cfg.head_dim));
    }
    valid_ = true;
}

Attention::~Attention() = default;

bool        Attention::ok()         const { return valid_; }
const char* Attention::last_error() const { return err_; }

bool Attention::forward(const void* Q, const void* K, const void* V,
                        void* O, cudaStream_t stream) {
    if (!valid_) return false;
    const int B = cfg_.batch, S = cfg_.seq, H = cfg_.n_heads, D = cfg_.head_dim;
    const int block = (D > 128) ? D : 128;
    const size_t smem =
        static_cast<size_t>(D) * sizeof(__nv_bfloat16) +
        static_cast<size_t>(S) * sizeof(float) +
        32 * sizeof(float); // block-reduce scratch
    dim3 grid(S, H, B);
    attention_row_kernel<<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<      __nv_bfloat16*>(O),
        B, S, H, D, cfg_.scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
