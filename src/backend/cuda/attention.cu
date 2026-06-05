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

// ---------------------------------------------------------------------------
// Flash-attention kernel: one warp per query, 8 warps per CTA, K/V tiled into
// shared memory and reused across the CTA's 8 queries (≈8× less K/V HBM traffic
// than the row kernel, which re-reads all of K/V from HBM for every query).
// Online softmax keeps a running (max, sum, output) — smem is bounded by the
// tile, not S, so long sequences (VAE at S=16384) no longer hit the smem ceiling.
//
// Templated on DPT = D/32 (dims each lane owns): lane owns dims {lane, lane+32,
// ...}. q/acc live in registers (2·DPT floats). Scores are a warp dot-reduce.
// ---------------------------------------------------------------------------
template <int DPT>
__global__ void flash_attn_kernel(const __nv_bfloat16* __restrict__ Q,
                                  const __nv_bfloat16* __restrict__ K,
                                  const __nv_bfloat16* __restrict__ V,
                                  __nv_bfloat16* __restrict__ O,
                                  int B, int S, int H, int TILE_K, float scale) {
    constexpr int D = DPT * 32;
    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps_per_block = blockDim.x >> 5;
    const int q = blockIdx.x * warps_per_block + warp;
    const int h = blockIdx.y;
    const int b = blockIdx.z;

    extern __shared__ __nv_bfloat16 s_kv[];
    __nv_bfloat16* sK = s_kv;                         // [TILE_K * D]
    __nv_bfloat16* sV = s_kv + TILE_K * D;            // [TILE_K * D]

    float q_reg[DPT], acc[DPT];
    #pragma unroll
    for (int i = 0; i < DPT; ++i) { q_reg[i] = 0.0f; acc[i] = 0.0f; }

    const bool active = (q < S);
    if (active) {
        const size_t base_q = ((static_cast<size_t>(b) * S + q) * H + h) * D;
        #pragma unroll
        for (int i = 0; i < DPT; ++i)
            q_reg[i] = __bfloat162float(Q[base_q + lane + 32 * i]);
    }
    float m = -INFINITY, l = 0.0f;

    const int n_tiles = (S + TILE_K - 1) / TILE_K;
    for (int j = 0; j < n_tiles; ++j) {
        const int k0 = j * TILE_K;
        // Cooperative load of K/V tile into smem (all threads, incl. inactive warps).
        const int tile_elems = TILE_K * D;
        for (int e = threadIdx.x; e < tile_elems; e += blockDim.x) {
            const int kk = e / D;
            const int dd = e - kk * D;
            const int kg = k0 + kk;
            if (kg < S) {
                const size_t base = ((static_cast<size_t>(b) * S + kg) * H + h) * D + dd;
                sK[e] = K[base];
                sV[e] = V[base];
            } else {
                sK[e] = __float2bfloat16(0.0f);
                sV[e] = __float2bfloat16(0.0f);
            }
        }
        __syncthreads();

        if (active) {
            const int kcount = (TILE_K < S - k0) ? TILE_K : (S - k0);
            for (int kk = 0; kk < kcount; ++kk) {
                float partial = 0.0f;
                #pragma unroll
                for (int i = 0; i < DPT; ++i)
                    partial += q_reg[i] * __bfloat162float(sK[kk * D + lane + 32 * i]);
                // Warp dot-reduce → full score (identical on all lanes).
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 16);
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 8);
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 4);
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 2);
                partial += __shfl_xor_sync(0xFFFFFFFFu, partial, 1);
                const float s = partial * scale;
                // Online softmax update.
                const float m_new = fmaxf(m, s);
                const float corr  = __expf(m - m_new);
                const float p     = __expf(s - m_new);
                l = l * corr + p;
                #pragma unroll
                for (int i = 0; i < DPT; ++i)
                    acc[i] = acc[i] * corr + p * __bfloat162float(sV[kk * D + lane + 32 * i]);
                m = m_new;
            }
        }
        __syncthreads();
    }

    if (active) {
        const float inv = 1.0f / (l + 1e-30f);
        const size_t base_o = ((static_cast<size_t>(b) * S + q) * H + h) * D;
        #pragma unroll
        for (int i = 0; i < DPT; ++i)
            O[base_o + lane + 32 * i] = __float2bfloat16(acc[i] * inv);
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

namespace {

// Raise the dynamic-smem cap above the 48 KiB default when a kernel needs it.
// Blackwell allows up to ~228 KiB/block. Returns false on failure.
template <class KernelPtr>
bool ensure_dynamic_smem(KernelPtr kernel, size_t smem, const char*& err) {
    if (smem <= 48u * 1024u) return true;
    const cudaError_t e = cudaFuncSetAttribute(
        kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, static_cast<int>(smem));
    if (e != cudaSuccess) {
        err = "attention: smem opt-in failed (sequence too long for this GPU)";
        return false;
    }
    return true;
}

template <int DPT>
bool launch_flash(const void* Q, const void* K, const void* V, void* O,
                  int B, int S, int H, float scale, const char*& err,
                  cudaStream_t stream) {
    constexpr int D = DPT * 32;
    constexpr int WARPS_PER_BLOCK = 8;
    const int block   = WARPS_PER_BLOCK * 32;
    const int TILE_K  = (D <= 128) ? 64 : 32;   // smem ≈ 32–64 KiB
    const size_t smem = static_cast<size_t>(2) * TILE_K * D * sizeof(__nv_bfloat16);
    if (!ensure_dynamic_smem(flash_attn_kernel<DPT>, smem, err)) return false;
    dim3 grid((S + WARPS_PER_BLOCK - 1) / WARPS_PER_BLOCK, H, B);
    flash_attn_kernel<DPT><<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<      __nv_bfloat16*>(O),
        B, S, H, TILE_K, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // anonymous namespace

bool Attention::forward(const void* Q, const void* K, const void* V,
                        void* O, cudaStream_t stream) {
    if (!valid_) return false;
    const int B = cfg_.batch, S = cfg_.seq, H = cfg_.n_heads, D = cfg_.head_dim;

    // Flash path for head dims that divide into 32-lane chunks (covers the
    // transformer's D=128 and the VAE's D=512). Falls back to the row kernel
    // for other dims.
    if (D % 32 == 0) {
        switch (D / 32) {
            case 2:  return launch_flash<2>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream);  // D=64
            case 4:  return launch_flash<4>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream);  // D=128
            case 8:  return launch_flash<8>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream);  // D=256
            case 16: return launch_flash<16>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream); // D=512
            default: break;  // fall through to row kernel
        }
    }

    // Fallback: row-at-a-time kernel (arbitrary D, scores in smem).
    const int block = (D > 128) ? D : 128;
    const size_t smem =
        static_cast<size_t>(D) * sizeof(__nv_bfloat16) +
        static_cast<size_t>(S) * sizeof(float) +
        32 * sizeof(float);
    if (!ensure_dynamic_smem(attention_row_kernel, smem, err_)) return false;
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
