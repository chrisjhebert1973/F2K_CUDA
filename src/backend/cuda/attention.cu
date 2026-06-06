// Self-attention: row-at-a-time fallback + a WMMA flash kernel (tensor-core
// QK^T and P@V) for head_dim 128.

#include "backend/cuda/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

#include <cmath>
#include <cstdint>
#include <cstdio>

namespace {

constexpr int MAX_BLOCK = 256;

// ---------------------------------------------------------------------------
// mma.sync m16n8k16 bf16 primitives (validated bit-exact in tools/mma_unit.cu).
// Fragment thread→element map (gid=lane/4, t=lane%4):
//   A[16x16] row-major: a0=A[gid][2t..1] a1=A[gid+8][2t..] a2=A[gid][2t+8..] a3=A[gid+8][2t+8..]
//   B[16x8]  col-major: b0=B[2t..1][gid] b1=B[2t+8..9][gid]   (k=row, n=col)
//   C[16x8]  fp32:      d0=C[gid][2t] d1=C[gid][2t+1] d2=C[gid+8][2t] d3=C[gid+8][2t+1]
// ---------------------------------------------------------------------------
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t(__bfloat16_as_ushort(hi)) << 16) | uint32_t(__bfloat16_as_ushort(lo));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// ldmatrix.x2 (2× 8x8 b16) → one mma B-fragment. Recipes validated in mma_unit.cu:
//   QK^T B (col-major K in smem): non-trans, addr at key row, hd-half col-block.
//   P@V  B (row-major V in smem): trans,     addr at key row (8 contiguous hd).
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void* p) {
    uint32_t a = static_cast<uint32_t>(__cvta_generic_to_shared(p));
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

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

// ---------------------------------------------------------------------------
// WMMA flash-attention kernel (FlashAttention-2 style) for head_dim D.
// 4 warps/CTA; each warp owns 16 query rows, so BM = 64 queries share each
// K/V tile load — 8× less K/V HBM traffic than the warp-per-query kernel above
// (the dominant cost: that kernel re-reads all K/V from HBM once per 8 queries).
// QK^T and P@V run on tensor cores (bf16 mma, fp32 accumulate); the online
// softmax keeps per-row (m, l) and rescales the fp32 O accumulator in smem.
// Instantiated for D=128 (the transformer); other dims use the kernels above.
// ---------------------------------------------------------------------------
using namespace nvcuda;

template <int D>
__global__ void flash_wmma_kernel(const __nv_bfloat16* __restrict__ Q,
                                  const __nv_bfloat16* __restrict__ K,
                                  const __nv_bfloat16* __restrict__ V,
                                  __nv_bfloat16* __restrict__ O,
                                  int B, int S, int H, float scale) {
    constexpr int BN = 48;          // keys per K/V tile (3 WMMA tiles; 48 fits
                                    // GB10's 99 KiB smem cap, BN=64 doesn't)
    constexpr int WARPS = 4;
    constexpr int BM = WARPS * 16;  // 64 queries per CTA (drives K/V reuse)

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q0 = blockIdx.x * BM + warp * 16;  // first query of this warp
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;

    extern __shared__ uint8_t smem[];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);  // [BN][D]
    __nv_bfloat16* Vs = Ks + BN * D;                             // [BN][D]
    __nv_bfloat16* Qs = Vs + BN * D;                             // [BM][D]
    __nv_bfloat16* Ps = Qs + BM * D;                             // [BM][BN]
    float* Ss = reinterpret_cast<float*>(Ps + BM * BN);          // [BM][BN]
    float* Os = Ss + BM * BN;                                    // [BM][D]
    float* Ms = Os + BM * D;                                     // [BM]
    float* Ls = Ms + BM;                                         // [BM]
    float* Cm = Ls + BM;                                         // [BM] per-row new max
    float* Cc = Cm + BM;                                         // [BM] per-row rescale

    __nv_bfloat16* qW = Qs + warp * 16 * D;
    __nv_bfloat16* pW = Ps + warp * 16 * BN;
    float* sW = Ss + warp * 16 * BN;
    float* oW = Os + warp * 16 * D;
    float* mW = Ms + warp * 16;
    float* lW = Ls + warp * 16;
    float* cmW = Cm + warp * 16;
    float* ccW = Cc + warp * 16;

    // Init O=0, m=-inf, l=0; load this warp's Q rows (zero-pad past S).
    for (int i = lane; i < 16 * D; i += 32) oW[i] = 0.0f;
    for (int i = lane; i < 16;     i += 32) { mW[i] = -INFINITY; lW[i] = 0.0f; }
    for (int i = lane; i < 16 * D; i += 32) {
        const int r = i / D, d = i % D, qq = q0 + r;
        qW[i] = (qq < S) ? Q[(((size_t)b * S + qq) * H + h) * D + d]
                         : __float2bfloat16(0.0f);
    }
    __syncthreads();

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qfrag[D / 16];
    #pragma unroll
    for (int kt = 0; kt < D / 16; ++kt) wmma::load_matrix_sync(qfrag[kt], qW + kt * 16, D);

    for (int k0 = 0; k0 < S; k0 += BN) {
        // Cooperative K/V tile load (all warps), zero-pad past S.
        for (int i = threadIdx.x; i < BN * D; i += blockDim.x) {
            const int n = i / D, d = i % D, kk = k0 + n;
            __nv_bfloat16 kv = __float2bfloat16(0.0f), vv = kv;
            if (kk < S) { const size_t off = (((size_t)b * S + kk) * H + h) * D + d; kv = K[off]; vv = V[off]; }
            Ks[i] = kv; Vs[i] = vv;
        }
        __syncthreads();

        // S = Q @ K^T  →  sW[16][BN] (fp32). K stored [BN][D] row-major is read
        // as col_major [D][BN] to get K^T.
        #pragma unroll
        for (int nt = 0; nt < BN / 16; ++nt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < D / 16; ++kt) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Ks + nt * 16 * D + kt * 16, D);
                wmma::mma_sync(acc, qfrag[kt], kf, acc);
            }
            wmma::store_matrix_sync(sW + nt * 16, acc, BN, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax. The per-row reductions (max, sum) stay on the 16
        // row-owning lanes, but the heavy elementwise work (exp over 16×BN, the
        // D-wide O rescale, and the PV add-back) is spread across all 32 lanes
        // via flat indexing — that epilogue, not the matmul, was the bottleneck.
        const int valid = min(BN, S - k0);
        // Phase 1 (16 lanes): scale scores, row max, store mnew + rescale corr.
        if (lane < 16) {
            const int r = lane;
            float* srow = sW + r * BN;
            float rmax = -INFINITY;
            for (int j = 0; j < BN; ++j) {
                const float v = (j < valid) ? srow[j] * scale : -INFINITY;
                srow[j] = v; rmax = fmaxf(rmax, v);
            }
            const float mold = mW[r], mnew = fmaxf(mold, rmax);
            cmW[r] = mnew; ccW[r] = __expf(mold - mnew); mW[r] = mnew;
        }
        __syncwarp();
        // Phase 2 (32 lanes): exp into pW (bf16) and sW (fp32, for the sum).
        for (int i = lane; i < 16 * BN; i += 32) {
            const int r = i / BN;
            const float p = __expf(sW[i] - cmW[r]);
            sW[i] = p; pW[i] = __float2bfloat16(p);
        }
        __syncwarp();
        // Phase 3 (16 lanes): row sum → update l (rescaled by corr).
        if (lane < 16) {
            const int r = lane; float s = 0.0f;
            for (int j = 0; j < BN; ++j) s += sW[r * BN + j];
            lW[r] = lW[r] * ccW[r] + s;
        }
        // Phase 4 (32 lanes): rescale the running O accumulator by corr.
        for (int i = lane; i < 16 * D; i += 32) oW[i] *= ccW[i / D];
        __syncthreads();

        // O += P @ V. Per [16,16] tile: WMMA into a fragment, store to scratch,
        // add into the fp32 O with all 32 lanes (256 elems → 8 iters/lane).
        #pragma unroll
        for (int nt = 0; nt < D / 16; ++nt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < BN / 16; ++kt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
                wmma::load_matrix_sync(pf, pW + kt * 16, BN);
                wmma::load_matrix_sync(vf, Vs + kt * 16 * D + nt * 16, D);
                wmma::mma_sync(acc, pf, vf, acc);
            }
            __syncwarp();
            wmma::store_matrix_sync(sW, acc, 16, wmma::mem_row_major);  // scratch [16][16]
            __syncwarp();
            for (int i = lane; i < 16 * 16; i += 32) {
                const int r = i >> 4, c = i & 15;
                oW[r * D + nt * 16 + c] += sW[i];
            }
            __syncwarp();
        }
        __syncthreads();
    }

    // O = O / l for valid query rows (all 32 lanes, flat over 16×D).
    for (int i = lane; i < 16 * D; i += 32) {
        const int r = i / D, d = i % D, qq = q0 + r;
        if (qq < S) {
            const float inv = 1.0f / (lW[r] + 1e-30f);
            O[(((size_t)b * S + qq) * H + h) * D + d] = __float2bfloat16(oW[i] * inv);
        }
    }
}

// ---------------------------------------------------------------------------
// WMMA flash-attention with a REGISTER-RESIDENT O accumulator (head_dim D).
// Same BM=64 / BN=48 tiling and K/V traffic as the older flash_wmma_kernel, but
// the O accumulator lives in wmma fragments (registers) and Q is loaded straight
// from global — dropping the 32 KiB Os + 16 KiB Qs smem arrays. smem ≈ 43 KiB
// → 2 CTAs/SM instead of 1, ~2× occupancy → 1.68× faster at S=4608 (the kernel
// was occupancy-bound, not bandwidth-bound — see bench_attention).
// The per-row online-softmax correction is applied to the register O via a smem
// store/scale/reload (wmma hides the fragment row→lane layout); the P@V add-back
// is a free register add since both accumulators share that layout.
// ---------------------------------------------------------------------------
template <int D>
__global__ void flash_wmma_regO_kernel(const __nv_bfloat16* __restrict__ Q,
                                       const __nv_bfloat16* __restrict__ K,
                                       const __nv_bfloat16* __restrict__ V,
                                       __nv_bfloat16* __restrict__ O,
                                       int B, int S, int H, float scale) {
    constexpr int BN = 48;
    constexpr int WARPS = 4;
    constexpr int BM = WARPS * 16;

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int q0 = blockIdx.x * BM + warp * 16;
    const int h  = blockIdx.y;
    const int b  = blockIdx.z;

    extern __shared__ uint8_t smem[];
    __nv_bfloat16* Ks = reinterpret_cast<__nv_bfloat16*>(smem);  // [BN][D]
    __nv_bfloat16* Vs = Ks + BN * D;                             // [BN][D]
    __nv_bfloat16* Ps = Vs + BN * D;                             // [BM][BN]
    float* Ss = reinterpret_cast<float*>(Ps + BM * BN);          // [BM][BN]
    float* Ms = Ss + BM * BN;                                    // [BM]
    float* Ls = Ms + BM;                                         // [BM]
    float* Cm = Ls + BM;                                         // [BM]
    float* Cc = Cm + BM;                                         // [BM]

    __nv_bfloat16* pW = Ps + warp * 16 * BN;  // [16][BN]; also Q-stage scratch
    float* sW  = Ss + warp * 16 * BN;   // [16][BN]; also O-rescale scratch [16][16]
    float* mW  = Ms + warp * 16;
    float* lW  = Ls + warp * 16;
    float* cmW = Cm + warp * 16;
    float* ccW = Cc + warp * 16;

    // O accumulator in registers (D/16 fragments per warp), running stats.
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> ofrag[D / 16];
    #pragma unroll
    for (int nt = 0; nt < D / 16; ++nt) wmma::fill_fragment(ofrag[nt], 0.0f);
    if (lane < 16) { mW[lane] = -INFINITY; lW[lane] = 0.0f; }

    // Q into fragments. Full query tiles load straight from global; a partial
    // last tile (q0+16 > S) is staged through pW with zero-padded rows.
    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> qfrag[D / 16];
    if (q0 + 16 <= S) {
        const __nv_bfloat16* Qrow = Q + (((size_t)b * S + q0) * H + h) * D;
        #pragma unroll
        for (int kt = 0; kt < D / 16; ++kt)
            wmma::load_matrix_sync(qfrag[kt], Qrow + kt * 16, H * D);
    } else {
        #pragma unroll
        for (int kt = 0; kt < D / 16; ++kt) {
            for (int i = lane; i < 16 * 16; i += 32) {
                const int r = i >> 4, c = i & 15, qq = q0 + r;
                pW[i] = (qq < S) ? Q[(((size_t)b * S + qq) * H + h) * D + kt * 16 + c]
                                 : __float2bfloat16(0.0f);
            }
            __syncwarp();
            wmma::load_matrix_sync(qfrag[kt], pW, 16);
            __syncwarp();
        }
    }
    __syncthreads();

    for (int k0 = 0; k0 < S; k0 += BN) {
        const int valid = min(BN, S - k0);   // real keys in this tile
        for (int i = threadIdx.x; i < BN * D; i += blockDim.x) {
            const int n = i / D, d = i % D, kk = k0 + n;
            __nv_bfloat16 kv = __float2bfloat16(0.0f), vv = kv;
            if (kk < S) { const size_t off = (((size_t)b * S + kk) * H + h) * D + d; kv = K[off]; vv = V[off]; }
            Ks[i] = kv; Vs[i] = vv;
        }
        __syncthreads();

        // S = Q @ K^T → sW (fp32).
        #pragma unroll
        for (int nt = 0; nt < BN / 16; ++nt) {
            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < D / 16; ++kt) {
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> kf;
                wmma::load_matrix_sync(kf, Ks + nt * 16 * D + kt * 16, D);
                wmma::mma_sync(acc, qfrag[kt], kf, acc);
            }
            wmma::store_matrix_sync(sW + nt * 16, acc, BN, wmma::mem_row_major);
        }
        __syncwarp();

        // Online softmax (keys >= valid are masked to -inf so they don't count).
        if (lane < 16) {
            const int r = lane; float* srow = sW + r * BN; float rmax = -INFINITY;
            for (int j = 0; j < BN; ++j) {
                const float v = (j < valid) ? srow[j] * scale : -INFINITY;
                srow[j] = v; rmax = fmaxf(rmax, v);
            }
            const float mold = mW[r], mnew = fmaxf(mold, rmax);
            cmW[r] = mnew; ccW[r] = __expf(mold - mnew); mW[r] = mnew;
        }
        __syncwarp();
        for (int i = lane; i < 16 * BN; i += 32) {
            const int r = i / BN; const float p = __expf(sW[i] - cmW[r]);
            sW[i] = p; pW[i] = __float2bfloat16(p);
        }
        __syncwarp();
        if (lane < 16) {
            const int r = lane; float s = 0.0f;
            for (int j = 0; j < BN; ++j) s += sW[r * BN + j];
            lW[r] = lW[r] * ccW[r] + s;
        }
        __syncwarp();

        // For each output column-tile: rescale O fragment by corr (store/scale/
        // reload through sW scratch), then add this tile's P@V (register add —
        // both accumulators share layout, so no row mapping needed).
        #pragma unroll
        for (int nt = 0; nt < D / 16; ++nt) {
            wmma::store_matrix_sync(sW, ofrag[nt], 16, wmma::mem_row_major);
            __syncwarp();
            for (int i = lane; i < 16 * 16; i += 32) sW[i] *= ccW[i >> 4];
            __syncwarp();
            wmma::load_matrix_sync(ofrag[nt], sW, 16, wmma::mem_row_major);
            __syncwarp();

            wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc;
            wmma::fill_fragment(acc, 0.0f);
            #pragma unroll
            for (int kt = 0; kt < BN / 16; ++kt) {
                wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> pf;
                wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::row_major> vf;
                wmma::load_matrix_sync(pf, pW + kt * 16, BN);
                wmma::load_matrix_sync(vf, Vs + kt * 16 * D + nt * 16, D);
                wmma::mma_sync(acc, pf, vf, acc);
            }
            #pragma unroll
            for (int i = 0; i < ofrag[nt].num_elements; ++i) ofrag[nt].x[i] += acc.x[i];
            __syncwarp();
        }
        __syncthreads();  // before next K/V tile overwrites Ks/Vs
    }

    // Normalize by 1/l and write out (store frag to sW to recover row layout).
    // Partial last query tile (q0+r >= S) must NOT write past the end of O.
    #pragma unroll
    for (int nt = 0; nt < D / 16; ++nt) {
        wmma::store_matrix_sync(sW, ofrag[nt], 16, wmma::mem_row_major);
        __syncwarp();
        for (int i = lane; i < 16 * 16; i += 32) {
            const int r = i >> 4, c = i & 15, qq = q0 + r;
            if (qq < S) {
                const float inv = 1.0f / (lW[r] + 1e-30f);
                O[(((size_t)b * S + qq) * H + h) * D + nt * 16 + c] =
                    __float2bfloat16(sW[i] * inv);
            }
        }
        __syncwarp();
    }
}

// ---------------------------------------------------------------------------
// mma.sync flash-attention (head_dim D) — removes regO's "rescale tax".
// Same BM=64/BN=48 tiling, but uses raw mma.sync m16n8k16 (documented fragment
// layout) instead of opaque wmma fragments. Consequences:
//   • O lives in C-fragment registers; the per-row online-softmax correction is
//     a direct register multiply (each thread owns rows {gid, gid+8} of O), so
//     NO smem store/scale/reload (the tax regO paid).
//   • The softmax row max/sum are register reductions across the 4 lanes that
//     share a row (__shfl_xor within the group of 4) — S never touches smem.
//   • P@V's contraction = QK^T's N dim and the C/A fragment layouts share the
//     (gid,gid+8) row + 2t column mapping, so the QK^T result registers pack
//     DIRECTLY into the P@V A-fragments (cast to bf16) — no P smem round-trip.
// The only inner-loop smem traffic is the unavoidable K/V global→smem staging.
// ---------------------------------------------------------------------------
template <int D>
__global__ void flash_mma_kernel(const __nv_bfloat16* __restrict__ Q,
                                 const __nv_bfloat16* __restrict__ K,
                                 const __nv_bfloat16* __restrict__ V,
                                 __nv_bfloat16* __restrict__ O,
                                 int B, int S, int H, float scale) {
    constexpr int BN = 48;
    constexpr int WARPS = 4;
    constexpr int BM = WARPS * 16;   // 64 queries / CTA
    constexpr int KT_QK = D / 16;    // QK^T contraction tiles (head dim)
    constexpr int NT_QK = BN / 8;    // QK^T n8-tiles (keys)
    constexpr int KT_PV = BN / 16;   // P@V contraction tiles (keys)
    constexpr int NT_PV = D / 8;     // P@V n8-tiles (head dim)

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int gid  = lane >> 2;      // 0..7
    const int t    = lane & 3;       // 0..3
    const int q0   = blockIdx.x * BM + warp * 16;
    const int h    = blockIdx.y;
    const int b    = blockIdx.z;
    const __nv_bfloat16 zero = __float2bfloat16(0.0f);

    extern __shared__ __nv_bfloat16 smem_mma[];
    __nv_bfloat16* Ks = smem_mma;          // [BN][D]
    __nv_bfloat16* Vs = Ks + BN * D;       // [BN][D]

    // This warp's Q rows {gid, gid+8} → A-fragments, held across the K loop.
    const int r_lo = q0 + gid, r_hi = q0 + gid + 8;
    uint32_t qf[KT_QK][4];
    #pragma unroll
    for (int kt = 0; kt < KT_QK; ++kt) {
        const int c = kt * 16 + 2 * t;
        const __nv_bfloat16* Qlo = Q + (((size_t)b * S + r_lo) * H + h) * D;
        const __nv_bfloat16* Qhi = Q + (((size_t)b * S + r_hi) * H + h) * D;
        const bool vlo = r_lo < S, vhi = r_hi < S;
        qf[kt][0] = vlo ? pack2(Qlo[c],   Qlo[c + 1]) : 0u;
        qf[kt][1] = vhi ? pack2(Qhi[c],   Qhi[c + 1]) : 0u;
        qf[kt][2] = vlo ? pack2(Qlo[c+8], Qlo[c + 9]) : 0u;
        qf[kt][3] = vhi ? pack2(Qhi[c+8], Qhi[c + 9]) : 0u;
    }

    float of[NT_PV][4];   // O accumulator (C-fragment layout)
    #pragma unroll
    for (int nt = 0; nt < NT_PV; ++nt) of[nt][0]=of[nt][1]=of[nt][2]=of[nt][3]=0.0f;
    float m_lo = -INFINITY, m_hi = -INFINITY, l_lo = 0.0f, l_hi = 0.0f;

    for (int k0 = 0; k0 < S; k0 += BN) {
        const int valid = min(BN, S - k0);
        for (int i = threadIdx.x; i < BN * D; i += blockDim.x) {
            const int n = i / D, d = i % D, kk = k0 + n;
            __nv_bfloat16 kv = zero, vv = zero;
            if (kk < S) { const size_t off=(((size_t)b*S+kk)*H+h)*D+d; kv=K[off]; vv=V[off]; }
            Ks[i] = kv; Vs[i] = vv;
        }
        __syncthreads();

        // S = Q @ K^T  → sc[NT_QK][4] (fp32), in C-fragment layout.
        float sc[NT_QK][4];
        const int kkey = lane & 7;             // key within the n8-tile (ldmatrix row)
        const int khalf = ((lane >> 3) & 1) * 8;
        #pragma unroll
        for (int nt = 0; nt < NT_QK; ++nt) {
            float acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
            #pragma unroll
            for (int kt = 0; kt < KT_QK; ++kt) {
                // B = K^T (col-major in Ks): ldmatrix non-trans, hd-halves as the
                // two 8x8 matrices. addr = Ks[key row][hd col].
                uint32_t bf[2];
                ldmatrix_x2(bf, &Ks[(nt * 8 + kkey) * D + kt * 16 + khalf]);
                mma_m16n8k16(acc, qf[kt], bf);
            }
            // Output key column for this n8-tile is (nt*8 + 2t) / (+1); mask padding.
            const int c0 = nt * 8 + 2 * t, c1 = c0 + 1;
            sc[nt][0] = (c0 < valid) ? acc[0] * scale : -INFINITY;
            sc[nt][1] = (c1 < valid) ? acc[1] * scale : -INFINITY;
            sc[nt][2] = (c0 < valid) ? acc[2] * scale : -INFINITY;
            sc[nt][3] = (c1 < valid) ? acc[3] * scale : -INFINITY;
        }

        // Row max over the thread's keys, then across the 4 lanes sharing a row.
        float rmax_lo = -INFINITY, rmax_hi = -INFINITY;
        #pragma unroll
        for (int nt = 0; nt < NT_QK; ++nt) {
            rmax_lo = fmaxf(rmax_lo, fmaxf(sc[nt][0], sc[nt][1]));
            rmax_hi = fmaxf(rmax_hi, fmaxf(sc[nt][2], sc[nt][3]));
        }
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xFFFFFFFFu, rmax_lo, 1));
        rmax_lo = fmaxf(rmax_lo, __shfl_xor_sync(0xFFFFFFFFu, rmax_lo, 2));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xFFFFFFFFu, rmax_hi, 1));
        rmax_hi = fmaxf(rmax_hi, __shfl_xor_sync(0xFFFFFFFFu, rmax_hi, 2));

        const float mnew_lo = fmaxf(m_lo, rmax_lo), mnew_hi = fmaxf(m_hi, rmax_hi);
        const float corr_lo = __expf(m_lo - mnew_lo), corr_hi = __expf(m_hi - mnew_hi);
        m_lo = mnew_lo; m_hi = mnew_hi;

        // P = exp(S - m) (masked entries → exp(-inf)=0); keep bf16 in C-layout
        // for direct repacking into P@V A-fragments. Accumulate row sums.
        __nv_bfloat16 pcb[NT_QK][4];
        float psum_lo = 0.0f, psum_hi = 0.0f;
        #pragma unroll
        for (int nt = 0; nt < NT_QK; ++nt) {
            const float p0 = __expf(sc[nt][0] - mnew_lo), p1 = __expf(sc[nt][1] - mnew_lo);
            const float p2 = __expf(sc[nt][2] - mnew_hi), p3 = __expf(sc[nt][3] - mnew_hi);
            psum_lo += p0 + p1; psum_hi += p2 + p3;
            pcb[nt][0]=__float2bfloat16(p0); pcb[nt][1]=__float2bfloat16(p1);
            pcb[nt][2]=__float2bfloat16(p2); pcb[nt][3]=__float2bfloat16(p3);
        }
        psum_lo += __shfl_xor_sync(0xFFFFFFFFu, psum_lo, 1);
        psum_lo += __shfl_xor_sync(0xFFFFFFFFu, psum_lo, 2);
        psum_hi += __shfl_xor_sync(0xFFFFFFFFu, psum_hi, 1);
        psum_hi += __shfl_xor_sync(0xFFFFFFFFu, psum_hi, 2);
        l_lo = l_lo * corr_lo + psum_lo;
        l_hi = l_hi * corr_hi + psum_hi;

        // Rescale running O by corr — direct register multiply (d0,d1=row lo; d2,d3=row hi).
        #pragma unroll
        for (int nt = 0; nt < NT_PV; ++nt) {
            of[nt][0]*=corr_lo; of[nt][1]*=corr_lo; of[nt][2]*=corr_hi; of[nt][3]*=corr_hi;
        }

        // O += P @ V. A-frags come straight from the QK^T result registers (pcb);
        // B-frags read V from smem (key=row, head-dim=col).
        const int vkey = lane & 15;   // key within the 16-key tile (ldmatrix row)
        #pragma unroll
        for (int nt = 0; nt < NT_PV; ++nt) {        // output head-dim 8-tiles
            #pragma unroll
            for (int kt = 0; kt < KT_PV; ++kt) {    // key 16-tiles
                const uint32_t af[4] = {
                    pack2(pcb[2*kt][0],   pcb[2*kt][1]),
                    pack2(pcb[2*kt][2],   pcb[2*kt][3]),
                    pack2(pcb[2*kt+1][0], pcb[2*kt+1][1]),
                    pack2(pcb[2*kt+1][2], pcb[2*kt+1][3]) };
                // B = V (row-major in Vs): ldmatrix trans, addr at key row, hd col.
                uint32_t bf[2];
                ldmatrix_x2_trans(bf, &Vs[(kt * 16 + vkey) * D + nt * 8]);
                mma_m16n8k16(of[nt], af, bf);
            }
        }
        __syncthreads();   // before next tile overwrites Ks/Vs
    }

    // Normalize by 1/l and write out (O_C[nt].d0 = O[row lo][nt*8+2t], etc.).
    const float inv_lo = 1.0f / (l_lo + 1e-30f), inv_hi = 1.0f / (l_hi + 1e-30f);
    #pragma unroll
    for (int nt = 0; nt < NT_PV; ++nt) {
        const int col0 = nt * 8 + 2 * t, col1 = col0 + 1;
        if (r_lo < S) {
            O[(((size_t)b*S+r_lo)*H+h)*D + col0] = __float2bfloat16(of[nt][0]*inv_lo);
            O[(((size_t)b*S+r_lo)*H+h)*D + col1] = __float2bfloat16(of[nt][1]*inv_lo);
        }
        if (r_hi < S) {
            O[(((size_t)b*S+r_hi)*H+h)*D + col0] = __float2bfloat16(of[nt][2]*inv_hi);
            O[(((size_t)b*S+r_hi)*H+h)*D + col1] = __float2bfloat16(of[nt][3]*inv_hi);
        }
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

template <int D>
bool launch_wmma(const void* Q, const void* K, const void* V, void* O,
                 int B, int S, int H, float scale, const char*& err,
                 cudaStream_t stream) {
    constexpr int BN = 48, WARPS = 4, BM = WARPS * 16;
    const int block = WARPS * 32;
    // smem: Ks+Vs+Qs+Ps (bf16) | Ss+Os+Ms+Ls+Cm+Cc (fp32)
    const size_t smem =
        (static_cast<size_t>(BN) * D + BN * D + BM * D + BM * BN) * sizeof(__nv_bfloat16) +
        (static_cast<size_t>(BM) * BN + BM * D + 4 * BM) * sizeof(float);
    if (!ensure_dynamic_smem(flash_wmma_kernel<D>, smem, err)) return false;
    dim3 grid((S + BM - 1) / BM, H, B);
    flash_wmma_kernel<D><<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<      __nv_bfloat16*>(O),
        B, S, H, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

// Production D=128 launcher: register-resident O accumulator (43 KiB smem →
// 2 CTA/SM) — 1.68× the old flash_wmma_kernel, bit-exact. Handles arbitrary S
// (partial query/key tiles are zero-padded and masked). See flash_wmma_regO_kernel.
template <int D>
bool launch_wmma_regO(const void* Q, const void* K, const void* V, void* O,
                      int B, int S, int H, float scale, const char*& err,
                      cudaStream_t stream) {
    constexpr int BN = 48, WARPS = 4, BM = WARPS * 16;
    const int block = WARPS * 32;
    // smem: Ks+Vs+Ps (bf16) | Ss+Ms+Ls+Cm+Cc (fp32). No Os/Qs (O is in registers,
    // Q loads straight from global) — that's the occupancy win vs launch_wmma.
    const size_t smem =
        (static_cast<size_t>(BN) * D + BN * D + BM * BN) * sizeof(__nv_bfloat16) +
        (static_cast<size_t>(BM) * BN + 4 * BM) * sizeof(float);
    if (!ensure_dynamic_smem(flash_wmma_regO_kernel<D>, smem, err)) return false;
    dim3 grid((S + BM - 1) / BM, H, B);
    flash_wmma_regO_kernel<D><<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<      __nv_bfloat16*>(O),
        B, S, H, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

// mma.sync flash kernel launcher (D=128). smem = K/V tile only (no O/Q/P/S
// staging) — 24 KiB at BN=48, register-limited rather than smem-limited.
template <int D>
bool launch_mma(const void* Q, const void* K, const void* V, void* O,
                int B, int S, int H, float scale, const char*& err,
                cudaStream_t stream) {
    constexpr int BN = 48, WARPS = 4, BM = WARPS * 16;
    const int block = WARPS * 32;
    const size_t smem = static_cast<size_t>(2) * BN * D * sizeof(__nv_bfloat16);
    if (!ensure_dynamic_smem(flash_mma_kernel<D>, smem, err)) return false;
    dim3 grid((S + BM - 1) / BM, H, B);
    flash_mma_kernel<D><<<grid, block, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(Q),
        static_cast<const __nv_bfloat16*>(K),
        static_cast<const __nv_bfloat16*>(V),
        static_cast<      __nv_bfloat16*>(O),
        B, S, H, scale);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // anonymous namespace

bool Attention::forward(const void* Q, const void* K, const void* V,
                        void* O, cudaStream_t stream) {
    if (!valid_) return false;
    const int B = cfg_.batch, S = cfg_.seq, H = cfg_.n_heads, D = cfg_.head_dim;

    // D=128 (the transformer) uses the mma.sync flash kernel: raw mma.sync
    // m16n8k16 + ldmatrix, register-resident O (the online-softmax rescale is a
    // register multiply — no smem round-trip), and QK^T result registers repacked
    // directly into the P@V A-fragments. Bandwidth-bound at ~279 GB/s (≈ GB10
    // peak) — 1.74× the earlier register-O wmma kernel (launch_wmma_regO, kept for
    // bench A/B). Other 32-divisible dims (incl. the VAE's D=512, whose smem is
    // too large for this tiling) use the warp-per-query flash kernel; everything
    // else uses the row kernel.
    if (D == 128) {
        return launch_mma<128>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream);
    }
    if (D % 32 == 0) {
        switch (D / 32) {
            case 2:  return launch_flash<2>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream);  // D=64
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

// Benchmark entry point for the register-O kernel (now the D=128 production
// path via forward()). Optionally prints the achieved CTAs/SM, then launches.
bool launch_wmma_regO_probe(const void* Q, const void* K, const void* V, void* O,
                            int B, int S, int H, float scale, bool report,
                            cudaStream_t stream) {
    constexpr int D = 128, BN = 48, WARPS = 4, BM = WARPS * 16;
    const int block = WARPS * 32;
    if (report) {
        const size_t smem =
            (static_cast<size_t>(BN) * D + BN * D + BM * BN) * sizeof(__nv_bfloat16) +
            (static_cast<size_t>(BM) * BN + 4 * BM) * sizeof(float);
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, flash_wmma_regO_kernel<D>, block, smem);
        std::printf("  [regO probe] smem=%.1f KiB  occupancy=%d CTA/SM (%d warps/SM)\n",
                    smem / 1024.0, blocks, blocks * WARPS);
    }
    const char* err = nullptr;
    return launch_wmma_regO<D>(Q, K, V, O, B, S, H, scale, err, stream);
}

// Benchmark entry point for the mma.sync kernel (the rescale-tax-free path).
bool launch_mma_probe(const void* Q, const void* K, const void* V, void* O,
                      int B, int S, int H, float scale, bool report,
                      cudaStream_t stream) {
    constexpr int D = 128, BN = 48, WARPS = 4, BM = WARPS * 16;
    const int block = WARPS * 32;
    if (report) {
        const size_t smem = static_cast<size_t>(2) * BN * D * sizeof(__nv_bfloat16);
        int blocks = 0;
        cudaOccupancyMaxActiveBlocksPerMultiprocessor(
            &blocks, flash_mma_kernel<D>, block, smem);
        std::printf("  [mma probe]  smem=%.1f KiB  occupancy=%d CTA/SM (%d warps/SM)\n",
                    smem / 1024.0, blocks, blocks * WARPS);
    }
    const char* err = nullptr;
    return launch_mma<D>(Q, K, V, O, B, S, H, scale, err, stream);
}

} // namespace f2k::cuda
