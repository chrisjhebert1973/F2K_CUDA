// Self-attention: row-at-a-time fallback + a WMMA flash kernel (tensor-core
// QK^T and P@V) for head_dim 128.

#include "backend/cuda/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <mma.h>

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

} // anonymous namespace

bool Attention::forward(const void* Q, const void* K, const void* V,
                        void* O, cudaStream_t stream) {
    if (!valid_) return false;
    const int B = cfg_.batch, S = cfg_.seq, H = cfg_.n_heads, D = cfg_.head_dim;

    // D=128 (the transformer) uses the WMMA flash kernel: tensor-core QK^T/P@V
    // and BM=64 queries/CTA → 8× less K/V HBM traffic. Other 32-divisible dims
    // (incl. the VAE's D=512, whose smem is too large for the WMMA tiling) use
    // the warp-per-query flash kernel; everything else uses the row kernel.
    if (D == 128) {
        return launch_wmma<128>(Q, K, V, O, B, S, H, cfg_.scale, err_, stream);
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

} // namespace f2k::cuda
