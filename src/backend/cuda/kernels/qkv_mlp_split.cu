#include "backend/cuda/kernels/qkv_mlp_split.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int BLOCK = 256;

__global__ void split_qkv_mlp_kernel(const __nv_bfloat16* __restrict__ fused,
                                     __nv_bfloat16* __restrict__ q,
                                     __nv_bfloat16* __restrict__ k,
                                     __nv_bfloat16* __restrict__ v,
                                     __nv_bfloat16* __restrict__ gate,
                                     __nv_bfloat16* __restrict__ up,
                                     int rows, int hidden, int ffn) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int total = 3 * hidden + 2 * ffn;

    const size_t fused_base = static_cast<size_t>(row) * total;
    const size_t hbase      = static_cast<size_t>(row) * hidden;
    const size_t fbase      = static_cast<size_t>(row) * ffn;

    // Walk the row in chunks of BLOCK elements, dispatch to the right dest.
    for (int c = tid; c < total; c += BLOCK) {
        const __nv_bfloat16 val = fused[fused_base + c];
        if (c < hidden) {
            q[hbase + c] = val;
        } else if (c < 2 * hidden) {
            k[hbase + (c - hidden)] = val;
        } else if (c < 3 * hidden) {
            v[hbase + (c - 2 * hidden)] = val;
        } else if (c < 3 * hidden + ffn) {
            gate[fbase + (c - 3 * hidden)] = val;
        } else {
            up[fbase + (c - 3 * hidden - ffn)] = val;
        }
    }
}

__global__ void concat_attn_mlp_kernel(const __nv_bfloat16* __restrict__ attn,
                                       const __nv_bfloat16* __restrict__ mlp,
                                       __nv_bfloat16* __restrict__ out,
                                       int rows, int hidden, int ffn) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int total = hidden + ffn;
    const size_t obase = static_cast<size_t>(row) * total;
    const size_t hbase = static_cast<size_t>(row) * hidden;
    const size_t fbase = static_cast<size_t>(row) * ffn;
    for (int c = tid; c < total; c += BLOCK) {
        out[obase + c] = (c < hidden) ? attn[hbase + c] : mlp[fbase + (c - hidden)];
    }
}

__global__ void split_half_kernel(const __nv_bfloat16* __restrict__ fused,
                                  __nv_bfloat16* __restrict__ gate,
                                  __nv_bfloat16* __restrict__ up,
                                  int rows, int half) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int total = 2 * half;
    const size_t fbase = static_cast<size_t>(row) * total;
    const size_t hbase = static_cast<size_t>(row) * half;
    for (int c = tid; c < total; c += BLOCK) {
        const __nv_bfloat16 v = fused[fbase + c];
        if (c < half) gate[hbase + c]          = v;
        else          up  [hbase + (c - half)] = v;
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool split_qkv_mlp_bf16(const void* fused, void* q, void* k, void* v,
                        void* gate, void* up,
                        int rows, int hidden, int ffn, cudaStream_t stream) {
    if (rows <= 0 || hidden <= 0 || ffn <= 0) return false;
    split_qkv_mlp_kernel<<<rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<__nv_bfloat16*>(q),
        static_cast<__nv_bfloat16*>(k),
        static_cast<__nv_bfloat16*>(v),
        static_cast<__nv_bfloat16*>(gate),
        static_cast<__nv_bfloat16*>(up),
        rows, hidden, ffn);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool concat_attn_mlp_bf16(const void* attn_out, const void* mlp_act, void* concat_out,
                          int rows, int hidden, int ffn, cudaStream_t stream) {
    if (rows <= 0 || hidden <= 0 || ffn <= 0) return false;
    concat_attn_mlp_kernel<<<rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(attn_out),
        static_cast<const __nv_bfloat16*>(mlp_act),
        static_cast<__nv_bfloat16*>(concat_out),
        rows, hidden, ffn);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool split_half_bf16(const void* fused, void* gate, void* up,
                     int rows, int half, cudaStream_t stream) {
    if (rows <= 0 || half <= 0) return false;
    split_half_kernel<<<rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(fused),
        static_cast<__nv_bfloat16*>(gate),
        static_cast<__nv_bfloat16*>(up),
        rows, half);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
