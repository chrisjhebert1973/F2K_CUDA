// 1D RoPE in-place kernel.

#include "backend/cuda/kernels/rope.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK_SIZE = 128;

__global__ void rope_inplace_kernel(__nv_bfloat16* __restrict__ x,
                                    const float* __restrict__ cos_table,
                                    const float* __restrict__ sin_table,
                                    const int32_t* __restrict__ pos_ids,
                                    int head_dim,
                                    int seq_len) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int half_dim = head_dim >> 1;

    const int pos = pos_ids[row];
    // Clamp out-of-range positions to seq_len-1 (no-op rotation would also be ok).
    const int pos_safe = (pos >= seq_len) ? (seq_len - 1) : (pos < 0 ? 0 : pos);
    const float* cos_row = cos_table + static_cast<size_t>(pos_safe) * half_dim;
    const float* sin_row = sin_table + static_cast<size_t>(pos_safe) * half_dim;

    __nv_bfloat16* row_ptr = x + static_cast<size_t>(row) * head_dim;

    for (int i = tid; i < half_dim; i += BLOCK_SIZE) {
        const float x0 = __bfloat162float(row_ptr[i]);            // dim 2i
        const float x1 = __bfloat162float(row_ptr[i + half_dim]); // dim 2i+1 (split layout)
        const float c  = cos_row[i];
        const float s  = sin_row[i];
        const float y0 = x0 * c - x1 * s;
        const float y1 = x0 * s + x1 * c;
        row_ptr[i]            = __float2bfloat16(y0);
        row_ptr[i + half_dim] = __float2bfloat16(y1);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool rope_inplace_bf16(void* x, const float* cos_table, const float* sin_table,
                       const int32_t* pos_ids,
                       int batch_rows, int head_dim, int seq_len,
                       cudaStream_t stream) {
    if (batch_rows <= 0 || head_dim <= 0 || (head_dim & 1)) return false;
    if (head_dim > 512) return false;
    rope_inplace_kernel<<<batch_rows, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x), cos_table, sin_table, pos_ids,
        head_dim, seq_len);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
