#include "backend/cuda/kernels/seq_ops.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

constexpr int BLOCK = 256;

__global__ void concat_two_streams_kernel(const __nv_bfloat16* __restrict__ A,
                                          const __nv_bfloat16* __restrict__ B,
                                          __nv_bfloat16* __restrict__ out,
                                          int batch, int S_a, int S_b, int D) {
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    if (s >= S_a + S_b || b >= batch) return;
    const int total_S = S_a + S_b;
    const size_t row_out = (static_cast<size_t>(b) * total_S + s) * D;
    if (s < S_a) {
        const size_t row_in = (static_cast<size_t>(b) * S_a + s) * D;
        for (int d = threadIdx.x; d < D; d += BLOCK) out[row_out + d] = A[row_in + d];
    } else {
        const size_t row_in = (static_cast<size_t>(b) * S_b + (s - S_a)) * D;
        for (int d = threadIdx.x; d < D; d += BLOCK) out[row_out + d] = B[row_in + d];
    }
}

__global__ void take_tail_kernel(const __nv_bfloat16* __restrict__ src,
                                 __nv_bfloat16* __restrict__ dst,
                                 int batch, int S_a, int S_b, int D) {
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    if (s >= S_b || b >= batch) return;
    const int total_S = S_a + S_b;
    const size_t row_src = (static_cast<size_t>(b) * total_S + (S_a + s)) * D;
    const size_t row_dst = (static_cast<size_t>(b) * S_b + s) * D;
    for (int d = threadIdx.x; d < D; d += BLOCK) dst[row_dst + d] = src[row_src + d];
}

} // anonymous namespace

namespace f2k::cuda {

bool concat_two_streams_bf16(const void* A, const void* B, void* out,
                             int batch, int S_a, int S_b, int D, cudaStream_t stream) {
    if (batch <= 0 || S_a < 0 || S_b < 0 || D <= 0) return false;
    dim3 grid(S_a + S_b, batch);
    concat_two_streams_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(A),
        static_cast<const __nv_bfloat16*>(B),
        static_cast<__nv_bfloat16*>(out),
        batch, S_a, S_b, D);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool take_tail_bf16(const void* src, void* dst,
                    int batch, int S_a, int S_b, int D, cudaStream_t stream) {
    if (batch <= 0 || S_a < 0 || S_b <= 0 || D <= 0) return false;
    dim3 grid(S_b, batch);
    take_tail_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src),
        static_cast<__nv_bfloat16*>(dst),
        batch, S_a, S_b, D);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
