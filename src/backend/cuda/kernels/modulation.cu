// Modulation + gated residual kernels for diffusion-transformer blocks.

#include "backend/cuda/kernels/modulation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

namespace {

constexpr int BLOCK = 256;

__global__ void modulate_kernel(const __nv_bfloat16* __restrict__ x,
                                __nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ scale,
                                const __nv_bfloat16* __restrict__ shift,
                                int batch_rows, int hidden_dim) {
    const int row = blockIdx.x;
    if (row >= batch_rows) return;
    const size_t base = static_cast<size_t>(row) * hidden_dim;
    for (int d = threadIdx.x; d < hidden_dim; d += BLOCK) {
        const float xv = __bfloat162float(x[base + d]);
        const float sv = __bfloat162float(scale[d]);
        const float bv = __bfloat162float(shift[d]);
        y[base + d] = __float2bfloat16(xv * (1.0f + sv) + bv);
    }
}

__global__ void gated_residual_kernel(__nv_bfloat16* __restrict__ y,
                                      const __nv_bfloat16* __restrict__ delta,
                                      const __nv_bfloat16* __restrict__ gate,
                                      int batch_rows, int hidden_dim) {
    const int row = blockIdx.x;
    if (row >= batch_rows) return;
    const size_t base = static_cast<size_t>(row) * hidden_dim;
    for (int d = threadIdx.x; d < hidden_dim; d += BLOCK) {
        const float yv = __bfloat162float(y[base + d]);
        const float dv = __bfloat162float(delta[base + d]);
        const float gv = __bfloat162float(gate[d]);
        y[base + d] = __float2bfloat16(yv + gv * dv);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool modulate_bf16(const void* x, void* y,
                   const void* scale, const void* shift,
                   int batch_rows, int hidden_dim, cudaStream_t stream) {
    if (batch_rows <= 0 || hidden_dim <= 0) return false;
    modulate_kernel<<<batch_rows, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<      __nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(scale),
        static_cast<const __nv_bfloat16*>(shift),
        batch_rows, hidden_dim);
    return cudaPeekAtLastError() == cudaSuccess;
}

bool gated_residual_bf16(void* y, const void* delta, const void* gate,
                         int batch_rows, int hidden_dim, cudaStream_t stream) {
    if (batch_rows <= 0 || hidden_dim <= 0) return false;
    gated_residual_kernel<<<batch_rows, BLOCK, 0, stream>>>(
        static_cast<      __nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(delta),
        static_cast<const __nv_bfloat16*>(gate),
        batch_rows, hidden_dim);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
