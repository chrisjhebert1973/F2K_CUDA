#include "backend/cuda/kernels/embed_lookup.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>

namespace {

__global__ void embed_lookup_kernel(const __nv_bfloat16* __restrict__ table,
                                    const int32_t* __restrict__ ids,
                                    __nv_bfloat16* __restrict__ out,
                                    int seq, int hidden, int vocab) {
    const int s = blockIdx.y;
    if (s >= seq) return;
    const int id = ids[s];
    if (id < 0 || id >= vocab) return; // ignore out-of-range; should not happen for clean inputs
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < hidden; h += blockDim.x * gridDim.x) {
        out[(size_t)s * hidden + h] = table[(size_t)id * hidden + h];
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool embed_lookup_bf16(const void* table, const int32_t* ids, void* out,
                       int seq, int hidden, int vocab, cudaStream_t stream) {
    if (seq <= 0 || hidden <= 0 || vocab <= 0) return false;
    const int BLOCK = 256;
    dim3 grid((hidden + BLOCK - 1) / BLOCK, seq);
    embed_lookup_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(table), ids,
        static_cast<__nv_bfloat16*>(out),
        seq, hidden, vocab);
    return cudaPeekAtLastError() == cudaSuccess;
}

} // namespace f2k::cuda
