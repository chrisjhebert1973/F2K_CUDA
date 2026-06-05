// Fused SiLU(gate) * up element-wise kernel, used by SwiGLU MLP blocks.
//
//   out[i] = silu(gate[i]) * up[i]
//   silu(x) = x / (1 + exp(-x))
//
// All tensors are BF16; compute is FP32 internally. Length = total element
// count (no shape — just a flat buffer).

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

bool silu_mul_bf16(const void* gate_bf16,
                   const void* up_bf16,
                   void*       out_bf16,
                   size_t      n_elements,
                   cudaStream_t stream = nullptr);

// In-place: x[i] = x[i] / (1 + exp(-x[i])). Pure activation, no multiply.
bool silu_inplace_bf16(void* x_bf16, size_t n_elements,
                       cudaStream_t stream = nullptr);

} // namespace f2k::cuda
