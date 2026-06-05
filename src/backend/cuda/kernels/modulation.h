// Diffusion-transformer modulation primitives.
//
// modulate:
//   y[r, d] = x[r, d] * (1 + scale[d]) + shift[d]
//
// gated_residual:
//   y[r, d] += gate[d] * delta[r, d]
//
// scale/shift/gate are BF16 [hidden_dim], broadcast across all rows.
// (Per-batch modulation can be implemented later as a strided variant —
// for batch=1 inference the [hidden_dim] form is exactly what FLUX uses
// per timestep after the modulation MLP.)

#pragma once

#include <cstddef>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

bool modulate_bf16(const void* x_bf16, void* y_bf16,
                   const void* scale_bf16, const void* shift_bf16,
                   int batch_rows, int hidden_dim,
                   cudaStream_t stream = nullptr);

bool gated_residual_bf16(void* y_bf16, const void* delta_bf16,
                         const void* gate_bf16,
                         int batch_rows, int hidden_dim,
                         cudaStream_t stream = nullptr);

} // namespace f2k::cuda
