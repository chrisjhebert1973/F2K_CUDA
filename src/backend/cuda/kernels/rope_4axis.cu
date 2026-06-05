#include "backend/cuda/kernels/rope_4axis.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>

namespace {

constexpr int BLOCK_SIZE = 128;

__global__ void rope_4axis_inplace_kernel(__nv_bfloat16* __restrict__ x,
                                          const float* __restrict__ cos_table,
                                          const float* __restrict__ sin_table,
                                          const int32_t* __restrict__ pos_ids,
                                          int head_dim,
                                          int seq_max) {
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int half_dim = head_dim >> 1;

    const int pos = pos_ids[row];
    const int pos_safe = (pos >= seq_max) ? (seq_max - 1) : (pos < 0 ? 0 : pos);
    const float* cos_row = cos_table + static_cast<size_t>(pos_safe) * half_dim;
    const float* sin_row = sin_table + static_cast<size_t>(pos_safe) * half_dim;

    __nv_bfloat16* row_ptr = x + static_cast<size_t>(row) * head_dim;

    for (int k = tid; k < half_dim; k += BLOCK_SIZE) {
        const float x0 = __bfloat162float(row_ptr[2 * k    ]);
        const float x1 = __bfloat162float(row_ptr[2 * k + 1]);
        const float c  = cos_row[k];
        const float s  = sin_row[k];
        row_ptr[2 * k    ] = __float2bfloat16(x0 * c - x1 * s);
        row_ptr[2 * k + 1] = __float2bfloat16(x0 * s + x1 * c);
    }
}

} // anonymous namespace

namespace f2k::cuda {

bool rope_4axis_inplace_bf16(void* x, const float* cos_table, const float* sin_table,
                              const int32_t* pos_ids,
                              int batch_rows, int head_dim, int seq_max,
                              cudaStream_t stream) {
    if (batch_rows <= 0 || head_dim <= 0 || (head_dim & 1)) return false;
    if (head_dim > 512) return false;
    rope_4axis_inplace_kernel<<<batch_rows, BLOCK_SIZE, 0, stream>>>(
        static_cast<__nv_bfloat16*>(x), cos_table, sin_table, pos_ids,
        head_dim, seq_max);
    return cudaPeekAtLastError() == cudaSuccess;
}

void build_rope_4axis_tables(const std::vector<int>& axes_dim,
                              const std::vector<std::vector<int>>& token_positions,
                              float theta,
                              std::vector<float>& cos_out,
                              std::vector<float>& sin_out) {
    const int K = (int)axes_dim.size();
    int head_dim = 0;
    for (int d : axes_dim) head_dim += d;
    const int half_dim = head_dim / 2;
    const int seq = (int)token_positions.size();

    cos_out.assign((size_t)seq * half_dim, 0.0f);
    sin_out.assign((size_t)seq * half_dim, 0.0f);

    // Precompute per-axis pair frequencies.
    //   freq[a][k] = 1 / theta^(2k / axes_dim[a])  for k in [0, axes_dim[a]/2)
    std::vector<std::vector<double>> freqs(K);
    for (int a = 0; a < K; ++a) {
        const int da = axes_dim[a];
        const int n_pairs = da / 2;
        freqs[a].resize(n_pairs);
        for (int k = 0; k < n_pairs; ++k) {
            freqs[a][k] = std::pow((double)theta, -2.0 * k / (double)da);
        }
    }

    // For each token, fill cos/sin entries axis by axis. Each axis contributes
    // axes_dim[a]/2 consecutive pair-entries to that token's row.
    for (int s = 0; s < seq; ++s) {
        int pair_off = 0;
        for (int a = 0; a < K; ++a) {
            const int n_pairs = axes_dim[a] / 2;
            const int pos_a = token_positions[s][a];
            for (int k = 0; k < n_pairs; ++k) {
                const double angle = (double)pos_a * freqs[a][k];
                cos_out[(size_t)s * half_dim + pair_off + k] = (float)std::cos(angle);
                sin_out[(size_t)s * half_dim + pair_off + k] = (float)std::sin(angle);
            }
            pair_off += n_pairs;
        }
    }
}

} // namespace f2k::cuda
