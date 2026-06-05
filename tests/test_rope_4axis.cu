// Test the 4-axis interleaved RoPE kernel and table builder against a small
// host reference. Verifies the math, indexing, and ordering of the [seq, D/2]
// cos/sin layout.

#include "backend/cuda/kernels/rope_4axis.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

struct Case {
    int seq, head_dim, n_heads;
    std::vector<int> axes_dim;
    float theta;
};

bool run_case(const Case& c) {
    std::printf("rope4axis seq=%d D=%d H=%d axes=[", c.seq, c.head_dim, c.n_heads);
    for (size_t i = 0; i < c.axes_dim.size(); ++i)
        std::printf("%s%d", i ? "," : "", c.axes_dim[i]);
    std::printf("] theta=%.0f  ", c.theta);

    // Build per-axis positions for each token. Sweep (h, w) varies for image-like
    // streams; here we just enumerate seq with axis 1 = s/W, axis 2 = s%W, others
    // = 0 — a typical "image stream" mapping. For 4 axes, we also stash s into
    // axis 0 to exercise every axis at least once.
    const int W = (int)std::sqrt((double)c.seq);
    std::vector<std::vector<int>> positions(c.seq, std::vector<int>(c.axes_dim.size(), 0));
    for (int s = 0; s < c.seq; ++s) {
        positions[s][0] = s % 7;                // T axis — small synthetic
        if (c.axes_dim.size() > 1) positions[s][1] = s / W;
        if (c.axes_dim.size() > 2) positions[s][2] = s % W;
        if (c.axes_dim.size() > 3) positions[s][3] = s;
    }

    std::vector<float> cos_h, sin_h;
    f2k::cuda::build_rope_4axis_tables(c.axes_dim, positions, c.theta, cos_h, sin_h);

    // Random Q row data.
    const int batch_rows = c.seq * c.n_heads;
    std::mt19937 rng(0xCAFEBABEULL);
    std::normal_distribution<float> dn(0.0f, 1.0f);
    std::vector<__nv_bfloat16> x((size_t)batch_rows * c.head_dim);
    for (auto& v : x) v = __float2bfloat16(dn(rng));

    // pos_ids: row r → token s = r / n_heads
    std::vector<int32_t> pos_ids(batch_rows);
    for (int r = 0; r < batch_rows; ++r) pos_ids[r] = r / c.n_heads;

    // Host reference (FP32):
    std::vector<float> ref((size_t)batch_rows * c.head_dim);
    for (int r = 0; r < batch_rows; ++r) {
        const int s = r / c.n_heads;
        const float* cr = cos_h.data() + (size_t)s * (c.head_dim / 2);
        const float* sr = sin_h.data() + (size_t)s * (c.head_dim / 2);
        for (int k = 0; k < c.head_dim / 2; ++k) {
            const float x0 = __bfloat162float(x[(size_t)r * c.head_dim + 2 * k    ]);
            const float x1 = __bfloat162float(x[(size_t)r * c.head_dim + 2 * k + 1]);
            ref[(size_t)r * c.head_dim + 2 * k    ] = x0 * cr[k] - x1 * sr[k];
            ref[(size_t)r * c.head_dim + 2 * k + 1] = x0 * sr[k] + x1 * cr[k];
        }
    }

    // GPU kernel.
    void *d_x; cudaMalloc(&d_x, x.size() * 2);
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    float *d_cos, *d_sin;
    cudaMalloc(&d_cos, cos_h.size() * 4);
    cudaMalloc(&d_sin, sin_h.size() * 4);
    cudaMemcpy(d_cos, cos_h.data(), cos_h.size() * 4, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_h.data(), sin_h.size() * 4, cudaMemcpyHostToDevice);
    int32_t* d_pos; cudaMalloc(&d_pos, pos_ids.size() * 4);
    cudaMemcpy(d_pos, pos_ids.data(), pos_ids.size() * 4, cudaMemcpyHostToDevice);

    const bool ok = f2k::cuda::rope_4axis_inplace_bf16(d_x, d_cos, d_sin, d_pos,
                                                       batch_rows, c.head_dim, c.seq);
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> y(x.size());
    cudaMemcpy(y.data(), d_x, y.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_pos);
    if (!ok) { std::printf("FAIL kernel\n"); return false; }

    double max_err = 0, sum_err = 0;
    for (size_t i = 0; i < y.size(); ++i) {
        const float dev = __bfloat162float(y[i]);
        const float r   = ref[i];
        const float e = std::fabs(dev - r);
        max_err = std::max((double)e, max_err);
        sum_err += e;
    }
    const double mean_err = sum_err / y.size();
    const bool pass = max_err < 0.05 && mean_err < 0.01;
    std::printf("%s  max=%.4f mean=%.5f\n", pass ? "PASS" : "FAIL", max_err, mean_err);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    // Small sanity (4 axes of 4 = head_dim 16)
    ok &= run_case({16, 16, 4, {4, 4, 4, 4}, 100.0f});
    // FLUX2 shape (head_dim=128, axes [32,32,32,32], theta=2000)
    ok &= run_case({256, 128, 32, {32, 32, 32, 32}, 2000.0f});
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
