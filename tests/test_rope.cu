// RoPE 1D in-place test.
// Generates Q [batch, seq, n_heads, head_dim], precomputes cos/sin tables
// with the standard theta-base formula, runs the kernel, and compares
// against a FP32 host reference.

#include "backend/cuda/kernels/rope.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_case(int B, int S, int H, int D, float theta) {
    const int batch_rows = B * S * H;
    const int half_dim   = D / 2;

    std::printf("RoPE B=%d S=%d H=%d D=%d  ", B, S, H, D);

    std::mt19937 rng(0xfeedfaceULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(batch_rows) * D);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));
    std::vector<__nv_bfloat16> hx_original = hx;

    // Build cos/sin tables: theta_i = base^(-2i/D) for i in [0, D/2).
    std::vector<float> cos_table(static_cast<size_t>(S) * half_dim);
    std::vector<float> sin_table(static_cast<size_t>(S) * half_dim);
    for (int p = 0; p < S; ++p) {
        for (int i = 0; i < half_dim; ++i) {
            const float freq = std::pow(theta, -2.0f * i / D);
            const float a    = p * freq;
            cos_table[p * half_dim + i] = std::cos(a);
            sin_table[p * half_dim + i] = std::sin(a);
        }
    }

    // pos_ids: each row's sequence index is its (row / H) % S.
    std::vector<int32_t> pos_ids(batch_rows);
    for (int b = 0; b < B; ++b)
        for (int s = 0; s < S; ++s)
            for (int h = 0; h < H; ++h)
                pos_ids[(b * S + s) * H + h] = s;

    void *d_x, *d_cos, *d_sin, *d_pos;
    cudaMalloc(&d_x,   hx.size() * sizeof(__nv_bfloat16));
    cudaMalloc(&d_cos, cos_table.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_table.size() * sizeof(float));
    cudaMalloc(&d_pos, pos_ids.size() * sizeof(int32_t));
    cudaMemcpy(d_x,   hx.data(),        hx.size() * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_table.data(), cos_table.size() * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_table.data(), sin_table.size() * sizeof(float),  cudaMemcpyHostToDevice);
    cudaMemcpy(d_pos, pos_ids.data(),   pos_ids.size() * sizeof(int32_t),  cudaMemcpyHostToDevice);

    const bool launched = f2k::cuda::rope_inplace_bf16(
        d_x, static_cast<const float*>(d_cos),
        static_cast<const float*>(d_sin),
        static_cast<const int32_t*>(d_pos),
        batch_rows, D, S);
    cudaDeviceSynchronize();
    if (!launched) { std::printf("FAIL launch\n"); return false; }

    cudaMemcpy(hx.data(), d_x, hx.size() * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_pos);

    // Host reference (FP32 from BF16-cast input).
    double max_err = 0, sum_err = 0;
    for (int r = 0; r < batch_rows; ++r) {
        const int p = pos_ids[r];
        for (int i = 0; i < half_dim; ++i) {
            const float x0 = bf16_to_fp32(hx_original[r * D + i]);
            const float x1 = bf16_to_fp32(hx_original[r * D + i + half_dim]);
            const float c  = cos_table[p * half_dim + i];
            const float s  = sin_table[p * half_dim + i];
            const float ref0 = x0 * c - x1 * s;
            const float ref1 = x0 * s + x1 * c;
            const double e0 = std::fabs(bf16_to_fp32(hx[r * D + i])            - ref0);
            const double e1 = std::fabs(bf16_to_fp32(hx[r * D + i + half_dim]) - ref1);
            max_err = std::max({max_err, e0, e1});
            sum_err += e0 + e1;
        }
    }
    const double mean = sum_err / (static_cast<double>(batch_rows) * D);
    const bool ok = (max_err < 0.05) && (mean < 0.005);
    std::printf("%s max=%.4f mean=%.5f\n", ok ? "PASS" : "FAIL", max_err, mean);
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1,  16,  1,  64,  10000.0f);
    ok &= run_case(2,  64,  4,  128, 10000.0f);
    ok &= run_case(1,  256, 8,  128, 500.0f);   // FLUX-like base
    ok &= run_case(4,  512, 24, 128, 10000.0f); // MMDiT-ish dims
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
