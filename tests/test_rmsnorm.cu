// Compares the RMSNorm BF16 kernel against a host FP32 reference.

#include "backend/cuda/kernels/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

// Simple FP32 → BF16 round-to-nearest-even via the intrinsic.
__nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_case(int batch_rows, int dim, float eps, double tol_max, double tol_mean) {
    std::printf("RMSNorm  B=%-4d D=%-5d eps=%.1e  ", batch_rows, dim, eps);

    const size_t bytes_x     = static_cast<size_t>(batch_rows) * dim * sizeof(__nv_bfloat16);
    const size_t bytes_gamma = static_cast<size_t>(dim) * sizeof(__nv_bfloat16);

    std::vector<__nv_bfloat16> h_x(static_cast<size_t>(batch_rows) * dim);
    std::vector<__nv_bfloat16> h_gamma(dim);
    std::vector<__nv_bfloat16> h_y(h_x.size());
    std::vector<float>         h_ref(h_x.size());

    std::mt19937 rng(0xc0ffeeULL);
    std::uniform_real_distribution<float> dx(-2.0f, 2.0f);
    std::uniform_real_distribution<float> dg(0.5f, 1.5f);
    for (auto& v : h_x)     v = fp32_to_bf16(dx(rng));
    for (auto& v : h_gamma) v = fp32_to_bf16(dg(rng));

    // Host reference (FP32 from BF16 inputs)
    const double inv_dim = 1.0 / static_cast<double>(dim);
    for (int b = 0; b < batch_rows; ++b) {
        double ssq = 0.0;
        for (int d = 0; d < dim; ++d) {
            const float v = bf16_to_fp32(h_x[b * dim + d]);
            ssq += static_cast<double>(v) * v;
        }
        const double rrms = 1.0 / std::sqrt(ssq * inv_dim + eps);
        for (int d = 0; d < dim; ++d) {
            const float v = bf16_to_fp32(h_x[b * dim + d]);
            const float g = bf16_to_fp32(h_gamma[d]);
            h_ref[b * dim + d] = static_cast<float>(v * rrms * g);
        }
    }

    void *d_x = nullptr, *d_g = nullptr, *d_y = nullptr;
    cudaMalloc(&d_x, bytes_x);
    cudaMalloc(&d_g, bytes_gamma);
    cudaMalloc(&d_y, bytes_x);
    cudaMemcpy(d_x, h_x.data(),     bytes_x,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, h_gamma.data(), bytes_gamma, cudaMemcpyHostToDevice);

    const bool launched = f2k::cuda::rmsnorm_bf16(d_x, d_g, d_y, batch_rows, dim, eps);
    cudaDeviceSynchronize();
    if (!launched) { std::printf("FAIL (launch error)\n"); return false; }

    cudaMemcpy(h_y.data(), d_y, bytes_x, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_g); cudaFree(d_y);

    double max_err = 0.0, sum_err = 0.0;
    for (size_t i = 0; i < h_y.size(); ++i) {
        const double e = std::fabs(bf16_to_fp32(h_y[i]) - h_ref[i]);
        if (e > max_err) max_err = e;
        sum_err += e;
    }
    const double mean_err = sum_err / static_cast<double>(h_y.size());
    const bool ok = (max_err < tol_max) && (mean_err < tol_mean);
    std::printf("%s  max=%.5f mean=%.5f (tol max=%.3f mean=%.4f)\n",
                ok ? "PASS" : "FAIL", max_err, mean_err, tol_max, tol_mean);
    return ok;
}

} // namespace

int main() {
    bool ok = true;
    // Tolerances reflect BF16's ~3-decimal-digit precision: per-element error
    // is dominated by the round-to-BF16 of the multiply result.
    ok &= run_case(1,    128,  1e-6f, 0.05, 0.005);
    ok &= run_case(8,    1024, 1e-6f, 0.05, 0.005);
    ok &= run_case(32,   3072, 1e-6f, 0.05, 0.005);
    ok &= run_case(128,  4096, 1e-6f, 0.06, 0.005);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
