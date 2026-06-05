// Validates Linear's MXFP8 path on random data and contrasts its accuracy with
// the NVFP4 path against an FP32 reference. Expectation: MXFP8 cos_sim ≳ 0.999
// (much tighter than NVFP4's ~0.99) — confirms the E4M3+UE8M0 quant/GEMM and
// the SFA/SFB layout wiring.

#include "backend/cuda/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {
inline __nv_bfloat16 f2b(float v) { return __float2bfloat16(v); }
inline float         b2f(__nv_bfloat16 v) { return __bfloat162float(v); }

struct Metrics { double cos, rel_l2; };

Metrics run(f2k::cuda::Linear::Precision prec, int M, int N, int K,
            const std::vector<__nv_bfloat16>& hx,
            const std::vector<__nv_bfloat16>& hW,
            const std::vector<double>& ref) {
    f2k::cuda::Linear::Config cfg;
    cfg.batch_rows = M; cfg.in_features = K; cfg.out_features = N;
    cfg.precision = prec; cfg.W_bf16 = hW.data();
    f2k::cuda::Linear lin(cfg);
    if (!lin.ok()) { std::printf("  ctor FAIL: %s\n", lin.last_error()); return {-1, 9}; }

    void *dx, *dy, *dws;
    cudaMalloc(&dx, hx.size() * 2);
    cudaMalloc(&dy, (size_t)M * N * 2);
    cudaMalloc(&dws, lin.workspace_size_bytes());
    cudaMemcpy(dx, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = lin.forward(dx, dy, dws, lin.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::printf("  forward FAIL: %s\n", lin.last_error());
               cudaFree(dx); cudaFree(dy); cudaFree(dws); return {-1, 9}; }
    std::vector<__nv_bfloat16> hy((size_t)M * N);
    cudaMemcpy(hy.data(), dy, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dy); cudaFree(dws);

    double dot = 0, na = 0, nb = 0, l2d = 0, l2r = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        const double a = b2f(hy[i]), b = ref[i];
        dot += a * b; na += a * a; nb += b * b;
        l2d += (a - b) * (a - b); l2r += b * b;
    }
    return { dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30),
             std::sqrt(l2d) / (std::sqrt(l2r) + 1e-30) };
}
} // namespace

int main() {
    const int M = 128, N = 512, K = 512;
    std::mt19937 rng(0xC0FFEE12u);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    std::vector<__nv_bfloat16> hx((size_t)M * K), hW((size_t)N * K);
    for (auto& v : hx) v = f2b(dist(rng) * 0.5f);
    for (auto& v : hW) v = f2b(dist(rng) * 0.1f);

    // FP32 reference: y = x @ W^T from the BF16-rounded inputs.
    std::vector<double> ref((size_t)M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double acc = 0;
            for (int k = 0; k < K; ++k)
                acc += (double)b2f(hx[(size_t)m * K + k]) * (double)b2f(hW[(size_t)n * K + k]);
            ref[(size_t)m * N + n] = acc;
        }

    using P = f2k::cuda::Linear::Precision;
    auto f4 = run(P::NVFP4, M, N, K, hx, hW, ref);
    auto f8 = run(P::MXFP8, M, N, K, hx, hW, ref);
    std::printf("NVFP4  cos_sim=%.5f  rel_l2=%.4f\n", f4.cos, f4.rel_l2);
    std::printf("MXFP8  cos_sim=%.5f  rel_l2=%.4f\n", f8.cos, f8.rel_l2);

    const bool pass = f8.cos > 0.998 && f8.cos > f4.cos && f8.rel_l2 < f4.rel_l2;
    std::printf("%s  (MXFP8 should beat NVFP4)\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
