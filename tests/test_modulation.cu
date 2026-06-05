// Modulation + gated_residual: compare to FP32 host reference.

#include "backend/cuda/kernels/modulation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool test_modulate(int rows, int D) {
    std::printf("modulate rows=%-4d D=%-4d  ", rows, D);
    std::mt19937 rng(0x1234ULL);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(rows)*D);
    std::vector<__nv_bfloat16> hs(D), hb(D);
    for (auto& v : hx) v = fp32_to_bf16(dist(rng));
    for (auto& v : hs) v = fp32_to_bf16(dist(rng) * 0.5f);
    for (auto& v : hb) v = fp32_to_bf16(dist(rng) * 0.2f);

    void *dx, *dy, *ds, *db;
    const size_t bytes = hx.size() * sizeof(__nv_bfloat16);
    cudaMalloc(&dx, bytes);
    cudaMalloc(&dy, bytes);
    cudaMalloc(&ds, D * sizeof(__nv_bfloat16));
    cudaMalloc(&db, D * sizeof(__nv_bfloat16));
    cudaMemcpy(dx, hx.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(ds, hs.data(), D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb.data(), D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    const bool ok = f2k::cuda::modulate_bf16(dx, dy, ds, db, rows, D);
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL launch\n"); return false; }
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dx); cudaFree(dy); cudaFree(ds); cudaFree(db);

    double max_e = 0, sum_e = 0;
    for (int r = 0; r < rows; ++r)
        for (int d = 0; d < D; ++d) {
            const float ref = bf16_to_fp32(hx[r*D+d]) * (1.0f + bf16_to_fp32(hs[d])) +
                              bf16_to_fp32(hb[d]);
            const double e = std::fabs(bf16_to_fp32(hy[r*D+d]) - ref);
            max_e = std::max(max_e, e);
            sum_e += e;
        }
    const double mean = sum_e / (static_cast<double>(rows) * D);
    const bool pass = max_e < 0.05 && mean < 0.005;
    std::printf("%s max=%.5f mean=%.5f\n", pass ? "PASS" : "FAIL", max_e, mean);
    return pass;
}

bool test_gated_residual(int rows, int D) {
    std::printf("gated_res rows=%-4d D=%-4d ", rows, D);
    std::mt19937 rng(0x5678ULL);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hy(static_cast<size_t>(rows)*D);
    std::vector<__nv_bfloat16> hd(static_cast<size_t>(rows)*D);
    std::vector<__nv_bfloat16> hg(D);
    for (auto& v : hy) v = fp32_to_bf16(dist(rng));
    for (auto& v : hd) v = fp32_to_bf16(dist(rng));
    for (auto& v : hg) v = fp32_to_bf16(dist(rng) * 0.3f);
    std::vector<__nv_bfloat16> hy_orig = hy;

    void *dy, *dd, *dg;
    const size_t bytes = hy.size() * sizeof(__nv_bfloat16);
    cudaMalloc(&dy, bytes);
    cudaMalloc(&dd, bytes);
    cudaMalloc(&dg, D * sizeof(__nv_bfloat16));
    cudaMemcpy(dy, hy.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dd, hd.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dg, hg.data(), D * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    const bool ok = f2k::cuda::gated_residual_bf16(dy, dd, dg, rows, D);
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL launch\n"); return false; }
    cudaMemcpy(hy.data(), dy, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dy); cudaFree(dd); cudaFree(dg);

    double max_e = 0, sum_e = 0;
    for (int r = 0; r < rows; ++r)
        for (int d = 0; d < D; ++d) {
            const float ref = bf16_to_fp32(hy_orig[r*D+d]) +
                              bf16_to_fp32(hg[d]) * bf16_to_fp32(hd[r*D+d]);
            const double e = std::fabs(bf16_to_fp32(hy[r*D+d]) - ref);
            max_e = std::max(max_e, e);
            sum_e += e;
        }
    const double mean = sum_e / (static_cast<double>(rows) * D);
    const bool pass = max_e < 0.05 && mean < 0.005;
    std::printf("%s max=%.5f mean=%.5f\n", pass ? "PASS" : "FAIL", max_e, mean);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_modulate(64, 128);
    ok &= test_modulate(256, 3072);
    ok &= test_gated_residual(64, 128);
    ok &= test_gated_residual(256, 3072);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
