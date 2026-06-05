// GroupNorm test — random input, gain, bias → compare to FP32 host reference.

#include "backend/cuda/kernels/groupnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_case(int N, int C, int H, int W, int G) {
    std::printf("GroupNorm N=%d C=%d H=%d W=%d G=%d  ", N, C, H, W, G);
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dg(0.9f, 1.1f);
    std::uniform_real_distribution<float> db(-0.1f, 0.1f);

    std::vector<__nv_bfloat16> hx((size_t)N*C*H*W), hg(C), hb(C);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));
    for (auto& v : hg) v = fp32_to_bf16(dg(rng));
    for (auto& v : hb) v = fp32_to_bf16(db(rng));

    void *d_x, *d_y, *d_g, *d_b;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_y, hx.size() * 2);
    cudaMalloc(&d_g, C * 2);
    cudaMalloc(&d_b, C * 2);
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_g, hg.data(), C * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, hb.data(), C * 2, cudaMemcpyHostToDevice);

    const float eps = 1e-6f;
    const bool ok = f2k::cuda::groupnorm_bf16(d_x, d_y, d_g, d_b, N, C, H, W, G, eps);
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_y, hx.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_g); cudaFree(d_b);
    if (!ok) { std::printf("FAIL launch\n"); return false; }

    // Host reference.
    std::vector<float> ref((size_t)N*C*H*W);
    const int cpg = C / G;
    const int gs  = cpg * H * W;
    for (int n = 0; n < N; ++n)
    for (int g = 0; g < G; ++g) {
        double s = 0, ss = 0;
        for (int lc = 0; lc < cpg; ++lc)
        for (int p = 0; p < H*W; ++p) {
            const float v = bf16_to_fp32(hx[((size_t)n*C + g*cpg + lc)*H*W + p]);
            s += v; ss += (double)v * v;
        }
        const double mean = s / gs;
        const double var  = ss / gs - mean * mean;
        const double rstd = 1.0 / std::sqrt(var + (double)eps);
        for (int lc = 0; lc < cpg; ++lc) {
            const int c_abs = g * cpg + lc;
            const float gv = bf16_to_fp32(hg[c_abs]);
            const float bv = bf16_to_fp32(hb[c_abs]);
            for (int p = 0; p < H*W; ++p) {
                const float v = bf16_to_fp32(hx[((size_t)n*C + c_abs)*H*W + p]);
                ref[((size_t)n*C + c_abs)*H*W + p] = (float)((v - mean) * rstd * gv + bv);
            }
        }
    }
    double max_e = 0, mean_e = 0;
    for (size_t i = 0; i < hy.size(); ++i) {
        const double e = std::fabs(bf16_to_fp32(hy[i]) - ref[i]);
        max_e = std::max(max_e, e);
        mean_e += e;
    }
    mean_e /= hy.size();
    const bool pass = max_e < 0.1 && mean_e < 0.005;
    std::printf("%s  max=%.4f mean=%.5f\n", pass ? "PASS" : "FAIL", max_e, mean_e);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1,  128, 16, 16, 32);   // small
    ok &= run_case(1,  256, 32, 32, 32);   // VAE mid-res
    ok &= run_case(1,  512,  8,  8, 32);   // VAE first stage
    ok &= run_case(1,  128, 64, 64, 32);   // final norm before conv_out
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
