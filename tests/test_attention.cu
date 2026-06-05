// Self-attention test against FP32 host reference.

#include "backend/cuda/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_case(int B, int S, int H, int D) {
    std::printf("Attn B=%d S=%-4d H=%d D=%-3d  ", B, S, H, D);

    std::mt19937 rng(0xA77E771090ULL);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    const size_t N = static_cast<size_t>(B) * S * H * D;
    std::vector<__nv_bfloat16> hQ(N), hK(N), hV(N);
    for (auto& v : hQ) v = fp32_to_bf16(d(rng));
    for (auto& v : hK) v = fp32_to_bf16(d(rng));
    for (auto& v : hV) v = fp32_to_bf16(d(rng));

    f2k::cuda::Attention::Config cfg{};
    cfg.batch    = B;
    cfg.seq      = S;
    cfg.n_heads  = H;
    cfg.head_dim = D;
    f2k::cuda::Attention att(cfg);
    if (!att.ok()) { std::printf("FAIL ctor: %s\n", att.last_error()); return false; }

    void *dQ, *dK, *dV, *dO;
    const size_t bytes = N * sizeof(__nv_bfloat16);
    cudaMalloc(&dQ, bytes);
    cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes);
    cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, hQ.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, hK.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, hV.data(), bytes, cudaMemcpyHostToDevice);

    const bool ok = att.forward(dQ, dK, dV, dO);
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL fwd: %s\n", att.last_error());
               cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); return false; }
    std::vector<__nv_bfloat16> hO(N);
    cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);

    // FP32 host reference.
    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> ref(N, 0.0f);
    std::vector<double> scores(static_cast<size_t>(S));
    for (int b = 0; b < B; ++b)
    for (int h = 0; h < H; ++h)
    for (int q = 0; q < S; ++q) {
        double mx = -1e30;
        for (int k = 0; k < S; ++k) {
            double dot = 0;
            for (int dd = 0; dd < D; ++dd) {
                dot += static_cast<double>(bf16_to_fp32(hQ[((static_cast<size_t>(b)*S + q)*H + h)*D + dd])) *
                       static_cast<double>(bf16_to_fp32(hK[((static_cast<size_t>(b)*S + k)*H + h)*D + dd]));
            }
            const double s = dot * scale;
            scores[k] = s;
            if (s > mx) mx = s;
        }
        double sum = 0;
        for (int k = 0; k < S; ++k) {
            scores[k] = std::exp(scores[k] - mx);
            sum += scores[k];
        }
        const double inv = 1.0 / sum;
        for (int dd = 0; dd < D; ++dd) {
            double acc = 0;
            for (int k = 0; k < S; ++k) {
                acc += scores[k] * static_cast<double>(bf16_to_fp32(hV[((static_cast<size_t>(b)*S + k)*H + h)*D + dd]));
            }
            ref[((static_cast<size_t>(b)*S + q)*H + h)*D + dd] = static_cast<float>(acc * inv);
        }
    }

    double dot=0, na=0, nb=0, sse=0, max_e=0;
    for (size_t i = 0; i < hO.size(); ++i) {
        const double a = bf16_to_fp32(hO[i]);
        const double b = ref[i];
        dot += a*b; na += a*a; nb += b*b;
        const double e = a-b;
        sse += e*e;
        max_e = std::max(max_e, std::fabs(e));
    }
    const double cos = dot / std::sqrt(na*nb + 1e-30);
    const double rel = std::sqrt(sse / (nb + 1e-30));
    const bool pass = cos > 0.99 && rel < 0.05;
    std::printf("%s cos=%.5f rel_l2=%.4f max_abs=%.4f\n",
                pass ? "PASS" : "FAIL", cos, rel, max_e);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 64,  1, 64);
    ok &= run_case(2, 128, 4, 128);
    ok &= run_case(1, 512, 8, 128);
    ok &= run_case(1, 1024, 4, 128);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
