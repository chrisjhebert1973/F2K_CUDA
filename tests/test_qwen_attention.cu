// Qwen3 GQA causal attention vs slow host reference.

#include "backend/cuda/qwen_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

void host_gqa_causal_ref(const std::vector<float>& Q, const std::vector<float>& K,
                          const std::vector<float>& V, std::vector<float>& O,
                          int B, int S, int Hq, int Hkv, int D) {
    const float scale = 1.0f / std::sqrt((float)D);
    for (int b = 0; b < B; ++b)
    for (int q = 0; q < S; ++q)
    for (int qh = 0; qh < Hq; ++qh) {
        const int kvh = qh * Hkv / Hq;
        std::vector<float> s(S, -INFINITY);
        float mx = -INFINITY;
        for (int k = 0; k <= q; ++k) {
            float dot = 0.0f;
            for (int d = 0; d < D; ++d) {
                dot += Q[(((b * S + q) * Hq + qh) * D) + d] *
                       K[(((b * S + k) * Hkv + kvh) * D) + d];
            }
            s[k] = dot * scale;
            mx = std::max(mx, s[k]);
        }
        float total = 0.0f;
        for (int k = 0; k <= q; ++k) { s[k] = std::exp(s[k] - mx); total += s[k]; }
        const float inv = 1.0f / (total + 1e-30f);
        for (int d = 0; d < D; ++d) {
            float acc = 0.0f;
            for (int k = 0; k <= q; ++k) {
                acc += s[k] * V[(((b * S + k) * Hkv + kvh) * D) + d];
            }
            O[(((b * S + q) * Hq + qh) * D) + d] = acc * inv;
        }
    }
}

bool run_case(int B, int S, int Hq, int Hkv, int D) {
    std::printf("Qwen-GQA B=%d S=%d Hq=%d Hkv=%d D=%d  ", B, S, Hq, Hkv, D);
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);

    const size_t qn = (size_t)B * S * Hq  * D;
    const size_t kn = (size_t)B * S * Hkv * D;
    std::vector<float> hQ(qn), hK(kn), hV(kn), hO_ref(qn), hO_dev_f(qn);
    std::vector<__nv_bfloat16> bQ(qn), bK(kn), bV(kn), bO(qn);
    for (size_t i = 0; i < qn; ++i) { hQ[i] = d(rng); bQ[i] = fp32_to_bf16(hQ[i]); }
    for (size_t i = 0; i < kn; ++i) { hK[i] = d(rng); bK[i] = fp32_to_bf16(hK[i]); }
    for (size_t i = 0; i < kn; ++i) { hV[i] = d(rng); bV[i] = fp32_to_bf16(hV[i]); }
    // Quantize Q,K,V to BF16 round-trip for fair comparison
    for (size_t i = 0; i < qn; ++i) hQ[i] = bf16_to_fp32(bQ[i]);
    for (size_t i = 0; i < kn; ++i) hK[i] = bf16_to_fp32(bK[i]);
    for (size_t i = 0; i < kn; ++i) hV[i] = bf16_to_fp32(bV[i]);

    host_gqa_causal_ref(hQ, hK, hV, hO_ref, B, S, Hq, Hkv, D);

    void *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, qn * 2); cudaMalloc(&dK, kn * 2);
    cudaMalloc(&dV, kn * 2); cudaMalloc(&dO, qn * 2);
    cudaMemcpy(dQ, bQ.data(), qn * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, bK.data(), kn * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, bV.data(), kn * 2, cudaMemcpyHostToDevice);

    f2k::cuda::QwenGQAAttention::Config cfg{};
    cfg.batch = B; cfg.seq = S; cfg.n_heads = Hq; cfg.n_kv_heads = Hkv; cfg.head_dim = D;
    const bool ok = f2k::cuda::QwenGQAAttention::forward(cfg, dQ, dK, dV, dO);
    cudaDeviceSynchronize();
    cudaMemcpy(bO.data(), dO, qn * 2, cudaMemcpyDeviceToHost);
    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    if (!ok) { std::printf("FAIL kernel\n"); return false; }

    double max_err = 0, sum_err = 0;
    for (size_t i = 0; i < qn; ++i) {
        const float r = hO_ref[i];
        const float a = bf16_to_fp32(bO[i]);
        const float e = std::fabs(r - a);
        max_err = std::max((double)e, max_err);
        sum_err += e;
    }
    const double mean_err = sum_err / qn;
    const bool pass = max_err < 0.05 && mean_err < 0.01;
    std::printf("%s  max=%.4f mean=%.5f\n", pass ? "PASS" : "FAIL", max_err, mean_err);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1,   8,  4, 1, 16);   // tiny sanity, group_size=4
    ok &= run_case(1,  32,  8, 2, 64);   // medium
    ok &= run_case(1, 128, 32, 8, 128);  // Qwen3 shape (seq=128 prompt)
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
