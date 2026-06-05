// JointAttention test vs FP32 reference.
//
// The reference is just standard attention computed on the concatenated
// [batch, seq_txt + seq_img, n_heads, head_dim] tensor, then split back into
// (O_txt, O_img). Our GPU JointAttention should match this within BF16 noise.

#include "backend/cuda/joint_attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_case(int B, int S_t, int S_i, int H, int D) {
    std::printf("JointAttn B=%d S_t=%-4d S_i=%-4d H=%d D=%-3d  ", B, S_t, S_i, H, D);

    std::mt19937 rng(0xABCDEF12ULL);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    const size_t Nt = static_cast<size_t>(B) * S_t * H * D;
    const size_t Ni = static_cast<size_t>(B) * S_i * H * D;
    std::vector<__nv_bfloat16> hQt(Nt), hKt(Nt), hVt(Nt);
    std::vector<__nv_bfloat16> hQi(Ni), hKi(Ni), hVi(Ni);
    for (auto& v : hQt) v = fp32_to_bf16(d(rng));
    for (auto& v : hKt) v = fp32_to_bf16(d(rng));
    for (auto& v : hVt) v = fp32_to_bf16(d(rng));
    for (auto& v : hQi) v = fp32_to_bf16(d(rng));
    for (auto& v : hKi) v = fp32_to_bf16(d(rng));
    for (auto& v : hVi) v = fp32_to_bf16(d(rng));

    f2k::cuda::JointAttention::Config cfg{};
    cfg.batch = B; cfg.seq_txt = S_t; cfg.seq_img = S_i;
    cfg.n_heads = H; cfg.head_dim = D;
    f2k::cuda::JointAttention ja(cfg);
    if (!ja.ok()) { std::printf("FAIL ctor: %s\n", ja.last_error()); return false; }

    const size_t b_t = Nt * sizeof(__nv_bfloat16);
    const size_t b_i = Ni * sizeof(__nv_bfloat16);
    void *dQt, *dKt, *dVt, *dQi, *dKi, *dVi, *dOt, *dOi, *dws;
    cudaMalloc(&dQt, b_t); cudaMalloc(&dKt, b_t); cudaMalloc(&dVt, b_t);
    cudaMalloc(&dQi, b_i); cudaMalloc(&dKi, b_i); cudaMalloc(&dVi, b_i);
    cudaMalloc(&dOt, b_t); cudaMalloc(&dOi, b_i);
    cudaMalloc(&dws, ja.workspace_size_bytes());
    cudaMemcpy(dQt, hQt.data(), b_t, cudaMemcpyHostToDevice);
    cudaMemcpy(dKt, hKt.data(), b_t, cudaMemcpyHostToDevice);
    cudaMemcpy(dVt, hVt.data(), b_t, cudaMemcpyHostToDevice);
    cudaMemcpy(dQi, hQi.data(), b_i, cudaMemcpyHostToDevice);
    cudaMemcpy(dKi, hKi.data(), b_i, cudaMemcpyHostToDevice);
    cudaMemcpy(dVi, hVi.data(), b_i, cudaMemcpyHostToDevice);

    const bool ok = ja.forward(dQt, dKt, dVt, dQi, dKi, dVi, dOt, dOi,
                                dws, ja.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL fwd: %s\n", ja.last_error());
        cudaFree(dQt); cudaFree(dKt); cudaFree(dVt);
        cudaFree(dQi); cudaFree(dKi); cudaFree(dVi);
        cudaFree(dOt); cudaFree(dOi); cudaFree(dws); return false; }
    std::vector<__nv_bfloat16> hOt(Nt), hOi(Ni);
    cudaMemcpy(hOt.data(), dOt, b_t, cudaMemcpyDeviceToHost);
    cudaMemcpy(hOi.data(), dOi, b_i, cudaMemcpyDeviceToHost);
    cudaFree(dQt); cudaFree(dKt); cudaFree(dVt);
    cudaFree(dQi); cudaFree(dKi); cudaFree(dVi);
    cudaFree(dOt); cudaFree(dOi); cudaFree(dws);

    // FP32 reference: concat on host, attention on concat, split.
    const int S = S_t + S_i;
    auto cat = [&](const std::vector<__nv_bfloat16>& A, int S_a,
                   const std::vector<__nv_bfloat16>& C, int S_c)
        -> std::vector<__nv_bfloat16>
    {
        std::vector<__nv_bfloat16> out(static_cast<size_t>(B)*S*H*D);
        for (int b = 0; b < B; ++b) {
            for (int s = 0; s < S_a; ++s)
                for (int hh = 0; hh < H; ++hh)
                    for (int dd = 0; dd < D; ++dd)
                        out[((static_cast<size_t>(b)*S + s)*H + hh)*D + dd] =
                          A[((static_cast<size_t>(b)*S_a + s)*H + hh)*D + dd];
            for (int s = 0; s < S_c; ++s)
                for (int hh = 0; hh < H; ++hh)
                    for (int dd = 0; dd < D; ++dd)
                        out[((static_cast<size_t>(b)*S + (S_a + s))*H + hh)*D + dd] =
                          C[((static_cast<size_t>(b)*S_c + s)*H + hh)*D + dd];
        }
        return out;
    };
    auto Qc = cat(hQt, S_t, hQi, S_i);
    auto Kc = cat(hKt, S_t, hKi, S_i);
    auto Vc = cat(hVt, S_t, hVi, S_i);

    const float scale = 1.0f / std::sqrt(static_cast<float>(D));
    std::vector<float> ref_O(static_cast<size_t>(B)*S*H*D, 0.0f);
    std::vector<double> scores(S);
    for (int b = 0; b < B; ++b)
    for (int hh = 0; hh < H; ++hh)
    for (int q = 0; q < S; ++q) {
        double mx = -1e30;
        for (int k = 0; k < S; ++k) {
            double dot = 0;
            for (int dd = 0; dd < D; ++dd) {
                dot += static_cast<double>(bf16_to_fp32(Qc[((static_cast<size_t>(b)*S + q)*H + hh)*D + dd])) *
                       static_cast<double>(bf16_to_fp32(Kc[((static_cast<size_t>(b)*S + k)*H + hh)*D + dd]));
            }
            const double s = dot * scale;
            scores[k] = s;
            if (s > mx) mx = s;
        }
        double sum = 0;
        for (int k = 0; k < S; ++k) { scores[k] = std::exp(scores[k] - mx); sum += scores[k]; }
        const double inv = 1.0 / sum;
        for (int dd = 0; dd < D; ++dd) {
            double acc = 0;
            for (int k = 0; k < S; ++k)
                acc += scores[k] * static_cast<double>(bf16_to_fp32(Vc[((static_cast<size_t>(b)*S + k)*H + hh)*D + dd]));
            ref_O[((static_cast<size_t>(b)*S + q)*H + hh)*D + dd] = static_cast<float>(acc * inv);
        }
    }
    // Split reference into ref_t / ref_i.
    std::vector<float> ref_t(Nt), ref_i(Ni);
    for (int b = 0; b < B; ++b) {
        for (int s = 0; s < S_t; ++s)
            for (int hh = 0; hh < H; ++hh)
                for (int dd = 0; dd < D; ++dd)
                    ref_t[((static_cast<size_t>(b)*S_t + s)*H + hh)*D + dd] =
                      ref_O[((static_cast<size_t>(b)*S + s)*H + hh)*D + dd];
        for (int s = 0; s < S_i; ++s)
            for (int hh = 0; hh < H; ++hh)
                for (int dd = 0; dd < D; ++dd)
                    ref_i[((static_cast<size_t>(b)*S_i + s)*H + hh)*D + dd] =
                      ref_O[((static_cast<size_t>(b)*S + (S_t + s))*H + hh)*D + dd];
    }

    auto compare = [](const std::vector<__nv_bfloat16>& got,
                      const std::vector<float>& ref, double& cos, double& rel, double& mx)
    {
        double dot = 0, na = 0, nb = 0, sse = 0; mx = 0;
        for (size_t i = 0; i < got.size(); ++i) {
            const double a = bf16_to_fp32(got[i]);
            const double b = ref[i];
            dot += a*b; na += a*a; nb += b*b;
            const double e = a - b;
            sse += e*e;
            mx = std::max(mx, std::fabs(e));
        }
        cos = dot / std::sqrt(na*nb + 1e-30);
        rel = std::sqrt(sse / (nb + 1e-30));
    };
    double cos_t, rel_t, mx_t, cos_i, rel_i, mx_i;
    compare(hOt, ref_t, cos_t, rel_t, mx_t);
    compare(hOi, ref_i, cos_i, rel_i, mx_i);
    const bool pass = cos_t > 0.99 && cos_i > 0.99 && rel_t < 0.05 && rel_i < 0.05;
    std::printf("%s txt[cos=%.5f rel=%.4f] img[cos=%.5f rel=%.4f]\n",
                pass ? "PASS" : "FAIL", cos_t, rel_t, cos_i, rel_i);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 32, 64,  4, 64);
    ok &= run_case(1, 64, 128, 4, 128);
    ok &= run_case(2, 32, 256, 8, 128);
    ok &= run_case(1, 64, 512, 8, 128);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
