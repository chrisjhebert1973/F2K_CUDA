// SwiGLU end-to-end test against an FP32 reference.

#include "backend/cuda/swiglu.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }
inline float         silu(float x) { return x / (1.0f + std::exp(-x)); }

bool run_case(int M, int H, int F) {
    std::printf("SwiGLU M=%-4d hidden=%-4d ffn=%-5d  ", M, H, F);

    std::mt19937 rng(0xDEADBEEFULL);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);

    std::vector<__nv_bfloat16> hx(static_cast<size_t>(M) * H);
    std::vector<__nv_bfloat16> hW_gate(static_cast<size_t>(F) * H);
    std::vector<__nv_bfloat16> hW_up  (static_cast<size_t>(F) * H);
    std::vector<__nv_bfloat16> hW_down(static_cast<size_t>(H) * F);
    for (auto& v : hx)      v = fp32_to_bf16(d(rng));
    for (auto& v : hW_gate) v = fp32_to_bf16(d(rng));
    for (auto& v : hW_up)   v = fp32_to_bf16(d(rng));
    for (auto& v : hW_down) v = fp32_to_bf16(d(rng));

    f2k::cuda::SwiGLU::Config cfg;
    cfg.batch_rows  = M;
    cfg.hidden_dim  = H;
    cfg.ffn_dim     = F;
    cfg.W_gate_bf16 = hW_gate.data();
    cfg.W_up_bf16   = hW_up.data();
    cfg.W_down_bf16 = hW_down.data();
    f2k::cuda::SwiGLU mlp(cfg);
    if (!mlp.ok()) { std::printf("FAIL ctor: %s\n", mlp.last_error()); return false; }

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x,  hx.size() * 2);
    cudaMalloc(&d_y,  static_cast<size_t>(M) * H * 2);
    cudaMalloc(&d_ws, mlp.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = mlp.forward(d_x, d_y, d_ws, mlp.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL forward: %s\n", mlp.last_error());
               cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws); return false; }
    std::vector<__nv_bfloat16> hy(static_cast<size_t>(M) * H);
    cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);

    // FP32 reference.
    std::vector<float> gate(static_cast<size_t>(M) * F);
    std::vector<float> upv (static_cast<size_t>(M) * F);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < F; ++n) {
            double a = 0, b = 0;
            for (int k = 0; k < H; ++k) {
                a += bf16_to_fp32(hx[m*H+k]) * bf16_to_fp32(hW_gate[n*H+k]);
                b += bf16_to_fp32(hx[m*H+k]) * bf16_to_fp32(hW_up  [n*H+k]);
            }
            gate[m*F+n] = a;
            upv [m*F+n] = b;
        }
    }
    std::vector<float> y_ref(static_cast<size_t>(M) * H, 0.0f);
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < H; ++n) {
            double acc = 0;
            for (int f = 0; f < F; ++f) {
                const float fused = silu(gate[m*F+f]) * upv[m*F+f];
                acc += fused * bf16_to_fp32(hW_down[n*F+f]);
            }
            y_ref[m*H+n] = acc;
        }
    }

    double dot=0, na=0, nb=0, sse=0;
    for (size_t i = 0; i < hy.size(); ++i) {
        const double a = bf16_to_fp32(hy[i]);
        const double b = y_ref[i];
        dot += a*b; na += a*a; nb += b*b;
        sse += (a-b)*(a-b);
    }
    const double cos = dot / std::sqrt(na*nb + 1e-30);
    const double rel = std::sqrt(sse / (nb + 1e-30));
    const bool pass = cos > 0.90 && rel < 0.45;
    std::printf("%s cos=%.4f rel_l2=%.4f\n", pass ? "PASS" : "FAIL", cos, rel);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(128, 256, 512);
    ok &= run_case(128, 256, 1024);
    ok &= run_case(256, 512, 1024);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
