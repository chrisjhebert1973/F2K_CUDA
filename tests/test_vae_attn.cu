// VAE spatial-self-attention smoke test (synthetic weights).

#include "backend/cuda/vae_attn.h"

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

bool run_case(int N, int C, int H, int W) {
    std::printf("VAEAttn N=%d C=%d %dx%d  S=%d  ", N, C, H, W, H * W);
    std::mt19937 rng(0xDEADBEEFULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dw(-0.05f, 0.05f);
    std::uniform_real_distribution<float> dg(0.9f, 1.1f);
    std::uniform_real_distribution<float> db(-0.05f, 0.05f);

    std::vector<__nv_bfloat16> hx((size_t)N*C*H*W);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));

    auto fill = [&](size_t n, std::uniform_real_distribution<float>& d) {
        std::vector<__nv_bfloat16> v(n);
        for (auto& x : v) x = fp32_to_bf16(d(rng));
        return v;
    };
    auto ng = fill(C, dg), nb = fill(C, db);
    auto qw = fill((size_t)C*C, dw), qb = fill(C, db);
    auto kw = fill((size_t)C*C, dw), kb = fill(C, db);
    auto vw = fill((size_t)C*C, dw), vb = fill(C, db);
    auto ow = fill((size_t)C*C, dw), ob = fill(C, db);

    f2k::cuda::VAEAttention::Config cfg{};
    cfg.N = N; cfg.C = C; cfg.H = H; cfg.W = W;
    cfg.norm_gain = ng.data(); cfg.norm_bias = nb.data();
    cfg.to_q_W = qw.data();    cfg.to_q_b   = qb.data();
    cfg.to_k_W = kw.data();    cfg.to_k_b   = kb.data();
    cfg.to_v_W = vw.data();    cfg.to_v_b   = vb.data();
    cfg.to_out_W = ow.data();  cfg.to_out_b = ob.data();

    f2k::cuda::VAEAttention blk(cfg);
    if (!blk.ok()) { std::printf("FAIL ctor: %s\n", blk.last_error()); return false; }

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_y, hx.size() * 2);
    cudaMalloc(&d_ws, blk.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = blk.forward(d_x, d_y, d_ws, blk.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL fwd: %s\n", blk.last_error()); return false; }

    int nans = 0; double mx = 0, sum = 0;
    for (auto v : hy) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++nans;
        mx = std::max(mx, (double)std::fabs(f));
        sum += std::fabs(f);
    }
    const double mean = sum / hy.size();
    const bool pass = nans == 0 && mx < 20.0 && mean > 0.05;
    std::printf("%s  nans=%d max=%.3f mean=%.3f ws=%.1f MiB\n",
                pass ? "PASS" : "FAIL", nans, mx, mean,
                blk.workspace_size_bytes() / 1048576.0);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 512, 16, 16);  // mid_block at 16×16 (S=256)
    ok &= run_case(1, 512, 32, 32);  // hypothetical higher-res midpoint (S=1024)
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
