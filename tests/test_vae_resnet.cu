// VAE ResnetBlock smoke test (synthetic weights; no real-model comparison yet).

#include "backend/cuda/vae_resnet.h"

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

bool run_case(int N, int C_in, int C_out, int H, int W) {
    const bool has_shortcut = (C_in != C_out);
    std::printf("Resnet N=%d %d→%d  %dx%d  shortcut=%s  ",
                N, C_in, C_out, H, W, has_shortcut ? "yes" : "no");
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dw(-0.3f, 0.3f);
    std::uniform_real_distribution<float> dg(0.9f, 1.1f);
    std::uniform_real_distribution<float> db(-0.05f, 0.05f);

    std::vector<__nv_bfloat16> hx((size_t)N*C_in*H*W);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));

    auto fill = [&](size_t n, std::uniform_real_distribution<float>& d) {
        std::vector<__nv_bfloat16> v(n);
        for (auto& x : v) x = fp32_to_bf16(d(rng));
        return v;
    };

    auto n1g = fill(C_in, dg),  n1b = fill(C_in, db);
    auto c1w = fill((size_t)C_out*C_in*9, dw),  c1b = fill(C_out, db);
    auto n2g = fill(C_out, dg), n2b = fill(C_out, db);
    auto c2w = fill((size_t)C_out*C_out*9, dw), c2b = fill(C_out, db);
    auto scw = has_shortcut ? fill((size_t)C_out*C_in, dw) : std::vector<__nv_bfloat16>{};
    auto scb = has_shortcut ? fill(C_out, db)                : std::vector<__nv_bfloat16>{};

    f2k::cuda::ResnetBlock::Config cfg{};
    cfg.N = N; cfg.C_in = C_in; cfg.H = H; cfg.W = W; cfg.C_out = C_out;
    cfg.norm1_gain = n1g.data(); cfg.norm1_bias = n1b.data();
    cfg.conv1_W    = c1w.data(); cfg.conv1_bias = c1b.data();
    cfg.norm2_gain = n2g.data(); cfg.norm2_bias = n2b.data();
    cfg.conv2_W    = c2w.data(); cfg.conv2_bias = c2b.data();
    if (has_shortcut) {
        cfg.shortcut_W    = scw.data();
        cfg.shortcut_bias = scb.data();
    }
    f2k::cuda::ResnetBlock blk(cfg);
    if (!blk.ok()) { std::printf("FAIL ctor: %s\n", blk.last_error()); return false; }

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_y, (size_t)N*C_out*H*W * 2);
    cudaMalloc(&d_ws, blk.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = blk.forward(d_x, d_y, d_ws, blk.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy((size_t)N*C_out*H*W);
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
    const bool pass = nans == 0 && mx < 100.0 && mean > 0.05;
    std::printf("%s  nans=%d max=%.3f mean=%.3f\n",
                pass ? "PASS" : "FAIL", nans, mx, mean);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 512, 512, 16, 16);  // mid-block resnet
    ok &= run_case(1, 512, 256, 32, 32);  // channel transition (up_blocks.2.resnets.0)
    ok &= run_case(1, 256, 128, 64, 64);  // channel transition (up_blocks.3.resnets.0)
    ok &= run_case(1, 128, 128, 128, 128); // late-stage resnet
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
