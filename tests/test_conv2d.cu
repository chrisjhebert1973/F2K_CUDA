// Conv2d test — random W + x → cuDNN forward → compare to host 7-loop ref.

#include "backend/cuda/conv2d.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

// Slow host reference: NCHW input/output, NCHW weights [C_out, C_in, kH, kW].
std::vector<float> host_conv2d_fp32(const std::vector<__nv_bfloat16>& x,
                                    const std::vector<__nv_bfloat16>& W,
                                    const std::vector<__nv_bfloat16>& b,
                                    int N, int C_in, int H, int Wd,
                                    int C_out, int kH, int kW,
                                    int padding) {
    const int H_out = H + 2*padding - kH + 1;
    const int W_out = Wd + 2*padding - kW + 1;
    std::vector<float> y((size_t)N * C_out * H_out * W_out, 0.0f);
    for (int n = 0; n < N; ++n)
    for (int co = 0; co < C_out; ++co) {
        const float bias = b.empty() ? 0.0f : bf16_to_fp32(b[co]);
        for (int yh = 0; yh < H_out; ++yh)
        for (int yw = 0; yw < W_out; ++yw) {
            double acc = 0;
            for (int ci = 0; ci < C_in; ++ci)
            for (int kh = 0; kh < kH; ++kh)
            for (int kw = 0; kw < kW; ++kw) {
                const int ih = yh + kh - padding;
                const int iw = yw + kw - padding;
                if (ih < 0 || ih >= H || iw < 0 || iw >= Wd) continue;
                const float xv = bf16_to_fp32(x[((size_t)n*C_in + ci)*H*Wd + ih*Wd + iw]);
                const float wv = bf16_to_fp32(W[(((size_t)co*C_in + ci)*kH + kh)*kW + kw]);
                acc += (double)xv * (double)wv;
            }
            y[((size_t)n*C_out + co)*H_out*W_out + yh*W_out + yw] = (float)(acc + bias);
        }
    }
    return y;
}

bool run_case(int N, int C_in, int H, int Wd, int C_out, int kH, int kW, int padding) {
    std::printf("Conv2d N=%d  %dx%dx%d  k=%dx%d pad=%d → C_out=%d  ",
                N, C_in, H, Wd, kH, kW, padding, C_out);
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dw(-0.3f, 0.3f);

    std::vector<__nv_bfloat16> hx((size_t)N*C_in*H*Wd);
    std::vector<__nv_bfloat16> hW((size_t)C_out*C_in*kH*kW);
    std::vector<__nv_bfloat16> hb(C_out);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));
    for (auto& v : hW) v = fp32_to_bf16(dw(rng));
    for (auto& v : hb) v = fp32_to_bf16(dx(rng) * 0.1f);

    f2k::cuda::Conv2d::Config cfg{};
    cfg.N = N; cfg.C_in = C_in; cfg.H_in = H; cfg.W_in = Wd;
    cfg.C_out = C_out; cfg.kH = kH; cfg.kW = kW;
    cfg.stride = 1; cfg.padding = padding;
    cfg.W_bf16 = hW.data();
    cfg.bias_bf16 = hb.data();
    f2k::cuda::Conv2d conv(cfg);
    if (!conv.ok()) { std::printf("FAIL ctor: %s\n", conv.last_error()); return false; }

    const int H_out = conv.H_out();
    const int W_out = conv.W_out();
    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_y, (size_t)N*C_out*H_out*W_out * 2);
    cudaMalloc(&d_ws, std::max<size_t>(1, conv.workspace_size_bytes()));
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = conv.forward(d_x, d_y, d_ws, conv.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy((size_t)N*C_out*H_out*W_out);
    cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL fwd: %s\n", conv.last_error()); return false; }

    auto ref = host_conv2d_fp32(hx, hW, hb, N, C_in, H, Wd, C_out, kH, kW, padding);
    double dot=0, na=0, nb=0, sse=0, mx=0;
    for (size_t i = 0; i < hy.size(); ++i) {
        const double a = bf16_to_fp32(hy[i]);
        const double b = ref[i];
        dot += a*b; na += a*a; nb += b*b;
        const double e = a-b;
        sse += e*e;
        mx = std::max(mx, std::fabs(e));
    }
    const double cos = dot / std::sqrt(na*nb + 1e-30);
    const double rel = std::sqrt(sse / (nb + 1e-30));
    const bool pass = cos > 0.99 && rel < 0.05;
    std::printf("%s  cos=%.5f rel=%.4f max=%.4f\n",
                pass ? "PASS" : "FAIL", cos, rel, mx);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 32,  16, 16,  64,  3, 3, 1);     // tiny
    ok &= run_case(1, 32,  16, 16, 512,  3, 3, 1);     // VAE conv_in shape
    ok &= run_case(1, 128, 32, 32, 256,  1, 1, 0);     // 1x1 (conv_shortcut)
    ok &= run_case(1, 128, 64, 64,   3,  3, 3, 1);     // VAE conv_out shape
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
