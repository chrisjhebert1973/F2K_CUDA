// Upsample 2x test.

#include "backend/cuda/kernels/upsample.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_case(int N, int C, int H, int W) {
    std::printf("Upsample N=%d C=%d %dx%d → %dx%d  ", N, C, H, W, H*2, W*2);
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx((size_t)N*C*H*W);
    for (auto& v : hx) v = fp32_to_bf16(d(rng));
    void *d_x, *d_y;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_y, (size_t)N*C*H*W*4 * 2);
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = f2k::cuda::upsample2x_nearest_bf16(d_x, d_y, N, C, H, W);
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy((size_t)N*C*H*W*4);
    cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y);
    if (!ok) { std::printf("FAIL launch\n"); return false; }
    int errs = 0;
    for (int n = 0; n < N; ++n)
    for (int c = 0; c < C; ++c)
    for (int yh = 0; yh < H*2; ++yh)
    for (int yw = 0; yw < W*2; ++yw) {
        const int xh = yh / 2, xw = yw / 2;
        const float ref = bf16_to_fp32(hx[((size_t)n*C + c)*H*W + xh*W + xw]);
        const float got = bf16_to_fp32(hy[((size_t)n*C + c)*H*W*4 + yh*W*2 + yw]);
        if (ref != got) ++errs;
    }
    const bool pass = errs == 0;
    std::printf("%s  errs=%d\n", pass ? "PASS" : "FAIL", errs);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 4,   8,  8);
    ok &= run_case(1, 64, 16, 16);
    ok &= run_case(2, 128, 32, 32);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
