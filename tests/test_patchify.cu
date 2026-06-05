// Patchify roundtrip: x → patchify → unpatchify → x.

#include "backend/cuda/kernels/patchify.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

namespace {

bool run_case(int N, int C, int H, int W, int p) {
    std::printf("Patchify roundtrip N=%d C=%d %dx%d p=%d  ", N, C, H, W, p);
    const int Hp = H / p, Wp = W / p, C_pkt = p * p * C;
    const size_t n_elem = (size_t)N * C * H * W;
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);

    std::vector<__nv_bfloat16> hx(n_elem), hy(n_elem);
    for (auto& v : hx) v = __float2bfloat16(d(rng));

    void *d_x, *d_tok, *d_y;
    cudaMalloc(&d_x, n_elem * 2);
    cudaMalloc(&d_tok, (size_t)N * Hp * Wp * C_pkt * 2);
    cudaMalloc(&d_y, n_elem * 2);
    cudaMemcpy(d_x, hx.data(), n_elem * 2, cudaMemcpyHostToDevice);

    bool ok = f2k::cuda::patchify_bf16(d_x, d_tok, N, C, H, W, p);
    ok &= f2k::cuda::unpatchify_bf16(d_tok, d_y, N, C, H, W, p);
    cudaDeviceSynchronize();
    cudaMemcpy(hy.data(), d_y, n_elem * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_tok); cudaFree(d_y);
    if (!ok) { std::printf("FAIL kernel\n"); return false; }

    int errs = 0;
    for (size_t i = 0; i < n_elem; ++i) {
        if (__bfloat162float(hx[i]) != __bfloat162float(hy[i])) ++errs;
    }
    std::printf("%s  errs=%d\n", errs == 0 ? "PASS" : "FAIL", errs);
    return errs == 0;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_case(1, 32,  16, 16, 2);
    ok &= run_case(1, 32,  32, 32, 2);
    ok &= run_case(1, 32,  16, 32, 2);
    ok &= run_case(2, 16,  8,  8,  2);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
