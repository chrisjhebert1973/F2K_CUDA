// VAEDecoder real-weights smoke test.
//
// Loads /home/chris/models/flux2-klein-9B/vae_f2k/vae.f2k1, builds the decoder,
// feeds a random latent of shape [1, 32, 16, 16], and verifies the output
// pixel tensor [1, 3, 128, 128] is finite and in a reasonable range.

#include "backend/cuda/vae_decoder.h"
#include "common/f2k_format.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

int main() {
    const char* path = "/home/chris/models/flux2-klein-9B/vae_f2k/vae.f2k1";
    f2k::F2KReader r;
    if (!r.open(path)) {
        std::printf("Failed to open %s\n", path);
        return 1;
    }
    std::printf("Loaded VAE: %zu tensors\n", r.names().size());

    const int N = 1, H_lat = 16, W_lat = 16;
    f2k::cuda::VAEDecoder::Config cfg{};
    cfg.N = N; cfg.H_lat = H_lat; cfg.W_lat = W_lat;
    cfg.prefix = "decoder";
    cfg.reader = &r;

    auto t0 = std::chrono::steady_clock::now();
    f2k::cuda::VAEDecoder dec(cfg);
    auto t1 = std::chrono::steady_clock::now();
    if (!dec.ok()) {
        std::printf("FAIL ctor: %s\n", dec.last_error());
        return 1;
    }
    const double build_ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    std::printf("VAEDecoder built in %.2f ms; workspace=%.1f MiB\n",
                build_ms, dec.workspace_size_bytes() / 1048576.0);
    std::printf("Output dims: [%d, 3, %d, %d]\n", N, dec.output_H(), dec.output_W());

    // Build a random latent.
    const size_t in_elems  = (size_t)N * 32 * H_lat * W_lat;
    const size_t out_elems = (size_t)N * 3  * dec.output_H() * dec.output_W();
    std::vector<__nv_bfloat16> h_x(in_elems);
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    for (auto& v : h_x) v = __float2bfloat16(d(rng));

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x, in_elems  * 2);
    cudaMalloc(&d_y, out_elems * 2);
    cudaMalloc(&d_ws, dec.workspace_size_bytes());
    cudaMemcpy(d_x, h_x.data(), in_elems * 2, cudaMemcpyHostToDevice);

    auto t2 = std::chrono::steady_clock::now();
    const bool ok = dec.forward(d_x, d_y, d_ws, dec.workspace_size_bytes());
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    if (!ok) {
        std::printf("FAIL forward: %s\n", dec.last_error());
        return 1;
    }
    const double fwd_ms = std::chrono::duration<double, std::milli>(t3 - t2).count();
    std::printf("Forward in %.2f ms\n", fwd_ms);

    std::vector<__nv_bfloat16> h_y(out_elems);
    cudaMemcpy(h_y.data(), d_y, out_elems * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);

    int nans = 0; double mx = 0, mn = 0, sum = 0;
    for (auto v : h_y) {
        const float f = __bfloat162float(v);
        if (!std::isfinite(f)) ++nans;
        mx = std::max(mx, (double)f);
        mn = std::min(mn, (double)f);
        sum += f;
    }
    const double mean = sum / h_y.size();
    std::printf("pixel stats: min=%.3f max=%.3f mean=%.3f nans=%d\n", mn, mx, mean, nans);

    const bool pass = nans == 0 && mx > -10.0 && mn < 10.0;
    std::printf("%s\n", pass ? "ALL OK" : "FAIL");
    return pass ? 0 : 1;
}
