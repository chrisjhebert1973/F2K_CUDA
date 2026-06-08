// Compare our VAEEncoder against the diffusers golden produced by
// tools/diffusers_vae_encode_dump.py.
//
//   build/cmp_vae_encode /tmp/vae_enc_img.bin /tmp/vae_enc_lat.bin
//
// Reads the preprocessed input image [3,H,W] f32, runs the encoder, takes the
// posterior mean (first 32 of 64 channels) and compares cos-sim / max-abs-diff
// against the golden mean latent [32,H/8,W/8] f32.

#include "backend/cuda/vae_encoder.h"
#include "common/f2k_format.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <vector>

static bool read_chw(const char* path, int& C, int& H, int& W, std::vector<float>& data) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path); return false; }
    int hdr[3]; f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    C = hdr[0]; H = hdr[1]; W = hdr[2];
    data.resize((size_t)C * H * W);
    f.read(reinterpret_cast<char*>(data.data()), data.size() * sizeof(float));
    return (bool)f;
}

int main(int argc, char** argv) {
    if (argc < 3) { std::fprintf(stderr, "usage: %s img.bin lat.bin\n", argv[0]); return 2; }
    const char* HOME = std::getenv("HOME");

    int C, H, W; std::vector<float> img;
    if (!read_chw(argv[1], C, H, W, img)) return 1;
    if (C != 3) { std::fprintf(stderr, "expected 3-channel image, got %d\n", C); return 1; }
    std::printf("input image: [%d,%d,%d]\n", C, H, W);

    int gC, gH, gW; std::vector<float> golden;
    if (!read_chw(argv[2], gC, gH, gW, golden)) return 1;
    std::printf("golden mean latent: [%d,%d,%d]\n", gC, gH, gW);

    // Upload image as BF16.
    std::vector<__nv_bfloat16> img_bf(img.size());
    for (size_t i = 0; i < img.size(); ++i) img_bf[i] = __float2bfloat16(img[i]);
    void* d_img = nullptr;
    cudaMalloc(&d_img, img_bf.size() * 2);
    cudaMemcpy(d_img, img_bf.data(), img_bf.size() * 2, cudaMemcpyHostToDevice);

    // Build encoder.
    f2k::F2KReader r;
    std::string vp = std::string(HOME) + "/models/flux2-klein-9B/vae_f2k/vae.f2k1";
    if (!r.open(vp)) { std::fprintf(stderr, "open vae: %s\n", vp.c_str()); return 1; }
    f2k::cuda::VAEEncoder::Config c{};
    c.N = 1; c.H_in = H; c.W_in = W; c.prefix = "encoder"; c.reader = &r;
    f2k::cuda::VAEEncoder enc(c);
    if (!enc.ok()) { std::fprintf(stderr, "encoder ctor: %s\n", enc.last_error()); return 1; }

    const int hl = enc.latent_H(), wl = enc.latent_W();
    if (hl != gH || wl != gW) {
        std::fprintf(stderr, "latent dim mismatch: ours [%d,%d] vs golden [%d,%d]\n", hl, wl, gH, gW);
        return 1;
    }
    const size_t moments_elems = (size_t)64 * hl * wl;
    void *d_moments = nullptr, *d_ws = nullptr;
    cudaMalloc(&d_moments, moments_elems * 2);
    cudaMalloc(&d_ws, enc.workspace_size_bytes());

    if (!enc.forward(d_img, d_moments, d_ws, enc.workspace_size_bytes())) {
        std::fprintf(stderr, "encoder forward: %s\n", enc.last_error()); return 1;
    }
    cudaDeviceSynchronize();

    // Take mean = first 32 channels.
    const size_t mean_elems = (size_t)32 * hl * wl;
    std::vector<__nv_bfloat16> mom_bf(moments_elems);
    cudaMemcpy(mom_bf.data(), d_moments, moments_elems * 2, cudaMemcpyDeviceToHost);

    // Cosine similarity + max abs diff over the mean.
    double dot = 0, na = 0, nb = 0, maxabs = 0;
    int nan_count = 0;
    for (size_t i = 0; i < mean_elems; ++i) {
        float a = __bfloat162float(mom_bf[i]);
        float b = golden[i];
        if (!std::isfinite(a)) { ++nan_count; continue; }
        dot += (double)a * b; na += (double)a * a; nb += (double)b * b;
        maxabs = std::max(maxabs, (double)std::fabs(a - b));
    }
    double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-12);
    std::printf("mean latent [32,%d,%d]: cos_sim=%.5f  max_abs_diff=%.4f  nans=%d\n",
                hl, wl, cos, maxabs, nan_count);
    std::printf("%s\n", cos >= 0.99 ? "PASS (cos >= 0.99)" : "CHECK (cos < 0.99)");
    return cos >= 0.99 ? 0 : 3;
}
