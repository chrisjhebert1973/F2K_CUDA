// Numerical comparison of our FluxTransformer against diffusers on a SINGLE
// forward with identical inputs. Localises any structural bug independently of
// the denoise loop / scheduler / VAE.
//
// Inputs (produced by tools/diffusers_block_dump.py):
//   /tmp/cmp_latent.bin   packed latent  [seq_img, 128]  bf16
//   /tmp/cmp_vel.bin      diffusers out  [seq_img, 128]  bf16
//   <embeds>              prompt_embeds  [seq_txt, 12288] bf16  (same file both sides)
//
// Usage: cmp_transformer <embeds.bin> [sigma]

#include "backend/cuda/flux_transformer.h"
#include "backend/cuda/sampler.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace {
inline float bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

std::vector<__nv_bfloat16> read_bf16(const std::string& path, int& rows, int& cols) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path.c_str()); std::exit(1); }
    int32_t hdr[2]; f.read(reinterpret_cast<char*>(hdr), 8);
    rows = hdr[0]; cols = hdr[1];
    std::vector<__nv_bfloat16> v((size_t)rows * cols);
    f.read(reinterpret_cast<char*>(v.data()), v.size() * sizeof(__nv_bfloat16));
    if (!f) { std::fprintf(stderr, "truncated %s\n", path.c_str()); std::exit(1); }
    return v;
}
} // namespace

int main(int argc, char** argv) {
    namespace fs = std::filesystem;
    if (argc < 2) { std::fprintf(stderr, "usage: cmp_transformer <embeds.bin> [sigma] [nvfp4|fp8]\n"); return 1; }
    const std::string embeds_path = argv[1];
    const float sigma = (argc > 2) ? std::stof(argv[2]) : 1.0f;
    const bool use_fp8 = (argc > 3) && (std::string(argv[3]) == "fp8");

    const char* home = std::getenv("HOME");
    const char* tf_dir = use_fp8 ? "transformer_mxfp8" : "transformer_f2k";
    const fs::path s1 = fs::path(home) / "models/flux2-klein-9B" / tf_dir / "shard-00001.f2k1";
    const fs::path s2 = fs::path(home) / "models/flux2-klein-9B" / tf_dir / "shard-00002.f2k1";

    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string())) {
        std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return 1;
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return 1; }

    // Load inputs.
    int lr, lc, vr, vc, er, ec;
    auto h_latent = read_bf16("/tmp/cmp_latent.bin", lr, lc);   // [seq_img, 128]
    auto h_ref    = read_bf16("/tmp/cmp_vel.bin",    vr, vc);   // [seq_img, 128]
    auto h_embeds = read_bf16(embeds_path,           er, ec);   // [seq_txt, 12288]
    const int seq_img = lr, in_ch = lc, seq_txt = er, t5_dim = ec;
    std::printf("seq_img=%d in_ch=%d  seq_txt=%d t5_dim=%d  sigma=%.4f\n",
                seq_img, in_ch, seq_txt, t5_dim, sigma);

    // Infer a square-ish patch grid.
    int H_p = (int)std::lround(std::sqrt((double)seq_img));
    while (H_p > 1 && seq_img % H_p != 0) --H_p;
    int W_p = seq_img / H_p;
    std::printf("patch grid: %d x %d\n", H_p, W_p);

    f2k::cuda::FluxTransformer::Config cfg;
    cfg.batch = 1; cfg.seq_img = seq_img; cfg.seq_txt = seq_txt;
    cfg.H_patches = H_p; cfg.W_patches = W_p;
    cfg.in_channels = in_ch; cfg.t5_dim = t5_dim; cfg.time_dim = 256;
    cfg.precision = use_fp8 ? f2k::cuda::Precision::MXFP8 : f2k::cuda::Precision::NVFP4;
    cfg.n_heads = 32; cfg.head_dim = 128; cfg.ffn_dim = 12288;
    cfg.num_double_blocks = 8; cfg.num_single_blocks = 24;
    cfg.rope_theta = 2000.0f; cfg.router = &router;
    f2k::cuda::FluxTransformer model(cfg);
    if (!model.ok()) { std::fprintf(stderr, "model ctor: %s\n", model.last_error()); return 1; }

    void *d_latent, *d_txt, *d_out;
    cudaMalloc(&d_latent, h_latent.size() * 2);
    cudaMalloc(&d_txt,    h_embeds.size() * 2);
    cudaMalloc(&d_out,    h_latent.size() * 2);
    cudaMemcpy(d_latent, h_latent.data(), h_latent.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_txt,    h_embeds.data(), h_embeds.size() * 2, cudaMemcpyHostToDevice);

    const auto h_temb = f2k::cuda::compute_timestep_embedding(sigma * 1000.0f, 256);
    void* d_temb; cudaMalloc(&d_temb, 256 * 2);
    cudaMemcpy(d_temb, h_temb.data(), 256 * 2, cudaMemcpyHostToDevice);

    void* d_ws; cudaMalloc(&d_ws, model.workspace_size_bytes());
    if (!model.forward(d_latent, d_txt, d_temb, d_out, d_ws, model.workspace_size_bytes())) {
        std::fprintf(stderr, "forward: %s\n", model.last_error()); return 1;
    }
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> h_out(h_latent.size());
    cudaMemcpy(h_out.data(), d_out, h_out.size() * 2, cudaMemcpyDeviceToHost);

    // Compare ours vs diffusers ref.
    double dot = 0, na = 0, nb = 0, l2diff = 0, l2ref = 0;
    double our_absmax = 0, ref_absmax = 0;
    for (size_t i = 0; i < h_out.size(); ++i) {
        const double a = bf16_to_fp32(h_out[i]);
        const double b = bf16_to_fp32(h_ref[i]);
        dot += a * b; na += a * a; nb += b * b;
        l2diff += (a - b) * (a - b); l2ref += b * b;
        our_absmax = std::max(our_absmax, std::fabs(a));
        ref_absmax = std::max(ref_absmax, std::fabs(b));
    }
    const double cos = dot / (std::sqrt(na) * std::sqrt(nb) + 1e-30);
    const double rel_l2 = std::sqrt(l2diff) / (std::sqrt(l2ref) + 1e-30);
    std::printf("\n=== FluxTransformer vs diffusers (single forward) ===\n");
    std::printf("cos_sim = %.5f   rel_l2 = %.4f\n", cos, rel_l2);
    std::printf("our  std=%.4f absmax=%.3f\n", std::sqrt(na / h_out.size()), our_absmax);
    std::printf("ref  std=%.4f absmax=%.3f\n", std::sqrt(nb / h_out.size()), ref_absmax);
    std::printf("%s\n", cos > 0.9 ? "PASS (structurally matches)" : "FAIL (structural bug remains)");
    return 0;
}
