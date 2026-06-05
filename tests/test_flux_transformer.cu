// FluxTransformer end-to-end test on real FLUX.2-klein-9B weights.

#include "backend/cuda/flux_transformer.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_real_model() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("FluxTransformer real-model: SKIPPED\n");
        return true;
    }

    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string())) {
        std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str());
        return false;
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) {
        std::fprintf(stderr, "router: %s\n", router.last_error().c_str());
        return false;
    }
    std::printf("Loaded %d double blocks + %d single blocks from F2K (%.2f GiB)\n",
                router.num_double_blocks(), router.num_single_blocks(),
                ld.total_data_bytes() / (1024.0*1024.0*1024.0));

    f2k::cuda::FluxTransformer::Config cfg;
    cfg.batch    = 1;
    cfg.seq_img  = 128;     // Small for memory; FLUX dims would be ~4096.
    cfg.seq_txt  = 128;
    cfg.in_channels = 128;
    cfg.t5_dim   = 12288;
    cfg.time_dim = 256;
    cfg.n_heads  = 32;
    cfg.head_dim = 128;
    cfg.ffn_dim  = 12288;
    cfg.num_double_blocks = 8;
    cfg.num_single_blocks = 24;
    cfg.rope_theta = 2000.0f;
    cfg.router = &router;

    const auto t0 = std::chrono::steady_clock::now();
    f2k::cuda::FluxTransformer model(cfg);
    if (!model.ok()) { std::fprintf(stderr, "ctor: %s\n", model.last_error()); return false; }
    const auto t1 = std::chrono::steady_clock::now();
    std::printf("Built FluxTransformer in %.2fs, workspace=%.2f MiB\n",
                std::chrono::duration<double>(t1 - t0).count(),
                model.workspace_size_bytes() / (1024.0*1024.0));

    // Random inputs.
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const int rows_img = cfg.batch * cfg.seq_img;
    const int rows_txt = cfg.batch * cfg.seq_txt;
    std::vector<__nv_bfloat16> h_img(static_cast<size_t>(rows_img) * cfg.in_channels);
    std::vector<__nv_bfloat16> h_txt(static_cast<size_t>(rows_txt) * cfg.t5_dim);
    std::vector<__nv_bfloat16> h_time(cfg.time_dim);
    for (auto& v : h_img)  v = fp32_to_bf16(d(rng));
    for (auto& v : h_txt)  v = fp32_to_bf16(d(rng));
    for (auto& v : h_time) v = fp32_to_bf16(d(rng));

    void *d_img, *d_txt, *d_time, *d_out, *d_ws;
    cudaMalloc(&d_img,  h_img.size()  * 2);
    cudaMalloc(&d_txt,  h_txt.size()  * 2);
    cudaMalloc(&d_time, h_time.size() * 2);
    cudaMalloc(&d_out,  h_img.size()  * 2);   // [rows_img, in_channels]
    cudaMalloc(&d_ws,   model.workspace_size_bytes());
    cudaMemcpy(d_img,  h_img.data(),  h_img.size()  * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_txt,  h_txt.data(),  h_txt.size()  * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_time, h_time.data(), h_time.size() * 2, cudaMemcpyHostToDevice);

    const auto t2 = std::chrono::steady_clock::now();
    const bool ok = model.forward(d_img, d_txt, d_time, d_out, d_ws,
                                   model.workspace_size_bytes());
    cudaDeviceSynchronize();
    const auto t3 = std::chrono::steady_clock::now();
    if (!ok) {
        std::fprintf(stderr, "forward: %s\n", model.last_error());
        cudaFree(d_img); cudaFree(d_txt); cudaFree(d_time); cudaFree(d_out); cudaFree(d_ws);
        return false;
    }
    const double secs = std::chrono::duration<double>(t3 - t2).count();
    std::printf("Forward: %.3fs at batch=1 seq_img=%d seq_txt=%d hidden=%d\n",
                secs, cfg.seq_img, cfg.seq_txt, cfg.n_heads * cfg.head_dim);

    std::vector<__nv_bfloat16> h_out(h_img.size());
    cudaMemcpy(h_out.data(), d_out, h_out.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_img); cudaFree(d_txt); cudaFree(d_time); cudaFree(d_out); cudaFree(d_ws);

    int nans = 0; double mx = 0, sm = 0;
    for (auto v : h_out) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++nans;
        mx = std::max(mx, (double)std::fabs(f));
        sm += std::fabs(f);
    }
    const double mean = sm / h_out.size();
    const bool pass = nans == 0 && mx < 1000.0 && mean > 0.005;
    std::printf("Output [%d, %d] BF16:  %s  nans=%d  max=%.3f  mean=%.4f\n",
                rows_img, cfg.in_channels, pass ? "PASS" : "FAIL", nans, mx, mean);
    return pass;
}

} // namespace

int main() {
    bool ok = run_real_model();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
