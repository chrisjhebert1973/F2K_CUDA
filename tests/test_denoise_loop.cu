// End-to-end flow-matching denoise loop:
//   - load FLUX.2-klein-9B from F2K
//   - build FluxTransformer + FlowMatchScheduler(4 steps)
//   - for each step: t → timestep_emb → forward(latent) → latent += dt * v_pred
//   - verify final latent is nan-free with sane magnitudes
//
// Uses random text embedding (no T5 yet) and random initial noise, so the
// output is not meaningful — but the full numerical pipeline is exercised.

#include "backend/cuda/flux_transformer.h"
#include "backend/cuda/sampler.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("Denoise loop: SKIPPED (F2K shards not present)\n");
        return true;
    }

    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string())) {
        std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return false;
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return false; }

    f2k::cuda::FluxTransformer::Config cfg;
    cfg.batch = 1; cfg.seq_img = 128; cfg.seq_txt = 128;
    cfg.in_channels = 128; cfg.t5_dim = 12288; cfg.time_dim = 256;
    cfg.n_heads = 32; cfg.head_dim = 128; cfg.ffn_dim = 12288;
    cfg.num_double_blocks = 8; cfg.num_single_blocks = 24;
    cfg.rope_theta = 2000.0f; cfg.router = &router;
    f2k::cuda::FluxTransformer model(cfg);
    if (!model.ok()) { std::fprintf(stderr, "ctor: %s\n", model.last_error()); return false; }
    std::printf("Built FluxTransformer (workspace=%.1f MiB)\n",
                model.workspace_size_bytes() / (1024.0 * 1024.0));

    const int num_steps = 4;
    f2k::cuda::FlowMatchScheduler sched(num_steps);
    std::printf("Scheduler: %d steps, linear t = [", num_steps);
    for (int i = 0; i <= num_steps; ++i)
        std::printf("%.3f%s", sched.t(i), i < num_steps ? ", " : "]\n");

    // Random initial state.
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    const int rows_img = cfg.batch * cfg.seq_img;
    const int rows_txt = cfg.batch * cfg.seq_txt;
    const size_t latent_n = static_cast<size_t>(rows_img) * cfg.in_channels;

    std::vector<__nv_bfloat16> h_x(latent_n);
    std::vector<__nv_bfloat16> h_txt(static_cast<size_t>(rows_txt) * cfg.t5_dim);
    for (auto& v : h_x)   v = fp32_to_bf16(d(rng));
    for (auto& v : h_txt) v = fp32_to_bf16(d(rng));
    const auto h_x_init = h_x;

    void *d_x, *d_v, *d_txt, *d_t_emb, *d_ws;
    cudaMalloc(&d_x,   latent_n * 2);
    cudaMalloc(&d_v,   latent_n * 2);
    cudaMalloc(&d_txt, h_txt.size() * 2);
    cudaMalloc(&d_t_emb, cfg.time_dim * 2);
    cudaMalloc(&d_ws,  model.workspace_size_bytes());
    cudaMemcpy(d_x,   h_x.data(),   latent_n * 2,        cudaMemcpyHostToDevice);
    cudaMemcpy(d_txt, h_txt.data(), h_txt.size() * 2,    cudaMemcpyHostToDevice);

    // Loop.
    const auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < num_steps; ++i) {
        const float t  = sched.t(i);
        const float dt = sched.dt(i);          // negative
        const auto h_t_emb = f2k::cuda::compute_timestep_embedding(t * 1000.0f, cfg.time_dim);
        cudaMemcpy(d_t_emb, h_t_emb.data(), cfg.time_dim * 2, cudaMemcpyHostToDevice);

        if (!model.forward(d_x, d_txt, d_t_emb, d_v, d_ws, model.workspace_size_bytes())) {
            std::fprintf(stderr, "step %d forward: %s\n", i, model.last_error());
            return false;
        }
        if (!f2k::cuda::axpy_bf16(d_x, d_v, dt, latent_n)) {
            std::fprintf(stderr, "step %d axpy\n", i); return false;
        }
        cudaDeviceSynchronize();

        // Per-step diagnostic: read back a small slice.
        std::vector<__nv_bfloat16> tmp(latent_n);
        cudaMemcpy(tmp.data(), d_x, latent_n * 2, cudaMemcpyDeviceToHost);
        int nans = 0; double mx = 0, sum = 0;
        for (auto v : tmp) {
            const float f = bf16_to_fp32(v);
            if (!std::isfinite(f)) ++nans;
            mx = std::max(mx, (double)std::fabs(f));
            sum += std::fabs(f);
        }
        std::printf("  step %d  t=%.3f dt=%+.3f  x[max=%.3f mean=%.3f nans=%d]\n",
                    i, t, dt, mx, sum / tmp.size(), nans);
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double loop_s = std::chrono::duration<double>(t1 - t0).count();
    std::printf("Loop total: %.3fs (%.1f ms / step)\n", loop_s, loop_s * 1000.0 / num_steps);

    std::vector<__nv_bfloat16> h_final(latent_n);
    cudaMemcpy(h_final.data(), d_x, latent_n * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_v); cudaFree(d_txt); cudaFree(d_t_emb); cudaFree(d_ws);

    int nans = 0; double mx = 0, sum = 0, init_sum = 0, drift_sum = 0;
    for (size_t i = 0; i < h_final.size(); ++i) {
        const float f = bf16_to_fp32(h_final[i]);
        if (!std::isfinite(f)) ++nans;
        mx = std::max(mx, (double)std::fabs(f));
        sum += std::fabs(f);
        init_sum += std::fabs(bf16_to_fp32(h_x_init[i]));
        drift_sum += std::fabs(f - bf16_to_fp32(h_x_init[i]));
    }
    const double mean = sum / h_final.size();
    const double init_mean = init_sum / h_final.size();
    const double drift = drift_sum / h_final.size();
    const bool pass = nans == 0 && mx < 1000.0 && mean > 0.05;
    std::printf("Final latent: %s  nans=%d  max=%.3f  mean=%.3f  init_mean=%.3f  mean_drift=%.3f\n",
                pass ? "PASS" : "FAIL", nans, mx, mean, init_mean, drift);
    return pass;
}

} // namespace

int main() {
    bool ok = run();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
