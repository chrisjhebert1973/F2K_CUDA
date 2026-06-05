// Qwen3-8B encoder smoke test with real F2K weights.
//
// Sanity-checks that the 36-layer stack runs end-to-end without NaNs and that
// the final output has plausible magnitudes. Does NOT compare against an HF
// reference (that requires a separate Python tool — see tools/qwen3_golden.py).

#include "backend/cuda/qwen_encoder.h"
#include "common/f2k_model_loader.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <random>
#include <vector>

namespace fs = std::filesystem;

int main() {
    const char* home = std::getenv("HOME");
    f2k::F2KModelLoader ld;
    for (int i = 1; i <= 4; ++i) {
        char p[256];
        std::snprintf(p, sizeof(p), "%s/models/flux2-klein-9B/qwen3_f2k/shard-%05d.f2k1", home, i);
        if (!fs::exists(p)) { std::printf("SKIP: %s missing\n", p); return 0; }
        if (!ld.add_shard(p)) { std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return 1; }
    }
    std::printf("Loaded %zu Qwen3 tensors across %zu shards\n", ld.total_tensors(), ld.num_shards());

    f2k::cuda::QwenEncoder::Config cfg{};
    cfg.seq = 128;
    cfg.loader = &ld;
    auto t0 = std::chrono::steady_clock::now();
    f2k::cuda::QwenEncoder enc(cfg);
    auto t1 = std::chrono::steady_clock::now();
    if (!enc.ok()) { std::fprintf(stderr, "ctor: %s\n", enc.last_error()); return 1; }
    std::printf("QwenEncoder built in %.2fs (workspace=%.1f MiB; output_hidden=%d)\n",
                std::chrono::duration<double>(t1 - t0).count(),
                enc.workspace_size_bytes() / 1048576.0,
                enc.output_hidden());

    // Dummy token IDs (random in vocab range — not real text, just a numeric smoke).
    std::mt19937 rng(0xCAFEBABEULL);
    std::uniform_int_distribution<int> di(0, 151000);
    std::vector<int32_t> h_ids(cfg.seq);
    for (auto& v : h_ids) v = di(rng);

    int32_t* d_ids = nullptr;
    cudaMalloc(&d_ids, h_ids.size() * sizeof(int32_t));
    cudaMemcpy(d_ids, h_ids.data(), h_ids.size() * sizeof(int32_t), cudaMemcpyHostToDevice);

    const size_t out_elems = (size_t)cfg.seq * enc.output_hidden();
    void *d_out = nullptr, *d_ws = nullptr;
    cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_ws, enc.workspace_size_bytes());

    auto t2 = std::chrono::steady_clock::now();
    bool ok = enc.forward(d_ids, d_out, d_ws, enc.workspace_size_bytes());
    cudaDeviceSynchronize();
    auto t3 = std::chrono::steady_clock::now();
    if (!ok) { std::fprintf(stderr, "fwd: %s\n", enc.last_error()); return 1; }
    std::printf("Forward: %.2fs\n", std::chrono::duration<double>(t3 - t2).count());

    std::vector<__nv_bfloat16> h_out(out_elems);
    cudaMemcpy(h_out.data(), d_out, out_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaFree(d_ids); cudaFree(d_out); cudaFree(d_ws);

    int nans = 0; double mn = 1e9, mx = -1e9, sum = 0;
    for (auto v : h_out) {
        const float f = __bfloat162float(v);
        if (!std::isfinite(f)) ++nans;
        mn = std::min(mn, (double)f); mx = std::max(mx, (double)f); sum += f;
    }
    const double mean = sum / out_elems;
    std::printf("Output stats: min=%.3f max=%.3f mean=%.3f nans=%d  (rows=%d, cols=%d)\n",
                mn, mx, mean, nans, cfg.seq, enc.output_hidden());

    const bool pass = nans == 0 && std::isfinite(mn) && std::isfinite(mx) && mx > -100 && mn < 100;
    std::printf("%s\n", pass ? "ALL OK" : "FAIL");
    return pass ? 0 : 1;
}
