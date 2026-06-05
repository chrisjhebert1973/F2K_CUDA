// Test ImageEmbedder, ContextEmbedder, FinalProjection — synth + real.

#include "backend/cuda/embeddings.h"
#include "backend/cuda/linear.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

struct PreQ {
    std::vector<uint8_t> packed, scales;
    int N, K;
};

PreQ quantize_for_storage(const std::vector<__nv_bfloat16>& W, int N, int K) {
    PreQ p; p.N = N; p.K = K;
    p.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
    p.scales.assign(static_cast<size_t>(N) * (K / 16), 0);
    constexpr float E2M1_MAX = 6.0f;
    const int n_kblocks = K / 16;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            float absmax = 0.0f;
            for (int i = 0; i < 16; ++i)
                absmax = std::max(absmax, std::fabs(bf16_to_fp32(W[(size_t)n*K + kb*16 + i])));
            const float scale = (absmax > 1e-30f) ? (absmax / E2M1_MAX) : 1.0f;
            cutlass::float_ue4m3_t sf(scale);
            p.scales[(size_t)n*n_kblocks + kb] = (uint8_t)sf.raw();
            for (int i = 0; i < 16; ++i) {
                const int k = kb*16 + i;
                cutlass::float_e2m1_t e2m1(bf16_to_fp32(W[(size_t)n*K + k]) / scale);
                const uint8_t nibble = (uint8_t)e2m1.raw() & 0xFu;
                const size_t bi = (size_t)n*(K/2) + (k >> 1);
                if ((k & 1) == 0) p.packed[bi] = (p.packed[bi] & 0xF0u) | nibble;
                else              p.packed[bi] = (p.packed[bi] & 0x0Fu) | (uint8_t)(nibble << 4);
            }
        }
    }
    return p;
}

std::vector<__nv_bfloat16> rand_weights(int N, int K, std::mt19937& rng) {
    auto glorot = std::uniform_real_distribution<float>(
        -std::sqrt(6.0f / K), std::sqrt(6.0f / K));
    std::vector<__nv_bfloat16> w((size_t)N * K);
    for (auto& v : w) v = fp32_to_bf16(glorot(rng));
    return w;
}

struct Sanity { int nans; double max_abs; double mean_abs; };
Sanity sanity(const std::vector<__nv_bfloat16>& y) {
    int nans = 0; double mx = 0, sm = 0;
    for (auto v : y) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++nans;
        mx = std::max(mx, (double)std::fabs(f));
        sm += std::fabs(f);
    }
    return {nans, mx, sm / y.size()};
}

// ---- Synthetic tests ----

bool synth_image_embedder() {
    const int batch_rows = 128, in_ch = 128, hidden = 1024;
    std::printf("ImageEmbedder synth  rows=%d in=%d hidden=%d  ", batch_rows, in_ch, hidden);
    std::mt19937 rng(0xCAFEBABEULL);
    auto W = rand_weights(hidden, in_ch, rng);
    auto pq = quantize_for_storage(W, hidden, in_ch);
    f2k::cuda::PreQuantNVFP4 P{ pq.packed.data(), pq.scales.data(), hidden, in_ch, 16, 1.0f };

    f2k::cuda::ImageEmbedder::Config cfg;
    cfg.batch_rows = batch_rows; cfg.in_channels = in_ch; cfg.hidden_dim = hidden;
    cfg.W = &P;
    f2k::cuda::ImageEmbedder emb(cfg);
    if (!emb.ok()) { std::printf("FAIL ctor: %s\n", emb.last_error()); return false; }

    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> x(batch_rows * in_ch);
    for (auto& v : x) v = fp32_to_bf16(dx(rng));

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x, x.size() * 2);
    cudaMalloc(&d_y, (size_t)batch_rows * hidden * 2);
    cudaMalloc(&d_ws, emb.workspace_size_bytes());
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = emb.forward(d_x, d_y, d_ws, emb.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> y(batch_rows * hidden);
    cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL forward: %s\n", emb.last_error()); return false; }
    auto s = sanity(y);
    const bool pass = s.nans == 0 && s.max_abs < 50.0 && s.mean_abs > 0.01;
    std::printf("%s nans=%d max=%.3f mean=%.3f\n", pass ? "PASS" : "FAIL", s.nans, s.max_abs, s.mean_abs);
    return pass;
}

bool synth_context_embedder() {
    const int batch_rows = 128, t5 = 1024, hidden = 512;
    std::printf("ContextEmbedder synth rows=%d t5=%d hidden=%d  ", batch_rows, t5, hidden);
    std::mt19937 rng(0xDEADBEEFULL);
    auto W = rand_weights(hidden, t5, rng);
    auto pq = quantize_for_storage(W, hidden, t5);
    f2k::cuda::PreQuantNVFP4 P{ pq.packed.data(), pq.scales.data(), hidden, t5, 16, 1.0f };

    f2k::cuda::ContextEmbedder::Config cfg;
    cfg.batch_rows = batch_rows; cfg.t5_dim = t5; cfg.hidden_dim = hidden; cfg.W = &P;
    f2k::cuda::ContextEmbedder emb(cfg);
    if (!emb.ok()) { std::printf("FAIL ctor: %s\n", emb.last_error()); return false; }

    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> x(batch_rows * t5);
    for (auto& v : x) v = fp32_to_bf16(dx(rng));

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x, x.size() * 2);
    cudaMalloc(&d_y, (size_t)batch_rows * hidden * 2);
    cudaMalloc(&d_ws, emb.workspace_size_bytes());
    cudaMemcpy(d_x, x.data(), x.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = emb.forward(d_x, d_y, d_ws, emb.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> y(batch_rows * hidden);
    cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL forward: %s\n", emb.last_error()); return false; }
    auto s = sanity(y);
    const bool pass = s.nans == 0 && s.max_abs < 50.0 && s.mean_abs > 0.01;
    std::printf("%s nans=%d max=%.3f mean=%.3f\n", pass ? "PASS" : "FAIL", s.nans, s.max_abs, s.mean_abs);
    return pass;
}

bool synth_final_projection() {
    const int batch_rows = 128, hidden = 1024, out_ch = 128;
    std::printf("FinalProjection synth rows=%d hidden=%d out=%d  ", batch_rows, hidden, out_ch);
    std::mt19937 rng(0x12345678ULL);
    auto W = rand_weights(out_ch, hidden, rng);
    auto pq = quantize_for_storage(W, out_ch, hidden);
    f2k::cuda::PreQuantNVFP4 P{ pq.packed.data(), pq.scales.data(), out_ch, hidden, 16, 1.0f };

    f2k::cuda::FinalProjection::Config cfg;
    cfg.batch_rows = batch_rows; cfg.hidden_dim = hidden; cfg.out_channels = out_ch;
    cfg.W_proj_out = &P;
    f2k::cuda::FinalProjection fp(cfg);
    if (!fp.ok()) { std::printf("FAIL ctor: %s\n", fp.last_error()); return false; }

    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(batch_rows * hidden), hsc(hidden), hsh(hidden);
    for (auto& v : hx)  v = fp32_to_bf16(dx(rng));
    std::uniform_real_distribution<float> dm(-0.1f, 0.1f);
    for (auto& v : hsc) v = fp32_to_bf16(dm(rng));
    for (auto& v : hsh) v = fp32_to_bf16(dm(rng));

    void *d_x, *d_y, *d_sc, *d_sh, *d_ws;
    cudaMalloc(&d_x, hx.size() * 2);
    cudaMalloc(&d_y, (size_t)batch_rows * out_ch * 2);
    cudaMalloc(&d_sc, hidden * 2); cudaMalloc(&d_sh, hidden * 2);
    cudaMalloc(&d_ws, fp.workspace_size_bytes());
    cudaMemcpy(d_x,  hx.data(),  hx.size() * 2,  cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc, hsc.data(), hidden * 2,     cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh, hsh.data(), hidden * 2,     cudaMemcpyHostToDevice);
    const bool ok = fp.forward(d_x, d_sc, d_sh, d_y, d_ws, fp.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> y(batch_rows * out_ch);
    cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_sc); cudaFree(d_sh); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL forward: %s\n", fp.last_error()); return false; }
    auto s = sanity(y);
    const bool pass = s.nans == 0 && s.max_abs < 50.0 && s.mean_abs > 0.01;
    std::printf("%s nans=%d max=%.3f mean=%.3f\n", pass ? "PASS" : "FAIL", s.nans, s.max_abs, s.mean_abs);
    return pass;
}

// ---- Real-model tests ----

bool real_model() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("Embeddings real-model:  SKIPPED\n");
        return true;
    }
    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string()))
        { std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return false; }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return false; }

    auto vp = [](const f2k::TensorView* t) {
        f2k::cuda::PreQuantNVFP4 r{};
        r.packed = t->data; r.scales = t->scales;
        r.N = (int)t->shape[0]; r.K = (int)t->shape[1]; return r;
    };
    auto P_x_emb   = vp(router.globals().x_embedder);
    auto P_ctx_emb = vp(router.globals().context_embedder);
    auto P_proj    = vp(router.globals().proj_out);

    const int rows = 256;        // batch_rows multiple of 128
    const int in_ch = P_x_emb.K;   // 128
    const int hidden = P_x_emb.N;  // 4096
    const int t5_dim = P_ctx_emb.K; // 12288

    std::mt19937 rng(0xFEEDFACEULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);

    bool all = true;

    // ImageEmbedder
    {
        f2k::cuda::ImageEmbedder::Config c{rows, in_ch, hidden, &P_x_emb};
        f2k::cuda::ImageEmbedder e(c);
        if (!e.ok()) { std::fprintf(stderr, "img emb ctor: %s\n", e.last_error()); return false; }
        std::vector<__nv_bfloat16> hx((size_t)rows * in_ch);
        for (auto& v : hx) v = fp32_to_bf16(dx(rng));
        void *d_x, *d_y, *d_ws;
        cudaMalloc(&d_x, hx.size() * 2);
        cudaMalloc(&d_y, (size_t)rows * hidden * 2);
        cudaMalloc(&d_ws, e.workspace_size_bytes());
        cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        const bool ok = e.forward(d_x, d_y, d_ws, e.workspace_size_bytes());
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> y((size_t)rows * hidden);
        cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
        if (!ok) { std::fprintf(stderr, "img emb fwd: %s\n", e.last_error()); return false; }
        auto s = sanity(y);
        const bool pass = s.nans == 0 && s.max_abs < 100.0;
        std::printf("  ImageEmbedder    [%d,%d→%d]  %s nans=%d max=%.3f mean=%.3f\n",
                    rows, in_ch, hidden, pass ? "PASS" : "FAIL", s.nans, s.max_abs, s.mean_abs);
        all &= pass;
    }
    // ContextEmbedder
    {
        f2k::cuda::ContextEmbedder::Config c{rows, t5_dim, hidden, &P_ctx_emb};
        f2k::cuda::ContextEmbedder e(c);
        if (!e.ok()) { std::fprintf(stderr, "ctx emb ctor: %s\n", e.last_error()); return false; }
        std::vector<__nv_bfloat16> hx((size_t)rows * t5_dim);
        for (auto& v : hx) v = fp32_to_bf16(dx(rng));
        void *d_x, *d_y, *d_ws;
        cudaMalloc(&d_x, hx.size() * 2);
        cudaMalloc(&d_y, (size_t)rows * hidden * 2);
        cudaMalloc(&d_ws, e.workspace_size_bytes());
        cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        const bool ok = e.forward(d_x, d_y, d_ws, e.workspace_size_bytes());
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> y((size_t)rows * hidden);
        cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
        if (!ok) { std::fprintf(stderr, "ctx emb fwd: %s\n", e.last_error()); return false; }
        auto s = sanity(y);
        const bool pass = s.nans == 0 && s.max_abs < 100.0;
        std::printf("  ContextEmbedder  [%d,%d→%d]  %s nans=%d max=%.3f mean=%.3f\n",
                    rows, t5_dim, hidden, pass ? "PASS" : "FAIL", s.nans, s.max_abs, s.mean_abs);
        all &= pass;
    }
    // FinalProjection
    {
        f2k::cuda::FinalProjection::Config c;
        c.batch_rows = rows; c.hidden_dim = hidden; c.out_channels = in_ch;
        c.W_proj_out = &P_proj;
        f2k::cuda::FinalProjection fp(c);
        if (!fp.ok()) { std::fprintf(stderr, "final ctor: %s\n", fp.last_error()); return false; }
        std::vector<__nv_bfloat16> hx((size_t)rows * hidden), hsc(hidden), hsh(hidden);
        for (auto& v : hx) v = fp32_to_bf16(dx(rng));
        std::uniform_real_distribution<float> dm(-0.1f, 0.1f);
        for (auto& v : hsc) v = fp32_to_bf16(dm(rng));
        for (auto& v : hsh) v = fp32_to_bf16(dm(rng));
        void *d_x, *d_y, *d_sc, *d_sh, *d_ws;
        cudaMalloc(&d_x, hx.size() * 2);
        cudaMalloc(&d_y, (size_t)rows * in_ch * 2);
        cudaMalloc(&d_sc, hidden * 2); cudaMalloc(&d_sh, hidden * 2);
        cudaMalloc(&d_ws, fp.workspace_size_bytes());
        cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sc, hsc.data(), hidden * 2, cudaMemcpyHostToDevice);
        cudaMemcpy(d_sh, hsh.data(), hidden * 2, cudaMemcpyHostToDevice);
        const bool ok = fp.forward(d_x, d_sc, d_sh, d_y, d_ws, fp.workspace_size_bytes());
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> y((size_t)rows * in_ch);
        cudaMemcpy(y.data(), d_y, y.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_sc); cudaFree(d_sh); cudaFree(d_ws);
        if (!ok) { std::fprintf(stderr, "final fwd: %s\n", fp.last_error()); return false; }
        auto s = sanity(y);
        const bool pass = s.nans == 0 && s.max_abs < 100.0;
        std::printf("  FinalProjection  [%d,%d→%d]  %s nans=%d max=%.3f mean=%.3f\n",
                    rows, hidden, in_ch, pass ? "PASS" : "FAIL", s.nans, s.max_abs, s.mean_abs);
        all &= pass;
    }

    return all;
}

} // namespace

int main() {
    bool ok = true;
    ok &= synth_image_embedder();
    ok &= synth_context_embedder();
    ok &= synth_final_projection();
    std::printf("Embeddings real-model:\n");
    ok &= real_model();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
