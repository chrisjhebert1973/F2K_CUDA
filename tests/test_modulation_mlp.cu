// ModulationMLP test:
//   1. Synthetic: random weights + input → all 17 output vectors non-NaN, sane magnitudes.
//   2. Real: load the 6 modulation linears from F2K (via TensorRouter globals),
//      run with random timestep_emb, verify.

#include "backend/cuda/modulation_mlp.h"
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

bool check_vec(const void* d, int hidden, int& nans, double& mx, double& mn, const char* tag) {
    std::vector<__nv_bfloat16> h(hidden);
    cudaMemcpy(h.data(), d, hidden * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    int b = 0;
    double max_abs = 0, sum_abs = 0;
    for (auto v : h) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++b;
        max_abs = std::max(max_abs, (double)std::fabs(f));
        sum_abs += std::fabs(f);
    }
    nans = b;
    mx = max_abs;
    mn = sum_abs / hidden;
    return b == 0;
}

bool run_synth() {
    const int hidden = 1024, time_dim = 256;
    std::printf("ModulationMLP synth hidden=%d time_dim=%d  ", hidden, time_dim);
    std::mt19937 rng(0xCAFEBABEULL);
    auto glorot = [&](int fan_in) {
        const float a = std::sqrt(6.0f / fan_in);
        return std::uniform_real_distribution<float>(-a, a);
    };
    auto fill = [&](int N, int K) {
        std::vector<__nv_bfloat16> w((size_t)N * K);
        auto d = glorot(K);
        for (auto& v : w) v = fp32_to_bf16(d(rng));
        return w;
    };
    auto W_t1 = fill(hidden, time_dim);
    auto W_t2 = fill(hidden, hidden);
    auto W_di = fill(6*hidden, hidden);
    auto W_dt = fill(6*hidden, hidden);
    auto W_sg = fill(3*hidden, hidden);
    auto W_no = fill(2*hidden, hidden);

    auto pq_t1 = quantize_for_storage(W_t1, hidden, time_dim);
    auto pq_t2 = quantize_for_storage(W_t2, hidden, hidden);
    auto pq_di = quantize_for_storage(W_di, 6*hidden, hidden);
    auto pq_dt = quantize_for_storage(W_dt, 6*hidden, hidden);
    auto pq_sg = quantize_for_storage(W_sg, 3*hidden, hidden);
    auto pq_no = quantize_for_storage(W_no, 2*hidden, hidden);

    auto vp = [](const PreQ& q) {
        f2k::cuda::PreQuantNVFP4 r{};
        r.packed = q.packed.data(); r.scales = q.scales.data();
        r.N = q.N; r.K = q.K; return r;
    };
    f2k::cuda::PreQuantNVFP4 P_t1 = vp(pq_t1), P_t2 = vp(pq_t2);
    f2k::cuda::PreQuantNVFP4 P_di = vp(pq_di), P_dt = vp(pq_dt);
    f2k::cuda::PreQuantNVFP4 P_sg = vp(pq_sg), P_no = vp(pq_no);

    f2k::cuda::ModulationMLP::Config cfg;
    cfg.hidden_dim = hidden; cfg.time_dim = time_dim;
    cfg.W_time_1 = &P_t1; cfg.W_time_2 = &P_t2;
    cfg.W_double_img = &P_di; cfg.W_double_txt = &P_dt;
    cfg.W_single = &P_sg; cfg.W_norm_out = &P_no;
    f2k::cuda::ModulationMLP mlp(cfg);
    if (!mlp.ok()) { std::printf("FAIL ctor: %s\n", mlp.last_error()); return false; }

    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> ht(time_dim);
    for (auto& v : ht) v = fp32_to_bf16(dx(rng));

    void *d_t, *d_ws;
    cudaMalloc(&d_t, time_dim * 2);
    cudaMalloc(&d_ws, mlp.workspace_size_bytes());
    cudaMemcpy(d_t, ht.data(), time_dim * 2, cudaMemcpyHostToDevice);

    f2k::cuda::ModulationMLP::Output o{};
    const bool ok = mlp.forward(d_t, o, d_ws, mlp.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL forward: %s\n", mlp.last_error());
               cudaFree(d_t); cudaFree(d_ws); return false; }

    int total_nans = 0;
    double max_overall = 0;
    auto check = [&](const void* p, const char* name) {
        int n; double mx, mn;
        check_vec(p, hidden, n, mx, mn, name);
        total_nans += n;
        max_overall = std::max(max_overall, mx);
        return n == 0;
    };
    bool all_ok = true;
    all_ok &= check(o.img_scale_attn, "img_scale_attn");
    all_ok &= check(o.img_shift_attn, "img_shift_attn");
    all_ok &= check(o.img_gate_attn,  "img_gate_attn");
    all_ok &= check(o.img_scale_mlp,  "img_scale_mlp");
    all_ok &= check(o.img_shift_mlp,  "img_shift_mlp");
    all_ok &= check(o.img_gate_mlp,   "img_gate_mlp");
    all_ok &= check(o.txt_scale_attn, "txt_scale_attn");
    all_ok &= check(o.txt_shift_attn, "txt_shift_attn");
    all_ok &= check(o.txt_gate_attn,  "txt_gate_attn");
    all_ok &= check(o.txt_scale_mlp,  "txt_scale_mlp");
    all_ok &= check(o.txt_shift_mlp,  "txt_shift_mlp");
    all_ok &= check(o.txt_gate_mlp,   "txt_gate_mlp");
    all_ok &= check(o.single_scale,   "single_scale");
    all_ok &= check(o.single_shift,   "single_shift");
    all_ok &= check(o.single_gate,    "single_gate");
    all_ok &= check(o.norm_out_scale, "norm_out_scale");
    all_ok &= check(o.norm_out_shift, "norm_out_shift");

    cudaFree(d_t); cudaFree(d_ws);
    const bool pass = all_ok && max_overall < 500.0;
    std::printf("%s nans=%d max=%.3f\n", pass ? "PASS" : "FAIL", total_nans, max_overall);
    return pass;
}

bool run_real() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("ModulationMLP real-model:  SKIPPED\n");
        return true;
    }
    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string()))
        { std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return false; }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return false; }

    auto v = [](const f2k::TensorView* t) {
        f2k::cuda::PreQuantNVFP4 r{};
        r.packed = t->data; r.scales = t->scales;
        r.N = (int)t->shape[0]; r.K = (int)t->shape[1]; return r;
    };
    auto P_t1 = v(router.globals().time_embed_1);
    auto P_t2 = v(router.globals().time_embed_2);
    auto P_di = v(router.globals().double_mod_img);
    auto P_dt = v(router.globals().double_mod_txt);
    auto P_sg = v(router.globals().single_mod);
    auto P_no = v(router.globals().norm_out);

    const int hidden = P_t2.N;            // 4096
    const int time_dim = P_t1.K;          // 256

    f2k::cuda::ModulationMLP::Config cfg;
    cfg.hidden_dim = hidden; cfg.time_dim = time_dim;
    cfg.W_time_1 = &P_t1; cfg.W_time_2 = &P_t2;
    cfg.W_double_img = &P_di; cfg.W_double_txt = &P_dt;
    cfg.W_single = &P_sg; cfg.W_norm_out = &P_no;
    f2k::cuda::ModulationMLP mlp(cfg);
    if (!mlp.ok()) { std::fprintf(stderr, "real ctor: %s\n", mlp.last_error()); return false; }

    std::mt19937 rng(0xDEADBEEFULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> ht(time_dim);
    for (auto& x : ht) x = fp32_to_bf16(d(rng));

    void *d_t, *d_ws;
    cudaMalloc(&d_t, time_dim * 2);
    cudaMalloc(&d_ws, mlp.workspace_size_bytes());
    cudaMemcpy(d_t, ht.data(), time_dim * 2, cudaMemcpyHostToDevice);

    f2k::cuda::ModulationMLP::Output o{};
    const bool ok = mlp.forward(d_t, o, d_ws, mlp.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::fprintf(stderr, "real forward: %s\n", mlp.last_error());
               cudaFree(d_t); cudaFree(d_ws); return false; }

    int total_nans = 0; double max_overall = 0, sum_mean = 0; int counted = 0;
    auto check = [&](const void* p, const char* name) {
        int n; double mx, mn;
        check_vec(p, hidden, n, mx, mn, name);
        total_nans += n;
        max_overall = std::max(max_overall, mx);
        sum_mean += mn;
        ++counted;
        return n == 0;
    };
    bool all_ok = true;
    all_ok &= check(o.img_scale_attn, "img_scale_attn");
    all_ok &= check(o.img_shift_attn, "img_shift_attn");
    all_ok &= check(o.img_gate_attn,  "img_gate_attn");
    all_ok &= check(o.img_scale_mlp,  "img_scale_mlp");
    all_ok &= check(o.img_shift_mlp,  "img_shift_mlp");
    all_ok &= check(o.img_gate_mlp,   "img_gate_mlp");
    all_ok &= check(o.txt_scale_attn, "txt_scale_attn");
    all_ok &= check(o.txt_shift_attn, "txt_shift_attn");
    all_ok &= check(o.txt_gate_attn,  "txt_gate_attn");
    all_ok &= check(o.txt_scale_mlp,  "txt_scale_mlp");
    all_ok &= check(o.txt_shift_mlp,  "txt_shift_mlp");
    all_ok &= check(o.txt_gate_mlp,   "txt_gate_mlp");
    all_ok &= check(o.single_scale,   "single_scale");
    all_ok &= check(o.single_shift,   "single_shift");
    all_ok &= check(o.single_gate,    "single_gate");
    all_ok &= check(o.norm_out_scale, "norm_out_scale");
    all_ok &= check(o.norm_out_shift, "norm_out_shift");

    cudaFree(d_t); cudaFree(d_ws);
    const double mean_of_means = sum_mean / counted;
    const bool pass = all_ok && max_overall < 500.0;
    std::printf("ModulationMLP real-model: %s nans=%d max=%.3f mean_of_means=%.3f (17 vecs of %d)\n",
                pass ? "PASS" : "FAIL", total_nans, max_overall, mean_of_means, hidden);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_synth();
    ok &= run_real();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
