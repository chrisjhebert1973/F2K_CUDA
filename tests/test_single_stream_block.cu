// SingleStreamBlock test.
//
//   1. Synthetic zero-gate identity: gate_mod = 0 → block leaves x unchanged.
//   2. Synthetic non-zero smoke: random weights, sane magnitudes, no NaN.
//   3. Real-model smoke: load single_transformer_blocks.0 from F2K, run forward.

#include "backend/cuda/single_stream_block.h"
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
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    int N, K;
};

PreQ quantize_for_storage(const std::vector<__nv_bfloat16>& W_bf16, int N, int K) {
    PreQ p;
    p.N = N; p.K = K;
    p.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
    p.scales.assign(static_cast<size_t>(N) * (K / 16), 0);
    constexpr float E2M1_MAX = 6.0f;
    const int n_kblocks = K / 16;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            float absmax = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float v = bf16_to_fp32(W_bf16[static_cast<size_t>(n) * K + kb * 16 + i]);
                absmax = std::max(absmax, std::fabs(v));
            }
            const float scale = (absmax > 1e-30f) ? (absmax / E2M1_MAX) : 1.0f;
            cutlass::float_ue4m3_t sf(scale);
            p.scales[static_cast<size_t>(n) * n_kblocks + kb] = static_cast<uint8_t>(sf.raw());
            for (int i = 0; i < 16; ++i) {
                const int k = kb * 16 + i;
                const float v = bf16_to_fp32(W_bf16[static_cast<size_t>(n) * K + k]);
                cutlass::float_e2m1_t e2m1(v / scale);
                const uint8_t nibble = static_cast<uint8_t>(e2m1.raw()) & 0xFu;
                const size_t byte_idx = static_cast<size_t>(n) * (K / 2) + (k >> 1);
                if ((k & 1) == 0) p.packed[byte_idx] = (p.packed[byte_idx] & 0xF0u) | nibble;
                else              p.packed[byte_idx] = (p.packed[byte_idx] & 0x0Fu) | static_cast<uint8_t>(nibble << 4);
            }
        }
    }
    return p;
}

void build_rope_tables(int seq, int head_dim, float theta,
                       std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int hd = head_dim / 2;
    cos_t.assign(static_cast<size_t>(seq) * hd, 0.0f);
    sin_t.assign(static_cast<size_t>(seq) * hd, 0.0f);
    for (int p = 0; p < seq; ++p)
        for (int i = 0; i < hd; ++i) {
            const float freq = std::pow(theta, -2.0f * i / head_dim);
            cos_t[p * hd + i] = std::cos(p * freq);
            sin_t[p * hd + i] = std::sin(p * freq);
        }
}

struct SynthW {
    PreQ qkv_mlp;
    PreQ out;
    std::vector<__nv_bfloat16> norm_q, norm_k;
};

SynthW make_synth_weights(int hidden, int ffn, int head_dim, std::mt19937& rng) {
    auto glorot = [&](int fan_in) {
        const float a = std::sqrt(6.0f / fan_in);
        return std::uniform_real_distribution<float>(-a, a);
    };
    const int fused_out = 3 * hidden + 2 * ffn;
    const int concat_in = hidden + ffn;

    std::vector<__nv_bfloat16> W_qkv(static_cast<size_t>(fused_out) * hidden);
    std::vector<__nv_bfloat16> W_out(static_cast<size_t>(hidden) * concat_in);
    {
        auto d = glorot(hidden);
        for (auto& v : W_qkv) v = fp32_to_bf16(d(rng));
    }
    {
        auto d = glorot(concat_in);
        for (auto& v : W_out) v = fp32_to_bf16(d(rng));
    }

    SynthW w;
    w.qkv_mlp = quantize_for_storage(W_qkv, fused_out, hidden);
    w.out     = quantize_for_storage(W_out, hidden, concat_in);
    w.norm_q.resize(head_dim);
    w.norm_k.resize(head_dim);
    std::uniform_real_distribution<float> dn(0.9f, 1.1f);
    for (auto& v : w.norm_q) v = fp32_to_bf16(dn(rng));
    for (auto& v : w.norm_k) v = fp32_to_bf16(dn(rng));
    return w;
}

bool run_synth(bool zero_gate) {
    const int B = 1, S = 128, H = 8, D = 128;
    const int hidden = H * D;
    const int ffn = 3 * hidden;            // mlp_ratio = 3 (FLUX.2-klein default)
    const int rows = B * S;
    std::printf("SingleStreamBlock %-12s b=%d s=%-4d H=%d D=%d hidden=%d ffn=%d  ",
                zero_gate ? "zero-gate" : "smoke", B, S, H, D, hidden, ffn);

    std::mt19937 rng(0x51591E81ULL);
    SynthW w = make_synth_weights(hidden, ffn, D, rng);

    f2k::cuda::PreQuantNVFP4 pq_qkv{}, pq_out{};
    pq_qkv.packed = w.qkv_mlp.packed.data();
    pq_qkv.scales = w.qkv_mlp.scales.data();
    pq_qkv.N = w.qkv_mlp.N; pq_qkv.K = w.qkv_mlp.K;
    pq_out.packed = w.out.packed.data();
    pq_out.scales = w.out.scales.data();
    pq_out.N = w.out.N; pq_out.K = w.out.K;

    f2k::cuda::SingleStreamBlock::Config cfg;
    cfg.batch = B; cfg.seq = S; cfg.n_heads = H; cfg.head_dim = D; cfg.ffn_dim = ffn;
    cfg.W_qkv_mlp_proj = &pq_qkv;
    cfg.W_out          = &pq_out;
    cfg.norm_q_bf16    = w.norm_q.data();
    cfg.norm_k_bf16    = w.norm_k.data();
    f2k::cuda::SingleStreamBlock blk(cfg);
    if (!blk.ok()) { std::printf("FAIL ctor: %s\n", blk.last_error()); return false; }

    // Modulation: zero shift, scale=0, gate=0 (identity) OR small non-zero.
    std::vector<__nv_bfloat16> hscale(hidden), hshift(hidden), hgate(hidden);
    std::uniform_real_distribution<float> dm(-0.1f, 0.1f);
    for (auto& v : hscale) v = fp32_to_bf16(zero_gate ? 0.0f : dm(rng));
    for (auto& v : hshift) v = fp32_to_bf16(zero_gate ? 0.0f : dm(rng));
    for (auto& v : hgate)  v = fp32_to_bf16(zero_gate ? 0.0f : (0.2f + dm(rng)));

    // x.
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(rows) * hidden);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));
    const auto hx_orig = hx;

    std::vector<float> cos_t, sin_t;
    build_rope_tables(S, D, 2000.0f, cos_t, sin_t);

    void *d_x, *d_sc, *d_sh, *d_ga, *d_cos, *d_sin, *d_ws;
    const size_t xbytes = hx.size() * sizeof(__nv_bfloat16);
    cudaMalloc(&d_x,   xbytes);
    cudaMalloc(&d_sc,  hidden * 2);
    cudaMalloc(&d_sh,  hidden * 2);
    cudaMalloc(&d_ga,  hidden * 2);
    cudaMalloc(&d_cos, cos_t.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws,  blk.workspace_size_bytes());
    cudaMemcpy(d_x,   hx.data(),     xbytes,       cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc,  hscale.data(), hidden * 2,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh,  hshift.data(), hidden * 2,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_ga,  hgate.data(),  hidden * 2,   cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t.data(), cos_t.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t.data(), sin_t.size()*sizeof(float), cudaMemcpyHostToDevice);

    f2k::cuda::SingleStreamBlock::Modulation m;
    m.scale = d_sc; m.shift = d_sh; m.gate = d_ga;
    const bool ok = blk.forward(d_x, m,
                                 static_cast<const float*>(d_cos),
                                 static_cast<const float*>(d_sin),
                                 d_ws, blk.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_x, xbytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_sc); cudaFree(d_sh); cudaFree(d_ga);
    cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL forward: %s\n", blk.last_error()); return false; }

    if (zero_gate) {
        double max_e = 0, sum_e = 0;
        for (size_t i = 0; i < hy.size(); ++i) {
            const double e = std::fabs(bf16_to_fp32(hy[i]) - bf16_to_fp32(hx_orig[i]));
            max_e = std::max(max_e, e);
            sum_e += e;
        }
        const double mean = sum_e / hy.size();
        const bool pass = max_e < 0.02 && mean < 0.001;
        std::printf("%s max=%.5f mean=%.6f\n", pass ? "PASS" : "FAIL", max_e, mean);
        return pass;
    } else {
        int n_bad = 0; double max_abs = 0, sum_abs = 0;
        for (auto v : hy) {
            const float f = bf16_to_fp32(v);
            if (!std::isfinite(f)) ++n_bad;
            max_abs = std::max(max_abs, (double)std::fabs(f));
            sum_abs += std::fabs(f);
        }
        const double mean = sum_abs / hy.size();
        const bool pass = (n_bad == 0) && (max_abs < 200.0) && (mean > 0.05);
        std::printf("%s nans=%d max=%.3f mean=%.3f\n", pass ? "PASS" : "FAIL", n_bad, max_abs, mean);
        return pass;
    }
}

bool real_model_smoke() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("SingleStreamBlock real-model:  SKIPPED\n");
        return true;
    }
    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string())) {
        std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return false;
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return false; }

    const auto& sb0 = router.single_blocks()[0];
    f2k::cuda::PreQuantNVFP4 pq_qkv{}, pq_out{};
    pq_qkv.packed = sb0.to_qkv_mlp_proj->data;
    pq_qkv.scales = sb0.to_qkv_mlp_proj->scales;
    pq_qkv.N = static_cast<int>(sb0.to_qkv_mlp_proj->shape[0]);
    pq_qkv.K = static_cast<int>(sb0.to_qkv_mlp_proj->shape[1]);
    pq_out.packed = sb0.to_out->data;
    pq_out.scales = sb0.to_out->scales;
    pq_out.N = static_cast<int>(sb0.to_out->shape[0]);
    pq_out.K = static_cast<int>(sb0.to_out->shape[1]);

    // FLUX.2-klein: n_heads=32, head_dim=128, ffn=12288.
    // Use a small seq for smoke (S must give batch_rows % 128 == 0).
    const int B = 1, S = 128, H = 32, D = 128;
    const int hidden = H * D;
    const int ffn = 12288;

    f2k::cuda::SingleStreamBlock::Config cfg;
    cfg.batch = B; cfg.seq = S; cfg.n_heads = H; cfg.head_dim = D; cfg.ffn_dim = ffn;
    cfg.W_qkv_mlp_proj = &pq_qkv;
    cfg.W_out          = &pq_out;
    cfg.norm_q_bf16    = sb0.norm_q->data;
    cfg.norm_k_bf16    = sb0.norm_k->data;
    f2k::cuda::SingleStreamBlock blk(cfg);
    if (!blk.ok()) { std::fprintf(stderr, "real ctor: %s\n", blk.last_error()); return false; }

    std::mt19937 rng(0x9E41AD0EULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::uniform_real_distribution<float> dm(-0.1f, 0.1f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(B*S) * hidden);
    std::vector<__nv_bfloat16> hsc(hidden), hsh(hidden), hga(hidden);
    for (auto& v : hx)  v = fp32_to_bf16(dx(rng));
    for (auto& v : hsc) v = fp32_to_bf16(dm(rng));
    for (auto& v : hsh) v = fp32_to_bf16(dm(rng));
    for (auto& v : hga) v = fp32_to_bf16(0.2f + dm(rng));

    std::vector<float> cos_t, sin_t;
    build_rope_tables(S, D, 2000.0f, cos_t, sin_t);

    void *d_x, *d_sc, *d_sh, *d_ga, *d_cos, *d_sin, *d_ws;
    const size_t xb = hx.size() * 2;
    cudaMalloc(&d_x,   xb);
    cudaMalloc(&d_sc,  hidden * 2);
    cudaMalloc(&d_sh,  hidden * 2);
    cudaMalloc(&d_ga,  hidden * 2);
    cudaMalloc(&d_cos, cos_t.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws,  blk.workspace_size_bytes());
    cudaMemcpy(d_x,   hx.data(),  xb,         cudaMemcpyHostToDevice);
    cudaMemcpy(d_sc,  hsc.data(), hidden * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh,  hsh.data(), hidden * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ga,  hga.data(), hidden * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t.data(), cos_t.size()*sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t.data(), sin_t.size()*sizeof(float), cudaMemcpyHostToDevice);

    f2k::cuda::SingleStreamBlock::Modulation m;
    m.scale = d_sc; m.shift = d_sh; m.gate = d_ga;
    const bool ok = blk.forward(d_x, m,
                                 static_cast<const float*>(d_cos),
                                 static_cast<const float*>(d_sin),
                                 d_ws, blk.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_x, xb, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_sc); cudaFree(d_sh); cudaFree(d_ga);
    cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
    if (!ok) { std::fprintf(stderr, "real forward: %s\n", blk.last_error()); return false; }

    int n_bad = 0; double max_abs = 0, sum_abs = 0;
    for (auto v : hy) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++n_bad;
        max_abs = std::max(max_abs, (double)std::fabs(f));
        sum_abs += std::fabs(f);
    }
    const double mean = sum_abs / hy.size();
    const bool pass = n_bad == 0 && max_abs < 200.0 && mean > 0.01;
    std::printf("SingleStreamBlock real-model: %s nans=%d max=%.3f mean=%.3f\n",
                pass ? "PASS" : "FAIL", n_bad, max_abs, mean);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    ok &= run_synth(/*zero_gate=*/true);
    ok &= run_synth(/*zero_gate=*/false);
    ok &= real_model_smoke();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
