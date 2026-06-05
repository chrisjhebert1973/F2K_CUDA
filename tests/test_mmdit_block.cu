// MMDiTBlock end-to-end smoke test.
//
// Two scenarios:
//   1. Zero-gate identity: gate_attn = gate_mlp = 0 → block should leave x
//      essentially unchanged. The internal computations still run (modulate,
//      Q/K/V, attention, proj, SwiGLU) but their results are gated to zero
//      and discarded in the residual adds.
//   2. Normal modulation: small non-zero gates, verify output is non-NaN and
//      magnitudes are sane (not blown up).

#include "backend/cuda/mmdit_block.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

struct Buffers {
    std::vector<__nv_bfloat16> W_norm1, W_norm2;
    std::vector<__nv_bfloat16> W_q, W_k, W_v, W_proj;
    std::vector<__nv_bfloat16> W_gate, W_up, W_down;
};

Buffers make_weights(int hidden, int ffn, std::mt19937& rng) {
    // Glorot-uniform: bound = sqrt(6 / fan_in). Keeps activation magnitudes
    // sane when stacked through the block's many linears.
    std::uniform_real_distribution<float> dG(0.9f, 1.1f);
    auto glorot = [&](int fan_in) {
        const float a = std::sqrt(6.0f / fan_in);
        return std::uniform_real_distribution<float>(-a, a);
    };
    Buffers b;
    b.W_norm1.resize(hidden); for (auto& v : b.W_norm1) v = fp32_to_bf16(dG(rng));
    b.W_norm2.resize(hidden); for (auto& v : b.W_norm2) v = fp32_to_bf16(dG(rng));
    auto fill = [&](std::vector<__nv_bfloat16>& v, size_t n, int fan_in) {
        auto d = glorot(fan_in);
        v.resize(n);
        for (auto& x : v) x = fp32_to_bf16(d(rng));
    };
    fill(b.W_q,    static_cast<size_t>(hidden) * hidden, hidden);
    fill(b.W_k,    static_cast<size_t>(hidden) * hidden, hidden);
    fill(b.W_v,    static_cast<size_t>(hidden) * hidden, hidden);
    fill(b.W_proj, static_cast<size_t>(hidden) * hidden, hidden);
    fill(b.W_gate, static_cast<size_t>(ffn) * hidden,    hidden);
    fill(b.W_up,   static_cast<size_t>(ffn) * hidden,    hidden);
    fill(b.W_down, static_cast<size_t>(hidden) * ffn,    ffn);
    return b;
}

void build_rope_tables(int seq, int head_dim, float theta,
                       std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int hd = head_dim / 2;
    cos_t.assign(static_cast<size_t>(seq) * hd, 0.0f);
    sin_t.assign(static_cast<size_t>(seq) * hd, 0.0f);
    for (int p = 0; p < seq; ++p) {
        for (int i = 0; i < hd; ++i) {
            const float freq = std::pow(theta, -2.0f * i / head_dim);
            cos_t[p * hd + i] = std::cos(p * freq);
            sin_t[p * hd + i] = std::sin(p * freq);
        }
    }
}

bool run_zero_gate(int batch, int seq, int n_heads, int head_dim, int ffn_dim) {
    const int hidden     = n_heads * head_dim;
    const int batch_rows = batch * seq;
    std::printf("MMDiTBlock zero-gate b=%d s=%-4d H=%-2d D=%-3d ffn=%-4d  ",
                batch, seq, n_heads, head_dim, ffn_dim);

    std::mt19937 rng(0xCAFEBABEULL);
    Buffers W = make_weights(hidden, ffn_dim, rng);

    // Modulation: scale=0, shift=0, gate=0 → residuals contribute nothing.
    std::vector<__nv_bfloat16> mod_zero(hidden, fp32_to_bf16(0.0f));

    // x input.
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(batch_rows) * hidden);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));
    const std::vector<__nv_bfloat16> hx_orig = hx;

    // RoPE tables.
    std::vector<float> cos_t, sin_t;
    build_rope_tables(seq, head_dim, 10000.0f, cos_t, sin_t);

    f2k::cuda::MMDiTBlock::Config cfg;
    cfg.batch = batch; cfg.seq = seq;
    cfg.n_heads = n_heads; cfg.head_dim = head_dim; cfg.ffn_dim = ffn_dim;
    cfg.W_norm1 = W.W_norm1.data();
    cfg.W_q     = W.W_q.data();
    cfg.W_k     = W.W_k.data();
    cfg.W_v     = W.W_v.data();
    cfg.W_proj  = W.W_proj.data();
    cfg.W_norm2 = W.W_norm2.data();
    cfg.W_gate  = W.W_gate.data();
    cfg.W_up    = W.W_up.data();
    cfg.W_down  = W.W_down.data();

    f2k::cuda::MMDiTBlock block(cfg);
    if (!block.ok()) { std::printf("FAIL ctor: %s\n", block.last_error()); return false; }

    void *d_x, *d_mod, *d_cos, *d_sin, *d_ws;
    const size_t x_bytes = hx.size() * sizeof(__nv_bfloat16);
    cudaMalloc(&d_x,   x_bytes);
    cudaMalloc(&d_mod, hidden * sizeof(__nv_bfloat16));
    cudaMalloc(&d_cos, cos_t.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws,  block.workspace_size_bytes());
    cudaMemcpy(d_x,   hx.data(),    x_bytes,                                   cudaMemcpyHostToDevice);
    cudaMemcpy(d_mod, mod_zero.data(), hidden * sizeof(__nv_bfloat16),         cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float),              cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float),              cudaMemcpyHostToDevice);

    f2k::cuda::MMDiTBlock::Modulation m;
    m.scale_attn = d_mod; m.shift_attn = d_mod; m.gate_attn = d_mod;
    m.scale_mlp  = d_mod; m.shift_mlp  = d_mod; m.gate_mlp  = d_mod;
    const bool ok = block.forward(d_x, m,
                                  static_cast<const float*>(d_cos),
                                  static_cast<const float*>(d_sin),
                                  d_ws, block.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL fwd: %s\n", block.last_error());
               cudaFree(d_x); cudaFree(d_mod); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
               return false; }

    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_x, x_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_mod); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);

    double max_e = 0, sum_e = 0;
    for (size_t i = 0; i < hy.size(); ++i) {
        const double e = std::fabs(bf16_to_fp32(hy[i]) - bf16_to_fp32(hx_orig[i]));
        max_e = std::max(max_e, e);
        sum_e += e;
    }
    const double mean = sum_e / static_cast<double>(hy.size());
    // BF16 noise from "+= 0" path: should be tiny.
    const bool pass = max_e < 0.02 && mean < 0.001;
    std::printf("%s max=%.5f mean=%.6f\n", pass ? "PASS" : "FAIL", max_e, mean);
    return pass;
}

bool run_smoke_nonzero(int batch, int seq, int n_heads, int head_dim, int ffn_dim) {
    const int hidden     = n_heads * head_dim;
    const int batch_rows = batch * seq;
    std::printf("MMDiTBlock smoke    b=%d s=%-4d H=%-2d D=%-3d ffn=%-4d  ",
                batch, seq, n_heads, head_dim, ffn_dim);

    std::mt19937 rng(0xDEADBEEFULL);
    Buffers W = make_weights(hidden, ffn_dim, rng);

    // Small but non-zero modulation.
    std::uniform_real_distribution<float> ds(-0.1f, 0.1f);
    std::vector<__nv_bfloat16> scale_attn(hidden), shift_attn(hidden), gate_attn(hidden);
    std::vector<__nv_bfloat16> scale_mlp (hidden), shift_mlp (hidden), gate_mlp (hidden);
    for (auto& v : scale_attn) v = fp32_to_bf16(ds(rng));
    for (auto& v : shift_attn) v = fp32_to_bf16(ds(rng));
    for (auto& v : gate_attn)  v = fp32_to_bf16(0.3f + ds(rng));
    for (auto& v : scale_mlp)  v = fp32_to_bf16(ds(rng));
    for (auto& v : shift_mlp)  v = fp32_to_bf16(ds(rng));
    for (auto& v : gate_mlp)   v = fp32_to_bf16(0.3f + ds(rng));

    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(batch_rows) * hidden);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng));
    const std::vector<__nv_bfloat16> hx_orig = hx;

    std::vector<float> cos_t, sin_t;
    build_rope_tables(seq, head_dim, 10000.0f, cos_t, sin_t);

    f2k::cuda::MMDiTBlock::Config cfg;
    cfg.batch = batch; cfg.seq = seq;
    cfg.n_heads = n_heads; cfg.head_dim = head_dim; cfg.ffn_dim = ffn_dim;
    cfg.W_norm1 = W.W_norm1.data();
    cfg.W_q     = W.W_q.data();
    cfg.W_k     = W.W_k.data();
    cfg.W_v     = W.W_v.data();
    cfg.W_proj  = W.W_proj.data();
    cfg.W_norm2 = W.W_norm2.data();
    cfg.W_gate  = W.W_gate.data();
    cfg.W_up    = W.W_up.data();
    cfg.W_down  = W.W_down.data();

    f2k::cuda::MMDiTBlock block(cfg);
    if (!block.ok()) { std::printf("FAIL ctor: %s\n", block.last_error()); return false; }

    void *d_x, *d_sa, *d_sh, *d_ga, *d_sm, *d_sh2, *d_gm, *d_cos, *d_sin, *d_ws;
    const size_t x_bytes  = hx.size() * sizeof(__nv_bfloat16);
    const size_t mb       = hidden * sizeof(__nv_bfloat16);
    cudaMalloc(&d_x,   x_bytes);
    cudaMalloc(&d_sa,  mb); cudaMalloc(&d_sh,  mb); cudaMalloc(&d_ga,  mb);
    cudaMalloc(&d_sm,  mb); cudaMalloc(&d_sh2, mb); cudaMalloc(&d_gm,  mb);
    cudaMalloc(&d_cos, cos_t.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws,  block.workspace_size_bytes());
    cudaMemcpy(d_x,   hx.data(),         x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sa,  scale_attn.data(), mb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh,  shift_attn.data(), mb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ga,  gate_attn.data(),  mb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sm,  scale_mlp.data(),  mb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_sh2, shift_mlp.data(),  mb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_gm,  gate_mlp.data(),   mb, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float), cudaMemcpyHostToDevice);

    f2k::cuda::MMDiTBlock::Modulation m;
    m.scale_attn = d_sa; m.shift_attn = d_sh; m.gate_attn = d_ga;
    m.scale_mlp  = d_sm; m.shift_mlp  = d_sh2; m.gate_mlp  = d_gm;

    const bool ok = block.forward(d_x, m,
                                  static_cast<const float*>(d_cos),
                                  static_cast<const float*>(d_sin),
                                  d_ws, block.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_x, x_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_sa); cudaFree(d_sh); cudaFree(d_ga);
    cudaFree(d_sm); cudaFree(d_sh2); cudaFree(d_gm);
    cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
    if (!ok) { std::printf("FAIL fwd: %s\n", block.last_error()); return false; }

    // Sanity: no NaN/Inf, output isn't identically zero.
    int n_bad = 0;
    double max_abs = 0;
    double sum_abs = 0;
    for (auto v : hy) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++n_bad;
        max_abs = std::max(max_abs, (double)std::fabs(f));
        sum_abs += std::fabs(f);
    }
    const double mean_abs = sum_abs / hy.size();
    // Generous range: stacked random linears can blow up moderately. The real
    // correctness signal is the zero-gate identity case above.
    const bool pass = (n_bad == 0) && (max_abs < 100.0) && (mean_abs > 0.05);
    std::printf("%s nans=%d max=%.3f mean=%.3f\n",
                pass ? "PASS" : "FAIL", n_bad, max_abs, mean_abs);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    // Modest size: hidden_dim = 256 (2 heads * 128), ffn = 512, seq = 128, batch = 1.
    ok &= run_zero_gate    (1, 128, 2, 128, 512);
    ok &= run_smoke_nonzero(1, 128, 2, 128, 512);
    // FLUX-ish (smaller seq): hidden = 1024 (8*128), ffn = 2048, seq = 256.
    ok &= run_zero_gate    (1, 256, 8, 128, 2048);
    ok &= run_smoke_nonzero(1, 256, 8, 128, 2048);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
