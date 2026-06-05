// MMDiTStack tests: N-deep zero-gate identity + non-zero smoke.

#include "backend/cuda/mmdit_stack.h"

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

struct PerBlockWeights {
    std::vector<__nv_bfloat16> W_norm1, W_norm2;
    std::vector<__nv_bfloat16> W_q, W_k, W_v, W_proj;
    std::vector<__nv_bfloat16> W_gate, W_up, W_down;
};

PerBlockWeights make_block_weights(int hidden, int ffn, std::mt19937& rng) {
    std::uniform_real_distribution<float> dG(0.9f, 1.1f);
    auto glorot = [&](int fan_in) {
        const float a = std::sqrt(6.0f / fan_in);
        return std::uniform_real_distribution<float>(-a, a);
    };
    PerBlockWeights b;
    b.W_norm1.resize(hidden); for (auto& v : b.W_norm1) v = fp32_to_bf16(dG(rng));
    b.W_norm2.resize(hidden); for (auto& v : b.W_norm2) v = fp32_to_bf16(dG(rng));
    auto fill = [&](std::vector<__nv_bfloat16>& v, size_t n, int fan_in) {
        auto d = glorot(fan_in); v.resize(n);
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
    for (int p = 0; p < seq; ++p)
        for (int i = 0; i < hd; ++i) {
            const float freq = std::pow(theta, -2.0f * i / head_dim);
            cos_t[p * hd + i] = std::cos(p * freq);
            sin_t[p * hd + i] = std::sin(p * freq);
        }
}

struct StackHandles {
    // Host weights + their per-block pointer arrays.
    std::vector<PerBlockWeights> blocks;
    std::vector<const void*> p_W_norm1, p_W_q, p_W_k, p_W_v, p_W_proj;
    std::vector<const void*> p_W_norm2, p_W_gate, p_W_up, p_W_down;
};

StackHandles make_stack_handles(int n_blocks, int hidden, int ffn, std::mt19937& rng) {
    StackHandles h;
    h.blocks.reserve(n_blocks);
    for (int b = 0; b < n_blocks; ++b) {
        h.blocks.push_back(make_block_weights(hidden, ffn, rng));
        h.p_W_norm1.push_back(h.blocks.back().W_norm1.data());
        h.p_W_q   .push_back(h.blocks.back().W_q.data());
        h.p_W_k   .push_back(h.blocks.back().W_k.data());
        h.p_W_v   .push_back(h.blocks.back().W_v.data());
        h.p_W_proj.push_back(h.blocks.back().W_proj.data());
        h.p_W_norm2.push_back(h.blocks.back().W_norm2.data());
        h.p_W_gate.push_back(h.blocks.back().W_gate.data());
        h.p_W_up  .push_back(h.blocks.back().W_up.data());
        h.p_W_down.push_back(h.blocks.back().W_down.data());
    }
    return h;
}

bool run_zero_gate_stack(int n_blocks, int batch, int seq, int n_heads,
                         int head_dim, int ffn_dim) {
    const int hidden     = n_heads * head_dim;
    const int batch_rows = batch * seq;
    std::printf("MMDiTStack zero-gate N=%d b=%d s=%-4d H=%-2d D=%-3d ffn=%-4d  ",
                n_blocks, batch, seq, n_heads, head_dim, ffn_dim);

    std::mt19937 rng(0xBADC0DEULL);
    auto H = make_stack_handles(n_blocks, hidden, ffn_dim, rng);

    f2k::cuda::MMDiTStack::Config cfg;
    cfg.batch = batch; cfg.seq = seq; cfg.n_heads = n_heads;
    cfg.head_dim = head_dim; cfg.ffn_dim = ffn_dim; cfg.num_blocks = n_blocks;
    cfg.W_norm1 = H.p_W_norm1.data();
    cfg.W_q     = H.p_W_q.data();
    cfg.W_k     = H.p_W_k.data();
    cfg.W_v     = H.p_W_v.data();
    cfg.W_proj  = H.p_W_proj.data();
    cfg.W_norm2 = H.p_W_norm2.data();
    cfg.W_gate  = H.p_W_gate.data();
    cfg.W_up    = H.p_W_up.data();
    cfg.W_down  = H.p_W_down.data();

    f2k::cuda::MMDiTStack stack(cfg);
    if (!stack.ok()) { std::printf("FAIL ctor: %s\n", stack.last_error()); return false; }

    // Single shared zero-vector for all blocks' modulations.
    std::vector<__nv_bfloat16> hzero(hidden, fp32_to_bf16(0.0f));
    void* d_zero = nullptr;
    cudaMalloc(&d_zero, hidden * sizeof(__nv_bfloat16));
    cudaMemcpy(d_zero, hzero.data(), hidden * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);

    // Build per-block pointer arrays that all alias d_zero.
    std::vector<const void*> mod_arr(n_blocks, d_zero);

    // x input.
    std::mt19937 rng2(0x12345);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(batch_rows) * hidden);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng2));
    const auto hx_orig = hx;

    std::vector<float> cos_t, sin_t;
    build_rope_tables(seq, head_dim, 10000.0f, cos_t, sin_t);

    void *d_x, *d_cos, *d_sin, *d_ws;
    const size_t x_bytes = hx.size() * sizeof(__nv_bfloat16);
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_cos, cos_t.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws, stack.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float), cudaMemcpyHostToDevice);

    f2k::cuda::MMDiTStack::Modulation m;
    m.scale_attn = mod_arr.data(); m.shift_attn = mod_arr.data(); m.gate_attn = mod_arr.data();
    m.scale_mlp  = mod_arr.data(); m.shift_mlp  = mod_arr.data(); m.gate_mlp  = mod_arr.data();

    const bool ok = stack.forward(d_x, m,
                                  static_cast<const float*>(d_cos),
                                  static_cast<const float*>(d_sin),
                                  d_ws, stack.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) { std::printf("FAIL fwd: %s\n", stack.last_error());
               cudaFree(d_x); cudaFree(d_zero); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
               return false; }

    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_x, x_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_zero); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);

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
}

bool run_smoke_stack(int n_blocks, int batch, int seq, int n_heads,
                     int head_dim, int ffn_dim) {
    const int hidden     = n_heads * head_dim;
    const int batch_rows = batch * seq;
    std::printf("MMDiTStack smoke    N=%d b=%d s=%-4d H=%-2d D=%-3d ffn=%-4d  ",
                n_blocks, batch, seq, n_heads, head_dim, ffn_dim);

    std::mt19937 rng(0xDEFACED1ULL);
    auto H = make_stack_handles(n_blocks, hidden, ffn_dim, rng);

    // Per-block modulation: small non-zero scale/shift, modest gate.
    std::uniform_real_distribution<float> ds(-0.1f, 0.1f);
    std::vector<std::vector<__nv_bfloat16>> mod_storage;
    mod_storage.reserve(n_blocks * 6);
    std::vector<const void*> p_sa, p_sh, p_ga, p_sm, p_sh2, p_gm;
    std::vector<void*> dev_ptrs;
    for (int b = 0; b < n_blocks; ++b) {
        for (int k = 0; k < 6; ++k) {
            std::vector<__nv_bfloat16> v(hidden);
            // gates are slots 2 and 5 → small positive constant + noise
            const bool is_gate = (k == 2 || k == 5);
            for (auto& x : v)
                x = fp32_to_bf16(is_gate ? (0.2f + ds(rng)) : ds(rng));
            mod_storage.push_back(std::move(v));
        }
    }
    // Upload each modulation tensor.
    for (auto& v : mod_storage) {
        void* dptr; cudaMalloc(&dptr, hidden * sizeof(__nv_bfloat16));
        cudaMemcpy(dptr, v.data(), hidden * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        dev_ptrs.push_back(dptr);
    }
    // Build per-slot pointer arrays.
    for (int b = 0; b < n_blocks; ++b) {
        p_sa .push_back(dev_ptrs[b*6 + 0]);
        p_sh .push_back(dev_ptrs[b*6 + 1]);
        p_ga .push_back(dev_ptrs[b*6 + 2]);
        p_sm .push_back(dev_ptrs[b*6 + 3]);
        p_sh2.push_back(dev_ptrs[b*6 + 4]);
        p_gm .push_back(dev_ptrs[b*6 + 5]);
    }

    f2k::cuda::MMDiTStack::Config cfg;
    cfg.batch = batch; cfg.seq = seq; cfg.n_heads = n_heads;
    cfg.head_dim = head_dim; cfg.ffn_dim = ffn_dim; cfg.num_blocks = n_blocks;
    cfg.W_norm1 = H.p_W_norm1.data();
    cfg.W_q     = H.p_W_q.data();
    cfg.W_k     = H.p_W_k.data();
    cfg.W_v     = H.p_W_v.data();
    cfg.W_proj  = H.p_W_proj.data();
    cfg.W_norm2 = H.p_W_norm2.data();
    cfg.W_gate  = H.p_W_gate.data();
    cfg.W_up    = H.p_W_up.data();
    cfg.W_down  = H.p_W_down.data();

    f2k::cuda::MMDiTStack stack(cfg);
    if (!stack.ok()) { std::printf("FAIL ctor: %s\n", stack.last_error()); return false; }

    std::mt19937 rng2(0xABCDULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(static_cast<size_t>(batch_rows) * hidden);
    for (auto& v : hx) v = fp32_to_bf16(dx(rng2));

    std::vector<float> cos_t, sin_t;
    build_rope_tables(seq, head_dim, 10000.0f, cos_t, sin_t);

    void *d_x, *d_cos, *d_sin, *d_ws;
    const size_t x_bytes = hx.size() * sizeof(__nv_bfloat16);
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_cos, cos_t.size() * sizeof(float));
    cudaMalloc(&d_sin, sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws, stack.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), x_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_cos, cos_t.data(), cos_t.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_sin, sin_t.data(), sin_t.size() * sizeof(float), cudaMemcpyHostToDevice);

    f2k::cuda::MMDiTStack::Modulation m;
    m.scale_attn = p_sa.data(); m.shift_attn = p_sh.data(); m.gate_attn = p_ga.data();
    m.scale_mlp  = p_sm.data(); m.shift_mlp  = p_sh2.data(); m.gate_mlp  = p_gm.data();

    const bool ok = stack.forward(d_x, m,
                                  static_cast<const float*>(d_cos),
                                  static_cast<const float*>(d_sin),
                                  d_ws, stack.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(hx.size());
    cudaMemcpy(hy.data(), d_x, x_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
    for (void* p : dev_ptrs) cudaFree(p);
    if (!ok) { std::printf("FAIL fwd: %s\n", stack.last_error()); return false; }

    int n_bad = 0; double max_abs = 0, sum_abs = 0;
    for (auto v : hy) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++n_bad;
        max_abs = std::max(max_abs, (double)std::fabs(f));
        sum_abs += std::fabs(f);
    }
    const double mean_abs = sum_abs / hy.size();
    const bool pass = (n_bad == 0) && (max_abs < 500.0) && (mean_abs > 0.05);
    std::printf("%s nans=%d max=%.3f mean=%.3f\n",
                pass ? "PASS" : "FAIL", n_bad, max_abs, mean_abs);
    return pass;
}

} // namespace

int main() {
    bool ok = true;
    // 4 blocks at modest dims — fast enough to be a sanity test.
    ok &= run_zero_gate_stack(4, 1, 128, 2, 128, 512);
    ok &= run_smoke_stack    (4, 1, 128, 2, 128, 512);
    // 8 blocks at larger dims — closer to a real MMDiT depth.
    ok &= run_zero_gate_stack(8, 1, 256, 8, 128, 2048);
    ok &= run_smoke_stack    (8, 1, 256, 8, 128, 2048);
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
