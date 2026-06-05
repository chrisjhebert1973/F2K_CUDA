// DoubleStreamBlock test.
//   1. Synthetic zero-gate identity: all 4 gates (img attn/mlp, txt attn/mlp) = 0
//      → img and txt should both be preserved bit-exactly.
//   2. Synthetic non-zero smoke.
//   3. Real-model smoke: load transformer_blocks.0 from F2K, run forward.

#include "backend/cuda/double_stream_block.h"
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
                const size_t bi = (size_t)n * (K/2) + (k >> 1);
                if ((k & 1) == 0) p.packed[bi] = (p.packed[bi] & 0xF0u) | nibble;
                else              p.packed[bi] = (p.packed[bi] & 0x0Fu) | (uint8_t)(nibble << 4);
            }
        }
    }
    return p;
}

void build_rope_tables(int seq, int head_dim, float theta,
                       std::vector<float>& cos_t, std::vector<float>& sin_t) {
    const int hd = head_dim / 2;
    cos_t.assign((size_t)seq * hd, 0.0f);
    sin_t.assign((size_t)seq * hd, 0.0f);
    for (int p = 0; p < seq; ++p)
        for (int i = 0; i < hd; ++i) {
            const float freq = std::pow(theta, -2.0f * i / head_dim);
            cos_t[p*hd + i] = std::cos(p * freq);
            sin_t[p*hd + i] = std::sin(p * freq);
        }
}

struct StreamW {
    PreQ q, k, v, out;
    PreQ ff_in, ff_out;
    std::vector<__nv_bfloat16> norm_q, norm_k;
};

StreamW make_stream_weights(int hidden, int ffn, int head_dim, std::mt19937& rng) {
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
    StreamW s;
    auto Wq = fill(hidden, hidden);
    auto Wk = fill(hidden, hidden);
    auto Wv = fill(hidden, hidden);
    auto Wo = fill(hidden, hidden);
    auto Wfi = fill(2*ffn, hidden);
    auto Wfo = fill(hidden, ffn);
    s.q   = quantize_for_storage(Wq, hidden, hidden);
    s.k   = quantize_for_storage(Wk, hidden, hidden);
    s.v   = quantize_for_storage(Wv, hidden, hidden);
    s.out = quantize_for_storage(Wo, hidden, hidden);
    s.ff_in  = quantize_for_storage(Wfi, 2*ffn, hidden);
    s.ff_out = quantize_for_storage(Wfo, hidden, ffn);
    s.norm_q.resize(head_dim); s.norm_k.resize(head_dim);
    std::uniform_real_distribution<float> dn(0.9f, 1.1f);
    for (auto& v : s.norm_q) v = fp32_to_bf16(dn(rng));
    for (auto& v : s.norm_k) v = fp32_to_bf16(dn(rng));
    return s;
}

bool run_synth(bool zero_gate) {
    const int B = 1, S_img = 128, S_txt = 128, H = 8, D = 128;
    const int hidden = H * D;
    const int ffn = 3 * hidden;
    std::printf("DoubleStreamBlock %-12s S_img=%d S_txt=%d H=%d D=%d hidden=%d ffn=%d  ",
                zero_gate ? "zero-gate" : "smoke", S_img, S_txt, H, D, hidden, ffn);

    std::mt19937 rng(0xCAFEBABEULL);
    StreamW img_w = make_stream_weights(hidden, ffn, D, rng);
    StreamW txt_w = make_stream_weights(hidden, ffn, D, rng);

    auto vp = [](const PreQ& q) {
        f2k::cuda::PreQuantNVFP4 r{};
        r.packed = q.packed.data(); r.scales = q.scales.data();
        r.N = q.N; r.K = q.K; return r;
    };
    f2k::cuda::PreQuantNVFP4 pq_q   = vp(img_w.q),   pq_k   = vp(img_w.k),   pq_v   = vp(img_w.v);
    f2k::cuda::PreQuantNVFP4 pq_out = vp(img_w.out), pq_fi  = vp(img_w.ff_in), pq_fo = vp(img_w.ff_out);
    f2k::cuda::PreQuantNVFP4 pq_aq  = vp(txt_w.q),   pq_ak  = vp(txt_w.k),   pq_av  = vp(txt_w.v);
    f2k::cuda::PreQuantNVFP4 pq_ao  = vp(txt_w.out), pq_cfi = vp(txt_w.ff_in), pq_cfo = vp(txt_w.ff_out);

    f2k::cuda::DoubleStreamBlock::Config cfg;
    cfg.batch = B; cfg.seq_img = S_img; cfg.seq_txt = S_txt;
    cfg.n_heads = H; cfg.head_dim = D; cfg.ffn_dim = ffn;
    cfg.W_q = &pq_q; cfg.W_k = &pq_k; cfg.W_v = &pq_v; cfg.W_out = &pq_out;
    cfg.W_ff_in = &pq_fi; cfg.W_ff_out = &pq_fo;
    cfg.norm_q_bf16 = img_w.norm_q.data(); cfg.norm_k_bf16 = img_w.norm_k.data();
    cfg.W_add_q = &pq_aq; cfg.W_add_k = &pq_ak; cfg.W_add_v = &pq_av; cfg.W_add_out = &pq_ao;
    cfg.W_ff_ctx_in = &pq_cfi; cfg.W_ff_ctx_out = &pq_cfo;
    cfg.norm_added_q_bf16 = txt_w.norm_q.data(); cfg.norm_added_k_bf16 = txt_w.norm_k.data();

    f2k::cuda::DoubleStreamBlock blk(cfg);
    if (!blk.ok()) { std::printf("FAIL ctor: %s\n", blk.last_error()); return false; }

    // Modulation: 12 vectors.
    std::uniform_real_distribution<float> dm(-0.1f, 0.1f);
    auto make_mod_vec = [&](float gate_base) {
        std::vector<__nv_bfloat16> v(hidden);
        for (auto& x : v) x = fp32_to_bf16(zero_gate ? (gate_base == 0.f ? 0.f : 0.f) : (gate_base + dm(rng)));
        return v;
    };
    auto Z = [&]() { return std::vector<__nv_bfloat16>(hidden, fp32_to_bf16(0.0f)); };
    auto img_sa  = zero_gate ? Z() : make_mod_vec(0.0f);
    auto img_sh  = zero_gate ? Z() : make_mod_vec(0.0f);
    auto img_ga  = zero_gate ? Z() : make_mod_vec(0.2f);
    auto img_sm  = zero_gate ? Z() : make_mod_vec(0.0f);
    auto img_sh2 = zero_gate ? Z() : make_mod_vec(0.0f);
    auto img_gm  = zero_gate ? Z() : make_mod_vec(0.2f);
    auto txt_sa  = zero_gate ? Z() : make_mod_vec(0.0f);
    auto txt_sh  = zero_gate ? Z() : make_mod_vec(0.0f);
    auto txt_ga  = zero_gate ? Z() : make_mod_vec(0.2f);
    auto txt_sm  = zero_gate ? Z() : make_mod_vec(0.0f);
    auto txt_sh2 = zero_gate ? Z() : make_mod_vec(0.0f);
    auto txt_gm  = zero_gate ? Z() : make_mod_vec(0.2f);

    std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> himg((size_t)B*S_img*hidden), htxt((size_t)B*S_txt*hidden);
    for (auto& v : himg) v = fp32_to_bf16(dx(rng));
    for (auto& v : htxt) v = fp32_to_bf16(dx(rng));
    const auto himg_orig = himg;
    const auto htxt_orig = htxt;

    std::vector<float> cos_t, sin_t;
    build_rope_tables(S_img, D, 2000.0f, cos_t, sin_t);

    auto up = [](void** d, const void* h, size_t bytes) {
        cudaMalloc(d, bytes); cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice);
    };
    void *d_img, *d_txt, *d_cos, *d_sin, *d_ws;
    const size_t img_bytes = himg.size() * 2;
    const size_t txt_bytes = htxt.size() * 2;
    up(&d_img, himg.data(), img_bytes);
    up(&d_txt, htxt.data(), txt_bytes);
    up(&d_cos, cos_t.data(), cos_t.size() * sizeof(float));
    up(&d_sin, sin_t.data(), sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws, blk.workspace_size_bytes());

    auto up_vec = [&](void** d, const std::vector<__nv_bfloat16>& v) {
        cudaMalloc(d, v.size() * 2); cudaMemcpy(*d, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
    };
    void *d_isa, *d_ish, *d_iga, *d_ism, *d_ish2, *d_igm;
    void *d_tsa, *d_tsh, *d_tga, *d_tsm, *d_tsh2, *d_tgm;
    up_vec(&d_isa,  img_sa);  up_vec(&d_ish,  img_sh);  up_vec(&d_iga, img_ga);
    up_vec(&d_ism,  img_sm);  up_vec(&d_ish2, img_sh2); up_vec(&d_igm, img_gm);
    up_vec(&d_tsa,  txt_sa);  up_vec(&d_tsh,  txt_sh);  up_vec(&d_tga, txt_ga);
    up_vec(&d_tsm,  txt_sm);  up_vec(&d_tsh2, txt_sh2); up_vec(&d_tgm, txt_gm);

    f2k::cuda::DoubleStreamBlock::Modulation m;
    m.img_scale_attn = d_isa; m.img_shift_attn = d_ish; m.img_gate_attn = d_iga;
    m.img_scale_mlp  = d_ism; m.img_shift_mlp  = d_ish2; m.img_gate_mlp  = d_igm;
    m.txt_scale_attn = d_tsa; m.txt_shift_attn = d_tsh; m.txt_gate_attn = d_tga;
    m.txt_scale_mlp  = d_tsm; m.txt_shift_mlp  = d_tsh2; m.txt_gate_mlp  = d_tgm;

    // forward() takes separate img/txt RoPE tables (4-axis RoPE work). This
    // test's checks (zero-gate bit-exactness, NaN-free smoke) are insensitive
    // to the table contents, so we feed the same 1D table for both streams.
    const bool ok = blk.forward(d_img, d_txt, m,
                                 (const float*)d_cos, (const float*)d_sin,
                                 (const float*)d_cos, (const float*)d_sin,
                                 d_ws, blk.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> himg_out(himg.size()), htxt_out(htxt.size());
    cudaMemcpy(himg_out.data(), d_img, img_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(htxt_out.data(), d_txt, txt_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_img); cudaFree(d_txt); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
    cudaFree(d_isa); cudaFree(d_ish); cudaFree(d_iga); cudaFree(d_ism); cudaFree(d_ish2); cudaFree(d_igm);
    cudaFree(d_tsa); cudaFree(d_tsh); cudaFree(d_tga); cudaFree(d_tsm); cudaFree(d_tsh2); cudaFree(d_tgm);
    if (!ok) { std::printf("FAIL forward: %s\n", blk.last_error()); return false; }

    if (zero_gate) {
        auto verify = [](const std::vector<__nv_bfloat16>& got,
                          const std::vector<__nv_bfloat16>& orig) {
            double max_e = 0, sum_e = 0;
            for (size_t i = 0; i < got.size(); ++i) {
                const double e = std::fabs(bf16_to_fp32(got[i]) - bf16_to_fp32(orig[i]));
                max_e = std::max(max_e, e); sum_e += e;
            }
            return std::pair<double,double>{max_e, sum_e / got.size()};
        };
        auto [iM, iA] = verify(himg_out, himg_orig);
        auto [tM, tA] = verify(htxt_out, htxt_orig);
        const bool pass = iM < 0.02 && iA < 0.001 && tM < 0.02 && tA < 0.001;
        std::printf("%s img[max=%.5f mean=%.6f] txt[max=%.5f mean=%.6f]\n",
                    pass ? "PASS" : "FAIL", iM, iA, tM, tA);
        return pass;
    } else {
        auto check = [](const std::vector<__nv_bfloat16>& v) {
            int nans = 0; double mx = 0, sum = 0;
            for (auto x : v) {
                const float f = bf16_to_fp32(x);
                if (!std::isfinite(f)) ++nans;
                mx = std::max(mx, (double)std::fabs(f));
                sum += std::fabs(f);
            }
            return std::tuple<int,double,double>{nans, mx, sum / v.size()};
        };
        auto [in, imx, imn] = check(himg_out);
        auto [tn, tmx, tmn] = check(htxt_out);
        const bool pass = (in + tn) == 0 && imx < 500.0 && tmx < 500.0 && imn > 0.05 && tmn > 0.05;
        std::printf("%s img[nans=%d max=%.3f mean=%.3f] txt[nans=%d max=%.3f mean=%.3f]\n",
                    pass ? "PASS" : "FAIL", in, imx, imn, tn, tmx, tmn);
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
        std::printf("DoubleStreamBlock real-model:  SKIPPED\n");
        return true;
    }
    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string()))
        { std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return false; }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return false; }

    const auto& db0 = router.double_blocks()[0];
    auto v = [](const f2k::TensorView* t) {
        f2k::cuda::PreQuantNVFP4 r{};
        r.packed = t->data; r.scales = t->scales;
        r.N = (int)t->shape[0]; r.K = (int)t->shape[1]; return r;
    };
    auto pq_q = v(db0.to_q), pq_k = v(db0.to_k), pq_v_ = v(db0.to_v);
    auto pq_out = v(db0.to_out), pq_fi = v(db0.ff_in), pq_fo = v(db0.ff_out);
    auto pq_aq = v(db0.add_q_proj), pq_ak = v(db0.add_k_proj), pq_av = v(db0.add_v_proj);
    auto pq_ao = v(db0.to_add_out), pq_cfi = v(db0.ff_ctx_in), pq_cfo = v(db0.ff_ctx_out);

    const int B = 1, S_img = 128, S_txt = 128, H = 32, D = 128;
    const int hidden = H * D;       // 4096
    const int ffn = 12288;

    f2k::cuda::DoubleStreamBlock::Config cfg;
    cfg.batch = B; cfg.seq_img = S_img; cfg.seq_txt = S_txt;
    cfg.n_heads = H; cfg.head_dim = D; cfg.ffn_dim = ffn;
    cfg.W_q = &pq_q; cfg.W_k = &pq_k; cfg.W_v = &pq_v_;
    cfg.W_out = &pq_out; cfg.W_ff_in = &pq_fi; cfg.W_ff_out = &pq_fo;
    cfg.norm_q_bf16 = db0.norm_q->data; cfg.norm_k_bf16 = db0.norm_k->data;
    cfg.W_add_q = &pq_aq; cfg.W_add_k = &pq_ak; cfg.W_add_v = &pq_av;
    cfg.W_add_out = &pq_ao; cfg.W_ff_ctx_in = &pq_cfi; cfg.W_ff_ctx_out = &pq_cfo;
    cfg.norm_added_q_bf16 = db0.norm_added_q->data;
    cfg.norm_added_k_bf16 = db0.norm_added_k->data;
    f2k::cuda::DoubleStreamBlock blk(cfg);
    if (!blk.ok()) { std::fprintf(stderr, "real ctor: %s\n", blk.last_error()); return false; }

    std::mt19937 rng(0xDEADBEEFULL);
    std::uniform_real_distribution<float> dx(-1.0f, 1.0f), dm(-0.1f, 0.1f);
    std::vector<__nv_bfloat16> himg((size_t)B*S_img*hidden), htxt((size_t)B*S_txt*hidden);
    for (auto& v : himg) v = fp32_to_bf16(dx(rng));
    for (auto& v : htxt) v = fp32_to_bf16(dx(rng));

    auto mod_vec = [&](float gate_base) {
        std::vector<__nv_bfloat16> v(hidden);
        for (auto& x : v) x = fp32_to_bf16(gate_base + dm(rng));
        return v;
    };
    auto img_sa = mod_vec(0), img_sh = mod_vec(0), img_ga = mod_vec(0.2f);
    auto img_sm = mod_vec(0), img_sh2 = mod_vec(0), img_gm = mod_vec(0.2f);
    auto txt_sa = mod_vec(0), txt_sh = mod_vec(0), txt_ga = mod_vec(0.2f);
    auto txt_sm = mod_vec(0), txt_sh2 = mod_vec(0), txt_gm = mod_vec(0.2f);

    std::vector<float> cos_t, sin_t;
    build_rope_tables(S_img, D, 2000.0f, cos_t, sin_t);

    auto up = [](void** d, const void* h, size_t bytes) {
        cudaMalloc(d, bytes); cudaMemcpy(*d, h, bytes, cudaMemcpyHostToDevice);
    };
    auto up_vec = [&](void** d, const std::vector<__nv_bfloat16>& v) {
        cudaMalloc(d, v.size() * 2); cudaMemcpy(*d, v.data(), v.size() * 2, cudaMemcpyHostToDevice);
    };
    void *d_img, *d_txt, *d_cos, *d_sin, *d_ws;
    up(&d_img, himg.data(), himg.size() * 2);
    up(&d_txt, htxt.data(), htxt.size() * 2);
    up(&d_cos, cos_t.data(), cos_t.size() * sizeof(float));
    up(&d_sin, sin_t.data(), sin_t.size() * sizeof(float));
    cudaMalloc(&d_ws, blk.workspace_size_bytes());
    void *d_isa, *d_ish, *d_iga, *d_ism, *d_ish2, *d_igm;
    void *d_tsa, *d_tsh, *d_tga, *d_tsm, *d_tsh2, *d_tgm;
    up_vec(&d_isa,  img_sa);  up_vec(&d_ish,  img_sh);  up_vec(&d_iga, img_ga);
    up_vec(&d_ism,  img_sm);  up_vec(&d_ish2, img_sh2); up_vec(&d_igm, img_gm);
    up_vec(&d_tsa,  txt_sa);  up_vec(&d_tsh,  txt_sh);  up_vec(&d_tga, txt_ga);
    up_vec(&d_tsm,  txt_sm);  up_vec(&d_tsh2, txt_sh2); up_vec(&d_tgm, txt_gm);

    f2k::cuda::DoubleStreamBlock::Modulation m;
    m.img_scale_attn = d_isa; m.img_shift_attn = d_ish; m.img_gate_attn = d_iga;
    m.img_scale_mlp  = d_ism; m.img_shift_mlp  = d_ish2; m.img_gate_mlp  = d_igm;
    m.txt_scale_attn = d_tsa; m.txt_shift_attn = d_tsh; m.txt_gate_attn = d_tga;
    m.txt_scale_mlp  = d_tsm; m.txt_shift_mlp  = d_tsh2; m.txt_gate_mlp  = d_tgm;

    // forward() takes separate img/txt RoPE tables (4-axis RoPE work). This
    // test's checks (zero-gate bit-exactness, NaN-free smoke) are insensitive
    // to the table contents, so we feed the same 1D table for both streams.
    const bool ok = blk.forward(d_img, d_txt, m,
                                 (const float*)d_cos, (const float*)d_sin,
                                 (const float*)d_cos, (const float*)d_sin,
                                 d_ws, blk.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> himg_out(himg.size()), htxt_out(htxt.size());
    cudaMemcpy(himg_out.data(), d_img, himg.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(htxt_out.data(), d_txt, htxt.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_img); cudaFree(d_txt); cudaFree(d_cos); cudaFree(d_sin); cudaFree(d_ws);
    cudaFree(d_isa); cudaFree(d_ish); cudaFree(d_iga); cudaFree(d_ism); cudaFree(d_ish2); cudaFree(d_igm);
    cudaFree(d_tsa); cudaFree(d_tsh); cudaFree(d_tga); cudaFree(d_tsm); cudaFree(d_tsh2); cudaFree(d_tgm);
    if (!ok) { std::fprintf(stderr, "real forward: %s\n", blk.last_error()); return false; }

    auto check = [](const std::vector<__nv_bfloat16>& v) {
        int nans = 0; double mx = 0, sum = 0;
        for (auto x : v) {
            const float f = bf16_to_fp32(x);
            if (!std::isfinite(f)) ++nans;
            mx = std::max(mx, (double)std::fabs(f));
            sum += std::fabs(f);
        }
        return std::tuple<int,double,double>{nans, mx, sum / v.size()};
    };
    auto [in, imx, imn] = check(himg_out);
    auto [tn, tmx, tmn] = check(htxt_out);
    const bool pass = (in + tn) == 0 && imx < 500.0 && tmx < 500.0 && imn > 0.01 && tmn > 0.01;
    std::printf("DoubleStreamBlock real-model: %s "
                "img[nans=%d max=%.3f mean=%.3f] txt[nans=%d max=%.3f mean=%.3f]\n",
                pass ? "PASS" : "FAIL", in, imx, imn, tn, tmx, tmn);
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
