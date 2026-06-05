#include "backend/cuda/qwen_encoder.h"

#include "backend/cuda/qwen_layer.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/kernels/embed_lookup.h"
#include "backend/cuda/kernels/rmsnorm.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

// Copy a [S, H] BF16 slice into the columns [col_off, col_off + H) of [S, total_H].
__global__ void scatter_cols_kernel(__nv_bfloat16* __restrict__ dst,
                                    const __nv_bfloat16* __restrict__ src,
                                    int S, int H, int total_H, int col_off) {
    const int s = blockIdx.y;
    if (s >= S) return;
    for (int h = blockIdx.x * blockDim.x + threadIdx.x; h < H; h += blockDim.x * gridDim.x) {
        dst[(size_t)s * total_H + col_off + h] = src[(size_t)s * H + h];
    }
}

void launch_scatter(void* dst, const void* src, int S, int H, int total_H, int col_off,
                    cudaStream_t stream) {
    const int BLOCK = 256;
    dim3 grid((H + BLOCK - 1) / BLOCK, S);
    scatter_cols_kernel<<<grid, BLOCK, 0, stream>>>(
        (__nv_bfloat16*)dst, (const __nv_bfloat16*)src,
        S, H, total_H, col_off);
}

// Build Qwen3-style RoPE tables (half-rotation, theta=rope_theta).
// cos/sin: [seq, head_dim/2] FP32.
void build_rope_tables(int seq, int head_dim, float theta,
                       std::vector<float>& cos, std::vector<float>& sin) {
    const int half = head_dim / 2;
    cos.resize((size_t)seq * half);
    sin.resize((size_t)seq * half);
    for (int p = 0; p < seq; ++p) {
        for (int i = 0; i < half; ++i) {
            const float inv_freq = 1.0f / std::pow(theta, (2.0f * i) / (float)head_dim);
            const float angle = (float)p * inv_freq;
            cos[(size_t)p * half + i] = std::cos(angle);
            sin[(size_t)p * half + i] = std::sin(angle);
        }
    }
}

// Helper: build PreQuantNVFP4 view from a TensorView (same logic as flux_transformer.cu).
f2k::cuda::PreQuantNVFP4 view_pq(const f2k::TensorView* t) {
    f2k::cuda::PreQuantNVFP4 r{};
    if (!t) return r;
    r.packed = t->data;
    r.scales = t->scales;
    r.N = (int)t->shape[0];
    r.K = (int)t->shape[1];
    r.microblock_size = t->microblock_size ? t->microblock_size : 16;
    r.tensor_scale = t->tensor_scale ? t->tensor_scale : 1.0f;
    return r;
}

} // anonymous namespace

namespace f2k::cuda {

struct QwenEncoder::Impl {
    Config cfg{};
    bool valid = false;
    std::string err;

    // Device buffers we own.
    void* d_embed_table = nullptr;     // [vocab, hidden] BF16 (uploaded copy)
    void* d_final_gamma = nullptr;     // [hidden] BF16
    float* d_rope_cos = nullptr;
    float* d_rope_sin = nullptr;
    int32_t* d_pos_q  = nullptr;
    int32_t* d_pos_k  = nullptr;

    // Pre-quant weight views (owned to outlive layer Linears).
    std::vector<PreQuantNVFP4> pq_qkv_q, pq_qkv_k, pq_qkv_v, pq_o;
    std::vector<PreQuantNVFP4> pq_g, pq_u, pq_d;

    std::vector<std::unique_ptr<QwenLayer>> layers;

    // Workspace offsets.
    size_t off_buf_a = 0;          // [B*S, hidden]
    size_t off_buf_b = 0;
    size_t off_layer_ws = 0;
    size_t layer_ws_max = 0;
    size_t total_ws = 0;
};

QwenEncoder::QwenEncoder(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;
    if (cfg.seq <= 0 || !cfg.loader)            { I.err = "bad cfg"; return; }
    if (cfg.hidden % 64 != 0 || cfg.seq % 128 != 0) { I.err = "seq%128/hidden%64 required"; return; }

    // Default capture: last 3 layers.
    auto cap = cfg.capture_layers;
    if (cap[0] < 0) cap = { cfg.n_layers - 3, cfg.n_layers - 2, cfg.n_layers - 1 };
    for (int v : cap) {
        if (v < 0 || v >= cfg.n_layers) { I.err = "capture_layers out of range"; return; }
    }
    I.cfg.capture_layers = cap;

    const std::string p = cfg.prefix + ".";

    // --- embed_tokens (BF16) ---
    {
        const auto* t = cfg.loader->find(p + "embed_tokens.weight");
        if (!t || t->shape.size() != 2) { I.err = "missing embed_tokens"; return; }
        const size_t bytes = (size_t)t->shape[0] * t->shape[1] * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_embed_table, bytes) != cudaSuccess) { I.err = "malloc embed"; return; }
        cudaMemcpy(I.d_embed_table, t->data, bytes, cudaMemcpyHostToDevice);
    }

    // --- final norm gamma (BF16) ---
    {
        const auto* t = cfg.loader->find(p + "norm.weight");
        if (!t) { I.err = "missing final norm"; return; }
        const size_t bytes = (size_t)cfg.hidden * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_final_gamma, bytes) != cudaSuccess) { I.err = "malloc final_gamma"; return; }
        cudaMemcpy(I.d_final_gamma, t->data, bytes, cudaMemcpyHostToDevice);
    }

    // --- RoPE tables ---
    {
        std::vector<float> hcos, hsin;
        build_rope_tables(cfg.seq, cfg.head_dim, cfg.rope_theta, hcos, hsin);
        const size_t bytes = hcos.size() * sizeof(float);
        if (cudaMalloc(&I.d_rope_cos, bytes) != cudaSuccess) { I.err = "malloc rope cos"; return; }
        if (cudaMalloc(&I.d_rope_sin, bytes) != cudaSuccess) { I.err = "malloc rope sin"; return; }
        cudaMemcpy(I.d_rope_cos, hcos.data(), bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(I.d_rope_sin, hsin.data(), bytes, cudaMemcpyHostToDevice);
    }

    // --- pos_ids ---
    {
        std::vector<int32_t> pq((size_t)cfg.seq * cfg.n_heads);
        std::vector<int32_t> pk((size_t)cfg.seq * cfg.n_kv_heads);
        for (int s = 0; s < cfg.seq; ++s) {
            for (int h = 0; h < cfg.n_heads;    ++h) pq[(size_t)s * cfg.n_heads    + h] = s;
            for (int h = 0; h < cfg.n_kv_heads; ++h) pk[(size_t)s * cfg.n_kv_heads + h] = s;
        }
        if (cudaMalloc(&I.d_pos_q, pq.size() * sizeof(int32_t)) != cudaSuccess) { I.err = "malloc pos_q"; return; }
        if (cudaMalloc(&I.d_pos_k, pk.size() * sizeof(int32_t)) != cudaSuccess) { I.err = "malloc pos_k"; return; }
        cudaMemcpy(I.d_pos_q, pq.data(), pq.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
        cudaMemcpy(I.d_pos_k, pk.data(), pk.size() * sizeof(int32_t), cudaMemcpyHostToDevice);
    }

    // --- 36 layers ---
    I.pq_qkv_q.resize(cfg.n_layers); I.pq_qkv_k.resize(cfg.n_layers);
    I.pq_qkv_v.resize(cfg.n_layers); I.pq_o.resize(cfg.n_layers);
    I.pq_g.resize(cfg.n_layers); I.pq_u.resize(cfg.n_layers); I.pq_d.resize(cfg.n_layers);
    I.layers.resize(cfg.n_layers);
    for (int i = 0; i < cfg.n_layers; ++i) {
        char buf[256];
        auto T = [&](const char* suffix) -> const f2k::TensorView* {
            std::snprintf(buf, sizeof(buf), "%slayers.%d.%s", p.c_str(), i, suffix);
            const auto* t = cfg.loader->find(buf);
            if (!t) { I.err = std::string("missing: ") + buf; }
            return t;
        };
        const auto* tin   = T("input_layernorm.weight");           if (!tin)   return;
        const auto* tpost = T("post_attention_layernorm.weight");  if (!tpost) return;
        const auto* tqn   = T("self_attn.q_norm.weight");          if (!tqn)   return;
        const auto* tkn   = T("self_attn.k_norm.weight");          if (!tkn)   return;
        const auto* tq    = T("self_attn.q_proj.weight");          if (!tq)    return;
        const auto* tk    = T("self_attn.k_proj.weight");          if (!tk)    return;
        const auto* tv    = T("self_attn.v_proj.weight");          if (!tv)    return;
        const auto* to    = T("self_attn.o_proj.weight");          if (!to)    return;
        const auto* tg    = T("mlp.gate_proj.weight");             if (!tg)    return;
        const auto* tu    = T("mlp.up_proj.weight");               if (!tu)    return;
        const auto* td    = T("mlp.down_proj.weight");             if (!td)    return;

        I.pq_qkv_q[i] = view_pq(tq);
        I.pq_qkv_k[i] = view_pq(tk);
        I.pq_qkv_v[i] = view_pq(tv);
        I.pq_o[i]     = view_pq(to);
        I.pq_g[i]     = view_pq(tg);
        I.pq_u[i]     = view_pq(tu);
        I.pq_d[i]     = view_pq(td);

        QwenLayer::Config lc{};
        lc.batch = 1; lc.seq = cfg.seq; lc.hidden = cfg.hidden;
        lc.n_heads = cfg.n_heads; lc.n_kv_heads = cfg.n_kv_heads;
        lc.head_dim = cfg.head_dim; lc.ffn_dim = cfg.ffn_dim;
        lc.rms_eps = cfg.rms_eps;
        lc.input_norm_gamma = tin->data;
        lc.post_norm_gamma  = tpost->data;
        lc.q_norm_gamma     = tqn->data;
        lc.k_norm_gamma     = tkn->data;
        lc.q_proj = &I.pq_qkv_q[i];
        lc.k_proj = &I.pq_qkv_k[i];
        lc.v_proj = &I.pq_qkv_v[i];
        lc.o_proj = &I.pq_o[i];
        lc.gate_proj = &I.pq_g[i];
        lc.up_proj   = &I.pq_u[i];
        lc.down_proj = &I.pq_d[i];

        I.layers[i] = std::make_unique<QwenLayer>(lc);
        if (!I.layers[i]->ok()) {
            I.err = std::string("layer ") + std::to_string(i) + ": " + I.layers[i]->last_error();
            return;
        }
        I.layer_ws_max = std::max(I.layer_ws_max, I.layers[i]->workspace_size_bytes());
    }

    // Workspace plan.
    const size_t hidden_bytes = (size_t)cfg.seq * cfg.hidden * sizeof(__nv_bfloat16);
    size_t c = 0;
    c = align_up(c, ALIGN); I.off_buf_a    = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_buf_b    = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_layer_ws = c; c += I.layer_ws_max;
    I.total_ws = c;

    I.valid = true;
}

QwenEncoder::~QwenEncoder() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_embed_table) cudaFree(I.d_embed_table);
    if (I.d_final_gamma) cudaFree(I.d_final_gamma);
    if (I.d_rope_cos)    cudaFree(I.d_rope_cos);
    if (I.d_rope_sin)    cudaFree(I.d_rope_sin);
    if (I.d_pos_q)       cudaFree(I.d_pos_q);
    if (I.d_pos_k)       cudaFree(I.d_pos_k);
}

bool        QwenEncoder::ok()         const { return impl_ && impl_->valid; }
const char* QwenEncoder::last_error() const { return impl_ ? impl_->err.c_str() : "no impl"; }
size_t      QwenEncoder::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }
int         QwenEncoder::output_hidden() const { return impl_ ? impl_->cfg.hidden * 3 : 0; }

bool QwenEncoder::forward(const int32_t* token_ids, void* out,
                          void* ws_v, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }
    const auto& cfg = I.cfg;

    uint8_t* ws = (uint8_t*)ws_v;
    void* A = ws + I.off_buf_a;
    void* B = ws + I.off_buf_b;
    void* layer_ws = ws + I.off_layer_ws;

    // 1. embed_tokens: token_ids → A
    if (!embed_lookup_bf16(I.d_embed_table, token_ids, A,
                            cfg.seq, cfg.hidden, cfg.vocab_size, stream))
        { I.err = "embed_lookup"; return false; }

    // 2. 36-layer forward, capturing layer outputs at configured indices.
    const int cap0 = cfg.capture_layers[0];
    const int cap1 = cfg.capture_layers[1];
    const int cap2 = cfg.capture_layers[2];

    for (int i = 0; i < cfg.n_layers; ++i) {
        // Alternate A → B → A → ...
        void* in  = (i & 1) ? B : A;
        void* outp = (i & 1) ? A : B;
        if (!I.layers[i]->forward(in, outp, I.d_rope_cos, I.d_rope_sin,
                                  I.d_pos_q, I.d_pos_k,
                                  layer_ws, I.layer_ws_max, stream))
            { I.err = std::string("layer ") + std::to_string(i) + ": " + I.layers[i]->last_error(); return false; }
        // The "post-layer hidden state" that diffusers extracts is what flows OUT of
        // this layer (i.e. the residual stream after this layer's MLP add).
        if (i == cap0) launch_scatter(out, outp, cfg.seq, cfg.hidden, 3 * cfg.hidden, 0,                  stream);
        if (i == cap1) launch_scatter(out, outp, cfg.seq, cfg.hidden, 3 * cfg.hidden, cfg.hidden,         stream);
        if (i == cap2) launch_scatter(out, outp, cfg.seq, cfg.hidden, 3 * cfg.hidden, 2 * cfg.hidden,     stream);
    }
    // After 36 layers (even number), data is in A. (i=0 writes A→B; i=35 writes B→A.)
    // But we DON'T need the final normed value for the conditioning path — only the
    // captured layer outputs. We still launch the final RMSNorm into A for completeness,
    // so that callers who want the last hidden state have it accessible if they ask
    // for capture_layers = {n-1, n-1, n-1}. For the standard 3-layer concat, this is
    // a no-op for the output.
    // (Skipping the final norm here keeps the encoder simpler; we can add it later.)
    (void)I.d_final_gamma; // referenced for symmetry; intentionally unused for v1
    return true;
}

} // namespace f2k::cuda
