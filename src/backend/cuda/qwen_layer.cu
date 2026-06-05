#include "backend/cuda/qwen_layer.h"

#include "backend/cuda/qwen_attention.h"
#include "backend/cuda/kernels/rmsnorm.h"
#include "backend/cuda/kernels/rope.h"
#include "backend/cuda/kernels/silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstring>
#include <memory>
#include <string>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

__global__ void add_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                size_t n) {
    const size_t i = (size_t)blockIdx.x * 256 + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

void launch_add(void* y, const void* a, const void* b, size_t n, cudaStream_t stream) {
    const int BLOCK = 256;
    add_bf16_kernel<<<(int)((n + BLOCK - 1) / BLOCK), BLOCK, 0, stream>>>(
        (__nv_bfloat16*)y, (const __nv_bfloat16*)a, (const __nv_bfloat16*)b, n);
}

} // anonymous namespace

namespace f2k::cuda {

struct QwenLayer::Impl {
    Config cfg{};
    bool valid = false;
    std::string err;

    // Norm gains on device.
    void* d_in_gamma   = nullptr;
    void* d_post_gamma = nullptr;
    void* d_q_gamma    = nullptr;
    void* d_k_gamma    = nullptr;

    std::unique_ptr<Linear> Wq, Wk, Wv, Wo, Wg, Wu, Wd;

    // Workspace offsets (relative to base).
    size_t off_normed = 0;
    size_t off_q = 0, off_k = 0, off_v = 0;
    size_t off_attn = 0;        // [B*S, Hq*D]
    size_t off_oproj = 0;       // [B*S, hidden]
    size_t off_residual_attn = 0; // hold x+oproj
    size_t off_post = 0;
    size_t off_gate = 0, off_up = 0;
    size_t off_silu = 0;
    size_t off_mlp_out = 0;
    size_t off_lin_ws = 0;
    size_t lin_ws_max = 0;
    size_t total_ws = 0;
};

QwenLayer::QwenLayer(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;

    auto bad = [&](const char* m) { I.err = m; };
    if (cfg.batch <= 0 || cfg.seq <= 0 || cfg.hidden <= 0) { bad("bad dims"); return; }
    if (cfg.n_heads <= 0 || cfg.n_kv_heads <= 0 || cfg.head_dim <= 0) { bad("bad attn dims"); return; }
    if (cfg.n_heads % cfg.n_kv_heads != 0) { bad("n_heads %% n_kv_heads != 0"); return; }
    if (!cfg.input_norm_gamma || !cfg.post_norm_gamma ||
        !cfg.q_norm_gamma || !cfg.k_norm_gamma) { bad("missing norm gammas"); return; }
    if (!cfg.q_proj || !cfg.k_proj || !cfg.v_proj || !cfg.o_proj ||
        !cfg.gate_proj || !cfg.up_proj || !cfg.down_proj) { bad("missing proj weights"); return; }

    auto upload = [&](void** dst, const void* src, int n_elem) {
        const size_t b = (size_t)n_elem * sizeof(__nv_bfloat16);
        if (cudaMalloc(dst, b) != cudaSuccess) return false;
        cudaMemcpy(*dst, src, b, cudaMemcpyHostToDevice);
        return true;
    };
    if (!upload(&I.d_in_gamma,   cfg.input_norm_gamma, cfg.hidden))   { bad("malloc in_gamma"); return; }
    if (!upload(&I.d_post_gamma, cfg.post_norm_gamma,  cfg.hidden))   { bad("malloc post_gamma"); return; }
    if (!upload(&I.d_q_gamma,    cfg.q_norm_gamma,     cfg.head_dim)) { bad("malloc q_gamma"); return; }
    if (!upload(&I.d_k_gamma,    cfg.k_norm_gamma,     cfg.head_dim)) { bad("malloc k_gamma"); return; }

    const int M = cfg.batch * cfg.seq;
    const int n_q_features  = cfg.n_heads    * cfg.head_dim;
    const int n_kv_features = cfg.n_kv_heads * cfg.head_dim;

    auto make_lin = [&](const PreQuantNVFP4* W) {
        Linear::Config lc{};
        lc.batch_rows   = M;
        lc.in_features  = cfg.hidden;
        lc.out_features = W->N;
        lc.W_preq = W;
        return std::make_unique<Linear>(lc);
    };
    auto make_lin_in = [&](const PreQuantNVFP4* W, int in_feat) {
        Linear::Config lc{};
        lc.batch_rows   = M;
        lc.in_features  = in_feat;
        lc.out_features = W->N;
        lc.W_preq = W;
        return std::make_unique<Linear>(lc);
    };

    I.Wq = make_lin(cfg.q_proj);
    I.Wk = make_lin(cfg.k_proj);
    I.Wv = make_lin(cfg.v_proj);
    I.Wo = make_lin_in(cfg.o_proj, n_q_features);
    I.Wg = make_lin(cfg.gate_proj);
    I.Wu = make_lin(cfg.up_proj);
    I.Wd = make_lin_in(cfg.down_proj, cfg.ffn_dim);
    for (auto* L : {I.Wq.get(), I.Wk.get(), I.Wv.get(), I.Wo.get(),
                     I.Wg.get(), I.Wu.get(), I.Wd.get()}) {
        if (!L->ok()) { I.err = std::string("Linear: ") + L->last_error(); return; }
        I.lin_ws_max = std::max(I.lin_ws_max, L->workspace_size_bytes());
    }

    // Workspace layout.
    const size_t hidden_bytes  = (size_t)M * cfg.hidden        * sizeof(__nv_bfloat16);
    const size_t q_bytes       = (size_t)M * n_q_features      * sizeof(__nv_bfloat16);
    const size_t kv_bytes      = (size_t)M * n_kv_features     * sizeof(__nv_bfloat16);
    const size_t ffn_bytes     = (size_t)M * cfg.ffn_dim       * sizeof(__nv_bfloat16);

    size_t c = 0;
    c = align_up(c, ALIGN); I.off_normed         = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_q              = c; c += q_bytes;
    c = align_up(c, ALIGN); I.off_k              = c; c += kv_bytes;
    c = align_up(c, ALIGN); I.off_v              = c; c += kv_bytes;
    c = align_up(c, ALIGN); I.off_attn           = c; c += q_bytes;
    c = align_up(c, ALIGN); I.off_oproj          = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_residual_attn  = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_post           = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_gate           = c; c += ffn_bytes;
    c = align_up(c, ALIGN); I.off_up             = c; c += ffn_bytes;
    c = align_up(c, ALIGN); I.off_silu           = c; c += ffn_bytes;
    c = align_up(c, ALIGN); I.off_mlp_out        = c; c += hidden_bytes;
    c = align_up(c, ALIGN); I.off_lin_ws         = c; c += I.lin_ws_max;
    I.total_ws = c;

    I.valid = true;
}

QwenLayer::~QwenLayer() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_in_gamma)   cudaFree(I.d_in_gamma);
    if (I.d_post_gamma) cudaFree(I.d_post_gamma);
    if (I.d_q_gamma)    cudaFree(I.d_q_gamma);
    if (I.d_k_gamma)    cudaFree(I.d_k_gamma);
}

bool        QwenLayer::ok()         const { return impl_ && impl_->valid; }
const char* QwenLayer::last_error() const { return impl_ ? impl_->err.c_str() : "no impl"; }
size_t      QwenLayer::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

bool QwenLayer::forward(const void* x, void* y,
                         const float* rope_cos, const float* rope_sin,
                         const int32_t* pos_ids_q, const int32_t* pos_ids_k,
                         void* ws_v, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }
    const auto& cfg = I.cfg;
    const int M = cfg.batch * cfg.seq;
    const int Hq  = cfg.n_heads;
    const int Hkv = cfg.n_kv_heads;
    const int D   = cfg.head_dim;

    uint8_t* ws = (uint8_t*)ws_v;
    void* normed = ws + I.off_normed;
    void* q      = ws + I.off_q;
    void* k      = ws + I.off_k;
    void* v      = ws + I.off_v;
    void* attn   = ws + I.off_attn;
    void* oproj  = ws + I.off_oproj;
    void* res1   = ws + I.off_residual_attn;
    void* post   = ws + I.off_post;
    void* gate   = ws + I.off_gate;
    void* up     = ws + I.off_up;
    void* silu   = ws + I.off_silu;
    void* mlpo   = ws + I.off_mlp_out;
    void* lin_ws = ws + I.off_lin_ws;

    // 1. input_layernorm
    if (!rmsnorm_bf16(x, I.d_in_gamma, normed, M, cfg.hidden, cfg.rms_eps, stream))
        { I.err = "in_norm"; return false; }

    // 2. q/k/v projections
    if (!I.Wq->forward(normed, q, lin_ws, I.lin_ws_max, stream)) { I.err = "q_proj"; return false; }
    if (!I.Wk->forward(normed, k, lin_ws, I.lin_ws_max, stream)) { I.err = "k_proj"; return false; }
    if (!I.Wv->forward(normed, v, lin_ws, I.lin_ws_max, stream)) { I.err = "v_proj"; return false; }

    // 3. q_norm/k_norm (per-head, per-token RMSNorm with gamma[head_dim])
    //    Treat q as [M*Hq, D] and k as [M*Hkv, D]; rmsnorm in place into the same buffers.
    //    Source and dest can both be q because rmsnorm_bf16 supports it (caller doesn't
    //    require non-aliasing).
    if (!rmsnorm_bf16(q, I.d_q_gamma, q, M * Hq, D, cfg.rms_eps, stream))
        { I.err = "q_norm"; return false; }
    if (!rmsnorm_bf16(k, I.d_k_gamma, k, M * Hkv, D, cfg.rms_eps, stream))
        { I.err = "k_norm"; return false; }

    // 4. RoPE in-place on q, k.
    if (!rope_inplace_bf16(q, rope_cos, rope_sin, pos_ids_q, M * Hq,  D, cfg.seq, stream))
        { I.err = "rope_q"; return false; }
    if (!rope_inplace_bf16(k, rope_cos, rope_sin, pos_ids_k, M * Hkv, D, cfg.seq, stream))
        { I.err = "rope_k"; return false; }

    // 5. GQA causal attention.
    QwenGQAAttention::Config ac{};
    ac.batch = cfg.batch; ac.seq = cfg.seq;
    ac.n_heads = Hq; ac.n_kv_heads = Hkv; ac.head_dim = D;
    if (!QwenGQAAttention::forward(ac, q, k, v, attn, stream)) { I.err = "gqa"; return false; }

    // 6. o_proj
    if (!I.Wo->forward(attn, oproj, lin_ws, I.lin_ws_max, stream)) { I.err = "o_proj"; return false; }

    // 7. residual: res1 = x + oproj
    launch_add(res1, x, oproj, (size_t)M * cfg.hidden, stream);

    // 8. post_attention_layernorm
    if (!rmsnorm_bf16(res1, I.d_post_gamma, post, M, cfg.hidden, cfg.rms_eps, stream))
        { I.err = "post_norm"; return false; }

    // 9. SwiGLU FFN: gate, up, silu(gate)*up, down
    if (!I.Wg->forward(post, gate, lin_ws, I.lin_ws_max, stream)) { I.err = "gate"; return false; }
    if (!I.Wu->forward(post, up,   lin_ws, I.lin_ws_max, stream)) { I.err = "up";   return false; }
    if (!silu_mul_bf16(gate, up, silu, (size_t)M * cfg.ffn_dim, stream))
        { I.err = "silu_mul"; return false; }
    if (!I.Wd->forward(silu, mlpo, lin_ws, I.lin_ws_max, stream)) { I.err = "down"; return false; }

    // 10. residual: y = res1 + mlp_out
    launch_add(y, res1, mlpo, (size_t)M * cfg.hidden, stream);
    return true;
}

} // namespace f2k::cuda
