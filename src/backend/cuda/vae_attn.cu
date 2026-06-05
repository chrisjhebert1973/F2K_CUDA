// VAE spatial self-attention.

#include "backend/cuda/vae_attn.h"

#include "backend/cuda/attention.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/kernels/groupnorm.h"
#include "backend/cuda/kernels/transpose_chw.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

__global__ void add_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    y[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
}

} // anonymous namespace

namespace f2k::cuda {

struct VAEAttention::Impl {
    Config cfg{};
    bool valid = false;
    std::string err;

    void* d_norm_gain = nullptr;
    void* d_norm_bias = nullptr;

    std::unique_ptr<Linear> Wq, Wk, Wv, Wo;
    std::unique_ptr<Attention> attn;

    // Workspace offsets.
    size_t off_normed_nchw = 0;   // [N, C, H, W]
    size_t off_normed_nsc  = 0;   // [N, S, C]
    size_t off_q = 0, off_k = 0, off_v = 0;        // [N, S, C]
    size_t off_attn_o = 0;        // [N, S, C]
    size_t off_proj_o = 0;        // [N, S, C]
    size_t off_proj_o_nchw = 0;   // [N, C, H, W]
    size_t off_lin_ws = 0;        // shared scratch for the 4 Linear calls
    size_t total_ws = 0;
};

VAEAttention::VAEAttention(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;

    if (cfg.N <= 0 || cfg.C <= 0 || cfg.H <= 0 || cfg.W <= 0)
        { I.err = "VAEAttention: bad dims"; return; }
    const int S = cfg.H * cfg.W;
    if ((cfg.N * S) % 128 != 0) { I.err = "N*H*W must be %128 for NVFP4 Linear"; return; }
    if (cfg.C % 128 != 0)       { I.err = "C must be %128"; return; }

    auto upload = [&](void** dst, const void* host, int n_elem) -> bool {
        const size_t b = static_cast<size_t>(n_elem) * sizeof(__nv_bfloat16);
        if (cudaMalloc(dst, b) != cudaSuccess) return false;
        cudaMemcpy(*dst, host, b, cudaMemcpyHostToDevice);
        return true;
    };
    if (!upload(&I.d_norm_gain, cfg.norm_gain, cfg.C)) { I.err = "malloc norm gain"; return; }
    if (!upload(&I.d_norm_bias, cfg.norm_bias, cfg.C)) { I.err = "malloc norm bias"; return; }

    auto make_lin = [&](const void* W, const void* b) {
        Linear::Config lc{};
        lc.batch_rows   = cfg.N * S;
        lc.in_features  = cfg.C;
        lc.out_features = cfg.C;
        lc.W_bf16       = W;
        lc.bias_bf16    = b;
        return std::make_unique<Linear>(lc);
    };
    I.Wq = make_lin(cfg.to_q_W,  cfg.to_q_b);
    I.Wk = make_lin(cfg.to_k_W,  cfg.to_k_b);
    I.Wv = make_lin(cfg.to_v_W,  cfg.to_v_b);
    I.Wo = make_lin(cfg.to_out_W, cfg.to_out_b);
    for (auto* L : {I.Wq.get(), I.Wk.get(), I.Wv.get(), I.Wo.get()}) {
        if (!L->ok()) { I.err = std::string("Linear: ") + L->last_error(); return; }
    }

    Attention::Config ac{};
    ac.batch    = cfg.N;
    ac.seq      = S;
    ac.n_heads  = 1;
    ac.head_dim = cfg.C;
    ac.scale    = 0.0f;   // default = 1/sqrt(C)
    I.attn = std::make_unique<Attention>(ac);
    if (!I.attn->ok()) { I.err = std::string("Attention: ") + I.attn->last_error(); return; }

    // Workspace layout.
    const size_t nchw_bytes = static_cast<size_t>(cfg.N) * cfg.C * cfg.H * cfg.W * sizeof(__nv_bfloat16);
    const size_t nsc_bytes  = static_cast<size_t>(cfg.N) * S       * cfg.C       * sizeof(__nv_bfloat16);
    size_t lin_ws = 0;
    for (auto* L : {I.Wq.get(), I.Wk.get(), I.Wv.get(), I.Wo.get()})
        lin_ws = std::max(lin_ws, L->workspace_size_bytes());

    size_t c = 0;
    c = align_up(c, ALIGN); I.off_normed_nchw = c; c += nchw_bytes;
    c = align_up(c, ALIGN); I.off_normed_nsc  = c; c += nsc_bytes;
    c = align_up(c, ALIGN); I.off_q           = c; c += nsc_bytes;
    c = align_up(c, ALIGN); I.off_k           = c; c += nsc_bytes;
    c = align_up(c, ALIGN); I.off_v           = c; c += nsc_bytes;
    c = align_up(c, ALIGN); I.off_attn_o      = c; c += nsc_bytes;
    c = align_up(c, ALIGN); I.off_proj_o      = c; c += nsc_bytes;
    c = align_up(c, ALIGN); I.off_proj_o_nchw = c; c += nchw_bytes;
    c = align_up(c, ALIGN); I.off_lin_ws      = c; c += lin_ws;
    I.total_ws = c;

    I.valid = true;
}

VAEAttention::~VAEAttention() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_norm_gain) cudaFree(I.d_norm_gain);
    if (I.d_norm_bias) cudaFree(I.d_norm_bias);
}

bool        VAEAttention::ok()         const { return impl_ && impl_->valid; }
const char* VAEAttention::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t VAEAttention::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

bool VAEAttention::forward(const void* x, void* y,
                           void* ws_v, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }
    const auto& cfg = I.cfg;
    const int S = cfg.H * cfg.W;

    uint8_t* ws = static_cast<uint8_t*>(ws_v);
    void* h_nchw       = ws + I.off_normed_nchw;
    void* h_nsc        = ws + I.off_normed_nsc;
    void* q            = ws + I.off_q;
    void* k            = ws + I.off_k;
    void* v            = ws + I.off_v;
    void* attn_o       = ws + I.off_attn_o;
    void* proj_o       = ws + I.off_proj_o;
    void* proj_o_nchw  = ws + I.off_proj_o_nchw;
    void* lin_ws       = ws + I.off_lin_ws;
    const size_t lin_ws_sz = std::max({I.Wq->workspace_size_bytes(), I.Wk->workspace_size_bytes(),
                                       I.Wv->workspace_size_bytes(), I.Wo->workspace_size_bytes()});

    if (!groupnorm_bf16(x, h_nchw, I.d_norm_gain, I.d_norm_bias,
                        cfg.N, cfg.C, cfg.H, cfg.W, cfg.num_groups, cfg.eps, stream))
        { I.err = "groupnorm"; return false; }
    if (!nchw_to_nsc_bf16(h_nchw, h_nsc, cfg.N, cfg.C, cfg.H, cfg.W, stream))
        { I.err = "nchw→nsc"; return false; }

    if (!I.Wq->forward(h_nsc, q, lin_ws, lin_ws_sz, stream)) { I.err = "to_q"; return false; }
    if (!I.Wk->forward(h_nsc, k, lin_ws, lin_ws_sz, stream)) { I.err = "to_k"; return false; }
    if (!I.Wv->forward(h_nsc, v, lin_ws, lin_ws_sz, stream)) { I.err = "to_v"; return false; }

    // n_heads=1, head_dim=C ⇒ (B, S, 1, C) is already the right layout.
    if (!I.attn->forward(q, k, v, attn_o, stream)) { I.err = "attention"; return false; }

    if (!I.Wo->forward(attn_o, proj_o, lin_ws, lin_ws_sz, stream)) { I.err = "to_out"; return false; }

    if (!nsc_to_nchw_bf16(proj_o, proj_o_nchw, cfg.N, cfg.C, cfg.H, cfg.W, stream))
        { I.err = "nsc→nchw"; return false; }

    const size_t n_elem = static_cast<size_t>(cfg.N) * cfg.C * cfg.H * cfg.W;
    const int BLOCK = 256;
    add_bf16_kernel<<<static_cast<int>((n_elem + BLOCK - 1) / BLOCK), BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(proj_o_nchw),
        n_elem);
    if (cudaPeekAtLastError() != cudaSuccess) { I.err = "add"; return false; }
    return true;
}

} // namespace f2k::cuda
