// MMDiTBlock — orchestrates Linear / RMSNorm / RoPE / Attention / SwiGLU /
// modulate / gated_residual into one transformer block.

#include "backend/cuda/mmdit_block.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/swiglu.h"
#include "backend/cuda/attention.h"
#include "backend/cuda/kernels/rmsnorm.h"
#include "backend/cuda/kernels/rope.h"
#include "backend/cuda/kernels/modulation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

// Tiny kernel: expand per-row pos_ids (seq indices) to per-(row, head) pos_ids
// for the RoPE call. With pos[r] = r % seq, output[r * n_heads + h] = pos[r].
__global__ void fill_pos_ids_expanded(int32_t* out, int batch_rows, int n_heads, int seq) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_rows * n_heads) return;
    const int r = idx / n_heads;
    out[idx] = r % seq;
}

} // anonymous namespace

namespace f2k::cuda {

struct MMDiTBlock::Impl {
    int batch = 0, seq = 0, n_heads = 0, head_dim = 0, ffn_dim = 0;
    int batch_rows = 0;
    int hidden_dim = 0;
    float rms_eps = 1e-6f;

    bool        valid = false;
    std::string err;

    // Owned sub-components (each holds its own static weight buffers).
    std::unique_ptr<Linear>    lin_q, lin_k, lin_v, lin_proj;
    std::unique_ptr<SwiGLU>    mlp;
    std::unique_ptr<Attention> attn;

    // Device-resident BF16 norm gains, copied from host at construction.
    void* d_W_norm1 = nullptr;
    void* d_W_norm2 = nullptr;

    // Workspace partitioning (offsets in bytes).
    size_t off_norm_buf  = 0;   // [batch_rows, hidden_dim] BF16, reused
    size_t off_mod_buf   = 0;   // [batch_rows, hidden_dim] BF16
    size_t off_Q         = 0;   // [batch_rows, hidden_dim] BF16
    size_t off_K         = 0;
    size_t off_V         = 0;
    size_t off_attn_out  = 0;
    size_t off_resid_buf = 0;   // proj_out / mlp_out target
    size_t off_pos_ids   = 0;   // [batch_rows * n_heads] int32

    size_t off_lin_q_ws  = 0;
    size_t off_lin_k_ws  = 0;
    size_t off_lin_v_ws  = 0;
    size_t off_lin_proj_ws = 0;
    size_t off_mlp_ws    = 0;

    size_t total_ws = 0;
};

MMDiTBlock::MMDiTBlock(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.batch      = cfg.batch;
    I.seq        = cfg.seq;
    I.n_heads    = cfg.n_heads;
    I.head_dim   = cfg.head_dim;
    I.ffn_dim    = cfg.ffn_dim;
    I.rms_eps    = cfg.rms_eps;
    I.batch_rows = cfg.batch * cfg.seq;
    I.hidden_dim = cfg.n_heads * cfg.head_dim;

    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (I.batch <= 0 || I.seq <= 0 || I.n_heads <= 0 || I.head_dim <= 0 || I.ffn_dim <= 0)
        { fail("MMDiTBlock: zero/negative dim"); return; }
    if (I.batch_rows % 128 != 0)
        { fail("batch * seq must be a multiple of 128 (Linear M constraint)"); return; }
    if (I.hidden_dim % 64 != 0)
        { fail("hidden_dim must be a multiple of 64"); return; }
    if (I.ffn_dim % 128 != 0)
        { fail("ffn_dim must be a multiple of 128"); return; }
    if (I.head_dim & 1)
        { fail("head_dim must be even (RoPE)"); return; }
    if (!cfg.W_norm1 || !cfg.W_q || !cfg.W_k || !cfg.W_v || !cfg.W_proj ||
        !cfg.W_norm2 || !cfg.W_gate || !cfg.W_up || !cfg.W_down)
        { fail("MMDiTBlock: a weight pointer is null"); return; }

    // Norm gains → device.
    const size_t norm_bytes = static_cast<size_t>(I.hidden_dim) * sizeof(__nv_bfloat16);
    if (cudaMalloc(&I.d_W_norm1, norm_bytes) != cudaSuccess) { fail("malloc W_norm1"); return; }
    if (cudaMalloc(&I.d_W_norm2, norm_bytes) != cudaSuccess) { fail("malloc W_norm2"); return; }
    cudaMemcpy(I.d_W_norm1, cfg.W_norm1, norm_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(I.d_W_norm2, cfg.W_norm2, norm_bytes, cudaMemcpyHostToDevice);

    // Linears.
    auto make_lin = [&](int K, int N, const void* W) {
        Linear::Config c;
        c.batch_rows   = I.batch_rows;
        c.in_features  = K;
        c.out_features = N;
        c.W_bf16       = W;
        return std::make_unique<Linear>(c);
    };
    I.lin_q    = make_lin(I.hidden_dim, I.hidden_dim, cfg.W_q);
    I.lin_k    = make_lin(I.hidden_dim, I.hidden_dim, cfg.W_k);
    I.lin_v    = make_lin(I.hidden_dim, I.hidden_dim, cfg.W_v);
    I.lin_proj = make_lin(I.hidden_dim, I.hidden_dim, cfg.W_proj);
    if (!I.lin_q->ok())    { fail(std::string("lin_q: ") + I.lin_q->last_error()); return; }
    if (!I.lin_k->ok())    { fail(std::string("lin_k: ") + I.lin_k->last_error()); return; }
    if (!I.lin_v->ok())    { fail(std::string("lin_v: ") + I.lin_v->last_error()); return; }
    if (!I.lin_proj->ok()) { fail(std::string("lin_proj: ") + I.lin_proj->last_error()); return; }

    // SwiGLU.
    SwiGLU::Config sc;
    sc.batch_rows  = I.batch_rows;
    sc.hidden_dim  = I.hidden_dim;
    sc.ffn_dim     = I.ffn_dim;
    sc.W_gate_bf16 = cfg.W_gate;
    sc.W_up_bf16   = cfg.W_up;
    sc.W_down_bf16 = cfg.W_down;
    I.mlp = std::make_unique<SwiGLU>(sc);
    if (!I.mlp->ok()) { fail(std::string("mlp: ") + I.mlp->last_error()); return; }

    // Attention.
    Attention::Config ac{};
    ac.batch    = I.batch;
    ac.seq      = I.seq;
    ac.n_heads  = I.n_heads;
    ac.head_dim = I.head_dim;
    I.attn = std::make_unique<Attention>(ac);
    if (!I.attn->ok()) { fail(std::string("attn: ") + I.attn->last_error()); return; }

    // Workspace plan.
    const size_t bh = static_cast<size_t>(I.batch_rows) * I.hidden_dim * sizeof(__nv_bfloat16);
    const size_t pos_bytes = static_cast<size_t>(I.batch_rows) * I.n_heads * sizeof(int32_t);

    I.off_norm_buf    = 0;
    I.off_mod_buf     = align_up(I.off_norm_buf + bh, ALIGN);
    I.off_Q           = align_up(I.off_mod_buf  + bh, ALIGN);
    I.off_K           = align_up(I.off_Q        + bh, ALIGN);
    I.off_V           = align_up(I.off_K        + bh, ALIGN);
    I.off_attn_out    = align_up(I.off_V        + bh, ALIGN);
    I.off_resid_buf   = align_up(I.off_attn_out + bh, ALIGN);
    I.off_pos_ids     = align_up(I.off_resid_buf + bh, ALIGN);
    I.off_lin_q_ws    = align_up(I.off_pos_ids   + pos_bytes, ALIGN);
    I.off_lin_k_ws    = align_up(I.off_lin_q_ws    + I.lin_q->workspace_size_bytes(),    ALIGN);
    I.off_lin_v_ws    = align_up(I.off_lin_k_ws    + I.lin_k->workspace_size_bytes(),    ALIGN);
    I.off_lin_proj_ws = align_up(I.off_lin_v_ws    + I.lin_v->workspace_size_bytes(),    ALIGN);
    I.off_mlp_ws      = align_up(I.off_lin_proj_ws + I.lin_proj->workspace_size_bytes(), ALIGN);
    I.total_ws        = I.off_mlp_ws + I.mlp->workspace_size_bytes();

    I.valid = true;
}

MMDiTBlock::~MMDiTBlock() {
    if (impl_) {
        if (impl_->d_W_norm1) cudaFree(impl_->d_W_norm1);
        if (impl_->d_W_norm2) cudaFree(impl_->d_W_norm2);
    }
}

bool        MMDiTBlock::ok()         const { return impl_ && impl_->valid; }
const char* MMDiTBlock::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t MMDiTBlock::workspace_size_bytes() const {
    return impl_ ? impl_->total_ws : 0;
}

bool MMDiTBlock::forward(void* x_bf16,
                         const Modulation& mod,
                         const float* rope_cos, const float* rope_sin,
                         void* workspace, size_t workspace_size,
                         cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.total_ws) { I.err = "workspace too small"; return false; }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* norm_buf  = ws + I.off_norm_buf;
    void* mod_buf   = ws + I.off_mod_buf;
    void* Q         = ws + I.off_Q;
    void* K         = ws + I.off_K;
    void* V         = ws + I.off_V;
    void* attn_out  = ws + I.off_attn_out;
    void* resid_buf = ws + I.off_resid_buf;
    int32_t* pos_ids = reinterpret_cast<int32_t*>(ws + I.off_pos_ids);

    void* lin_q_ws    = ws + I.off_lin_q_ws;
    void* lin_k_ws    = ws + I.off_lin_k_ws;
    void* lin_v_ws    = ws + I.off_lin_v_ws;
    void* lin_proj_ws = ws + I.off_lin_proj_ws;
    void* mlp_ws      = ws + I.off_mlp_ws;

    // Build per-(row, head) pos_ids once for both Q and K RoPE calls.
    {
        const int n = I.batch_rows * I.n_heads;
        const int block = 256;
        fill_pos_ids_expanded<<<(n + block - 1) / block, block, 0, stream>>>(
            pos_ids, I.batch_rows, I.n_heads, I.seq);
        if (cudaPeekAtLastError() != cudaSuccess) { I.err = "pos_ids launch"; return false; }
    }

    // ---- Attention sub-block ----
    if (!rmsnorm_bf16(x_bf16, I.d_W_norm1, norm_buf,
                      I.batch_rows, I.hidden_dim, I.rms_eps, stream)) {
        I.err = "rmsnorm1"; return false;
    }
    if (!modulate_bf16(norm_buf, mod_buf, mod.scale_attn, mod.shift_attn,
                       I.batch_rows, I.hidden_dim, stream)) {
        I.err = "modulate1"; return false;
    }
    if (!I.lin_q->forward(mod_buf, Q, lin_q_ws, I.lin_q->workspace_size_bytes(), stream))
        { I.err = std::string("lin_q: ") + I.lin_q->last_error(); return false; }
    if (!I.lin_k->forward(mod_buf, K, lin_k_ws, I.lin_k->workspace_size_bytes(), stream))
        { I.err = std::string("lin_k: ") + I.lin_k->last_error(); return false; }
    if (!I.lin_v->forward(mod_buf, V, lin_v_ws, I.lin_v->workspace_size_bytes(), stream))
        { I.err = std::string("lin_v: ") + I.lin_v->last_error(); return false; }

    // RoPE on Q and K in place. Treated as [batch_rows*n_heads, head_dim].
    if (!rope_inplace_bf16(Q, rope_cos, rope_sin, pos_ids,
                            I.batch_rows * I.n_heads, I.head_dim, I.seq, stream))
        { I.err = "rope Q"; return false; }
    if (!rope_inplace_bf16(K, rope_cos, rope_sin, pos_ids,
                            I.batch_rows * I.n_heads, I.head_dim, I.seq, stream))
        { I.err = "rope K"; return false; }

    if (!I.attn->forward(Q, K, V, attn_out, stream))
        { I.err = std::string("attn: ") + I.attn->last_error(); return false; }
    if (!I.lin_proj->forward(attn_out, resid_buf, lin_proj_ws,
                              I.lin_proj->workspace_size_bytes(), stream))
        { I.err = std::string("lin_proj: ") + I.lin_proj->last_error(); return false; }
    if (!gated_residual_bf16(x_bf16, resid_buf, mod.gate_attn,
                              I.batch_rows, I.hidden_dim, stream))
        { I.err = "gated_res attn"; return false; }

    // ---- MLP sub-block ----
    if (!rmsnorm_bf16(x_bf16, I.d_W_norm2, norm_buf,
                      I.batch_rows, I.hidden_dim, I.rms_eps, stream))
        { I.err = "rmsnorm2"; return false; }
    if (!modulate_bf16(norm_buf, mod_buf, mod.scale_mlp, mod.shift_mlp,
                       I.batch_rows, I.hidden_dim, stream))
        { I.err = "modulate2"; return false; }
    if (!I.mlp->forward(mod_buf, resid_buf, mlp_ws,
                        I.mlp->workspace_size_bytes(), stream))
        { I.err = std::string("mlp: ") + I.mlp->last_error(); return false; }
    if (!gated_residual_bf16(x_bf16, resid_buf, mod.gate_mlp,
                              I.batch_rows, I.hidden_dim, stream))
        { I.err = "gated_res mlp"; return false; }

    return true;
}

} // namespace f2k::cuda
