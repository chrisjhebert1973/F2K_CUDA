// SingleStreamBlock — composes Linear (NVFP4) + RMSNorm + RoPE + Attention +
// silu_mul + split/concat kernels for the FLUX.2-klein single-stream layer.

#include "backend/cuda/single_stream_block.h"

#include "backend/cuda/linear.h"
#include "backend/cuda/attention.h"
#include "backend/cuda/kernels/rmsnorm.h"
#include "backend/cuda/kernels/rope.h"
#include "backend/cuda/kernels/rope_4axis.h"
#include "backend/cuda/kernels/layernorm.h"
#include "backend/cuda/kernels/silu_mul.h"
#include "backend/cuda/kernels/modulation.h"
#include "backend/cuda/kernels/qkv_mlp_split.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

__global__ void fill_pos_ids_expanded(int32_t* out, int batch_rows, int n_heads, int seq) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_rows * n_heads) return;
    const int r = idx / n_heads;
    out[idx] = r % seq;
}

} // anonymous namespace

namespace f2k::cuda {

struct SingleStreamBlock::Impl {
    int batch = 0, seq = 0, n_heads = 0, head_dim = 0, ffn = 0;
    int batch_rows = 0, hidden = 0;
    float rms_eps = 1e-6f;

    bool valid = false;
    std::string err;

    std::unique_ptr<Linear>    lin_qkv;
    std::unique_ptr<Linear>    lin_out;
    std::unique_ptr<Attention> attn;

    // Device-resident BF16 norm gains.
    void* d_norm_q   = nullptr;     // [head_dim]
    void* d_norm_k   = nullptr;     // [head_dim]
    void* d_ones     = nullptr;     // [hidden] all 1.0, for pre-mod RMSNorm

    // Workspace partitioning.
    size_t off_norm_buf = 0;
    size_t off_mod_buf  = 0;
    size_t off_fused    = 0;
    size_t off_q        = 0;
    size_t off_k        = 0;
    size_t off_v        = 0;
    size_t off_gate     = 0;
    size_t off_up       = 0;
    size_t off_attn_out = 0;
    size_t off_mlp_act  = 0;
    size_t off_concat   = 0;
    size_t off_delta    = 0;
    size_t off_pos_ids  = 0;
    size_t off_lin_qkv_ws = 0;
    size_t off_lin_out_ws = 0;
    size_t total_ws = 0;
};

SingleStreamBlock::SingleStreamBlock(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.batch      = cfg.batch;
    I.seq        = cfg.seq;
    I.n_heads    = cfg.n_heads;
    I.head_dim   = cfg.head_dim;
    I.ffn        = cfg.ffn_dim;
    I.rms_eps    = cfg.rms_eps;
    I.batch_rows = cfg.batch * cfg.seq;
    I.hidden     = cfg.n_heads * cfg.head_dim;

    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (I.batch <= 0 || I.seq <= 0 || I.n_heads <= 0 || I.head_dim <= 0 || I.ffn <= 0)
        { fail("SingleStreamBlock: zero/negative dim"); return; }
    if (I.batch_rows % 128 != 0) { fail("batch*seq must be %% 128 (Linear M)"); return; }
    if (I.hidden % 64 != 0)      { fail("hidden must be %% 64");                 return; }
    if (I.ffn    % 128 != 0)     { fail("ffn must be %% 128");                   return; }
    if (I.head_dim & 1)          { fail("head_dim must be even (RoPE)");         return; }
    if (!cfg.W_qkv_mlp_proj || !cfg.W_out || !cfg.norm_q_bf16 || !cfg.norm_k_bf16)
        { fail("missing weight pointer"); return; }

    // Verify expected fused-linear shapes.
    const int fused_out = 3 * I.hidden + 2 * I.ffn;          // 36864 for klein
    const int concat_in = I.hidden + I.ffn;                  // 16384
    if (cfg.W_qkv_mlp_proj->N != fused_out || cfg.W_qkv_mlp_proj->K != I.hidden) {
        fail("W_qkv_mlp_proj shape: expected [" +
             std::to_string(fused_out) + ", " + std::to_string(I.hidden) + "]");
        return;
    }
    if (cfg.W_out->N != I.hidden || cfg.W_out->K != concat_in) {
        fail("W_out shape: expected [" +
             std::to_string(I.hidden) + ", " + std::to_string(concat_in) + "]");
        return;
    }

    // Linears.
    {
        Linear::Config c;
        c.batch_rows   = I.batch_rows;
        c.in_features  = I.hidden;
        c.out_features = fused_out;
        set_linear_weight(c, cfg.W_qkv_mlp_proj);
        I.lin_qkv = std::make_unique<Linear>(c);
        if (!I.lin_qkv->ok()) { fail(std::string("lin_qkv: ") + I.lin_qkv->last_error()); return; }
    }
    {
        Linear::Config c;
        c.batch_rows   = I.batch_rows;
        c.in_features  = concat_in;
        c.out_features = I.hidden;
        set_linear_weight(c, cfg.W_out);
        I.lin_out = std::make_unique<Linear>(c);
        if (!I.lin_out->ok()) { fail(std::string("lin_out: ") + I.lin_out->last_error()); return; }
    }

    // Attention.
    Attention::Config ac{};
    ac.batch    = I.batch;
    ac.seq      = I.seq;
    ac.n_heads  = I.n_heads;
    ac.head_dim = I.head_dim;
    I.attn = std::make_unique<Attention>(ac);
    if (!I.attn->ok()) { fail(std::string("attn: ") + I.attn->last_error()); return; }

    // Norm gains → device.
    const size_t gain_bytes = static_cast<size_t>(I.head_dim) * sizeof(__nv_bfloat16);
    if (cudaMalloc(&I.d_norm_q, gain_bytes) != cudaSuccess) { fail("malloc norm_q"); return; }
    if (cudaMalloc(&I.d_norm_k, gain_bytes) != cudaSuccess) { fail("malloc norm_k"); return; }
    cudaMemcpy(I.d_norm_q, cfg.norm_q_bf16, gain_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(I.d_norm_k, cfg.norm_k_bf16, gain_bytes, cudaMemcpyHostToDevice);

    // ones-gain for pre-mod RMSNorm (statistical normalization, no affine).
    {
        std::vector<__nv_bfloat16> ones(I.hidden, __float2bfloat16(1.0f));
        const size_t b = ones.size() * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_ones, b) != cudaSuccess) { fail("malloc ones"); return; }
        cudaMemcpy(I.d_ones, ones.data(), b, cudaMemcpyHostToDevice);
    }

    // Workspace plan.
    const size_t bh        = static_cast<size_t>(I.batch_rows) * I.hidden * sizeof(__nv_bfloat16);
    const size_t bf        = static_cast<size_t>(I.batch_rows) * I.ffn    * sizeof(__nv_bfloat16);
    const size_t b_fused   = static_cast<size_t>(I.batch_rows) * fused_out * sizeof(__nv_bfloat16);
    const size_t b_concat  = static_cast<size_t>(I.batch_rows) * concat_in * sizeof(__nv_bfloat16);
    const size_t pos_bytes = static_cast<size_t>(I.batch_rows) * I.n_heads * sizeof(int32_t);

    I.off_norm_buf  = 0;
    I.off_mod_buf   = align_up(I.off_norm_buf + bh, ALIGN);
    I.off_fused     = align_up(I.off_mod_buf  + bh, ALIGN);
    I.off_q         = align_up(I.off_fused    + b_fused, ALIGN);
    I.off_k         = align_up(I.off_q        + bh, ALIGN);
    I.off_v         = align_up(I.off_k        + bh, ALIGN);
    I.off_gate      = align_up(I.off_v        + bh, ALIGN);
    I.off_up        = align_up(I.off_gate     + bf, ALIGN);
    I.off_attn_out  = align_up(I.off_up       + bf, ALIGN);
    I.off_mlp_act   = align_up(I.off_attn_out + bh, ALIGN);
    I.off_concat    = align_up(I.off_mlp_act  + bf, ALIGN);
    I.off_delta     = align_up(I.off_concat   + b_concat, ALIGN);
    I.off_pos_ids   = align_up(I.off_delta    + bh, ALIGN);
    I.off_lin_qkv_ws = align_up(I.off_pos_ids   + pos_bytes, ALIGN);
    I.off_lin_out_ws = align_up(I.off_lin_qkv_ws + I.lin_qkv->workspace_size_bytes(), ALIGN);
    I.total_ws       = I.off_lin_out_ws + I.lin_out->workspace_size_bytes();

    I.valid = true;
}

SingleStreamBlock::~SingleStreamBlock() {
    if (impl_) {
        if (impl_->d_norm_q) cudaFree(impl_->d_norm_q);
        if (impl_->d_norm_k) cudaFree(impl_->d_norm_k);
        if (impl_->d_ones)   cudaFree(impl_->d_ones);
    }
}

bool        SingleStreamBlock::ok()         const { return impl_ && impl_->valid; }
const char* SingleStreamBlock::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t SingleStreamBlock::workspace_size_bytes() const {
    return impl_ ? impl_->total_ws : 0;
}

bool SingleStreamBlock::forward(void* x_bf16,
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
    void* fused     = ws + I.off_fused;
    void* Q         = ws + I.off_q;
    void* K         = ws + I.off_k;
    void* V         = ws + I.off_v;
    void* gate_buf  = ws + I.off_gate;
    void* up_buf    = ws + I.off_up;
    void* attn_out  = ws + I.off_attn_out;
    void* mlp_act   = ws + I.off_mlp_act;
    void* concat    = ws + I.off_concat;
    void* delta     = ws + I.off_delta;
    int32_t* pos_ids = reinterpret_cast<int32_t*>(ws + I.off_pos_ids);
    void* lin_qkv_ws = ws + I.off_lin_qkv_ws;
    void* lin_out_ws = ws + I.off_lin_out_ws;

    const int rows = I.batch_rows;
    const int fused_out = 3 * I.hidden + 2 * I.ffn;
    const int concat_in = I.hidden + I.ffn;

    // pos_ids for RoPE: pos[r * H + h] = r % seq.
    {
        const int n = rows * I.n_heads;
        const int blk = 256;
        fill_pos_ids_expanded<<<(n + blk - 1)/blk, blk, 0, stream>>>(
            pos_ids, rows, I.n_heads, I.seq);
        if (cudaPeekAtLastError() != cudaSuccess) { I.err = "pos_ids launch"; return false; }
    }

    // 1. Pre-modulation RMSNorm (no affine — ones gain).
    if (!layernorm_bf16(x_bf16, norm_buf, rows, I.hidden, I.rms_eps, stream))
        { I.err = "rmsnorm pre-mod"; return false; }
    // 2. Modulate.
    if (!modulate_bf16(norm_buf, mod_buf, mod.scale, mod.shift, rows, I.hidden, stream))
        { I.err = "modulate"; return false; }
    // 3. Fused QKV+MLP projection.
    if (!I.lin_qkv->forward(mod_buf, fused, lin_qkv_ws,
                             I.lin_qkv->workspace_size_bytes(), stream))
        { I.err = std::string("lin_qkv: ") + I.lin_qkv->last_error(); return false; }
    // 4. Split into q, k, v, gate, up.
    if (!split_qkv_mlp_bf16(fused, Q, K, V, gate_buf, up_buf,
                             rows, I.hidden, I.ffn, stream))
        { I.err = "split"; return false; }
    (void)fused_out;

    // 5. Per-head QK RMSNorm. Treat as [rows*n_heads, head_dim].
    if (!rmsnorm_bf16(Q, I.d_norm_q, Q, rows * I.n_heads, I.head_dim, I.rms_eps, stream))
        { I.err = "rmsnorm Q"; return false; }
    if (!rmsnorm_bf16(K, I.d_norm_k, K, rows * I.n_heads, I.head_dim, I.rms_eps, stream))
        { I.err = "rmsnorm K"; return false; }
    // 6. 4-axis RoPE on Q and K (combined [txt || img] sequence).
    if (!rope_4axis_inplace_bf16(Q, rope_cos, rope_sin, pos_ids,
                                  rows * I.n_heads, I.head_dim, I.seq, stream))
        { I.err = "rope Q"; return false; }
    if (!rope_4axis_inplace_bf16(K, rope_cos, rope_sin, pos_ids,
                                  rows * I.n_heads, I.head_dim, I.seq, stream))
        { I.err = "rope K"; return false; }
    // 7. Attention.
    if (!I.attn->forward(Q, K, V, attn_out, stream))
        { I.err = std::string("attn: ") + I.attn->last_error(); return false; }
    // 8. silu(gate) * up.
    const size_t mlp_n = static_cast<size_t>(rows) * I.ffn;
    if (!silu_mul_bf16(gate_buf, up_buf, mlp_act, mlp_n, stream))
        { I.err = "silu_mul"; return false; }
    // 9. Concat [attn_out || mlp_act].
    if (!concat_attn_mlp_bf16(attn_out, mlp_act, concat, rows, I.hidden, I.ffn, stream))
        { I.err = "concat"; return false; }
    (void)concat_in;
    // 10. Output projection (NVFP4).
    if (!I.lin_out->forward(concat, delta, lin_out_ws,
                              I.lin_out->workspace_size_bytes(), stream))
        { I.err = std::string("lin_out: ") + I.lin_out->last_error(); return false; }
    // 11. Gated residual: x += gate * delta.
    if (!gated_residual_bf16(x_bf16, delta, mod.gate, rows, I.hidden, stream))
        { I.err = "gated_residual"; return false; }

    return true;
}

} // namespace f2k::cuda
