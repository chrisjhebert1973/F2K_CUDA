// DoubleStreamBlock — orchestrates 12 Linears + JointAttention + the
// pre/post-attention bookkeeping for FLUX.2-klein's dual-stream layer.

#include "backend/cuda/double_stream_block.h"

#include "backend/cuda/linear.h"
#include "backend/cuda/joint_attention.h"
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

// pos_ids for RoPE on the image stream — repeat per-row seq idx across heads.
__global__ void fill_pos_ids_expanded(int32_t* out, int batch_rows, int n_heads, int seq) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= batch_rows * n_heads) return;
    const int r = idx / n_heads;
    out[idx] = r % seq;
}

} // anonymous namespace

namespace f2k::cuda {

struct DoubleStreamBlock::Impl {
    int batch = 0, seq_img = 0, seq_txt = 0;
    int n_heads = 0, head_dim = 0, ffn = 0;
    int rows_img = 0, rows_txt = 0;
    int hidden = 0;
    float rms_eps = 1e-6f;

    bool valid = false;
    std::string err;

    // ---- Image stream ----
    std::unique_ptr<Linear> lin_q, lin_k, lin_v;
    std::unique_ptr<Linear> lin_out;
    std::unique_ptr<Linear> lin_ff_in, lin_ff_out;
    void* d_norm_q = nullptr;
    void* d_norm_k = nullptr;

    // ---- Text stream ----
    std::unique_ptr<Linear> lin_add_q, lin_add_k, lin_add_v;
    std::unique_ptr<Linear> lin_add_out;
    std::unique_ptr<Linear> lin_ff_ctx_in, lin_ff_ctx_out;
    void* d_norm_added_q = nullptr;
    void* d_norm_added_k = nullptr;

    // ---- Shared ----
    std::unique_ptr<JointAttention> jattn;
    void* d_ones = nullptr;                  // [hidden] all 1.0 for pre-mod RMSNorm

    // ---- Workspace plan ----
    // Per-stream buffers (image and text get their own copies).
    struct StreamWS {
        size_t off_norm = 0;
        size_t off_mod  = 0;
        size_t off_q    = 0;
        size_t off_k    = 0;
        size_t off_v    = 0;
        size_t off_attn = 0;
        size_t off_proj = 0;
        size_t off_norm2 = 0;
        size_t off_mod2  = 0;
        size_t off_ff_fused = 0;
        size_t off_gate = 0;
        size_t off_up   = 0;
        size_t off_act  = 0;
        size_t off_ff_out_buf = 0;
        size_t off_lin_q_ws = 0, off_lin_k_ws = 0, off_lin_v_ws = 0;
        size_t off_lin_out_ws = 0;
        size_t off_lin_ff_in_ws = 0, off_lin_ff_out_ws = 0;
    };
    StreamWS img_ws, txt_ws;

    // Cross-stream: image + text pos_ids, JointAttention workspace.
    size_t off_pos_ids = 0;
    size_t off_pos_ids_txt = 0;
    size_t off_joint_ws = 0;

    size_t total_ws = 0;
};

DoubleStreamBlock::DoubleStreamBlock(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.batch    = cfg.batch;
    I.seq_img  = cfg.seq_img;
    I.seq_txt  = cfg.seq_txt;
    I.n_heads  = cfg.n_heads;
    I.head_dim = cfg.head_dim;
    I.ffn      = cfg.ffn_dim;
    I.rms_eps  = cfg.rms_eps;
    I.rows_img = cfg.batch * cfg.seq_img;
    I.rows_txt = cfg.batch * cfg.seq_txt;
    I.hidden   = cfg.n_heads * cfg.head_dim;

    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (I.batch <= 0 || I.seq_img <= 0 || I.seq_txt <= 0 ||
        I.n_heads <= 0 || I.head_dim <= 0 || I.ffn <= 0)
        { fail("DoubleStreamBlock: zero/negative dim"); return; }
    if (I.rows_img % 128 != 0 || I.rows_txt % 128 != 0)
        { fail("batch*seq_{img,txt} must be %% 128"); return; }
    if (I.hidden % 64 != 0)  { fail("hidden must be %% 64"); return; }
    if (I.ffn    % 128 != 0) { fail("ffn must be %% 128"); return; }
    if (I.head_dim & 1)      { fail("head_dim must be even"); return; }

    // Validate all weight pointers.
    const PreQuantNVFP4* req[] = {
        cfg.W_q, cfg.W_k, cfg.W_v, cfg.W_out, cfg.W_ff_in, cfg.W_ff_out,
        cfg.W_add_q, cfg.W_add_k, cfg.W_add_v, cfg.W_add_out,
        cfg.W_ff_ctx_in, cfg.W_ff_ctx_out };
    for (auto* p : req) if (!p) { fail("missing PreQuantNVFP4 weight"); return; }
    if (!cfg.norm_q_bf16 || !cfg.norm_k_bf16 ||
        !cfg.norm_added_q_bf16 || !cfg.norm_added_k_bf16)
        { fail("missing norm gain"); return; }

    // ---- Build the 12 Linears + 1 JointAttention. ----
    auto make_lin = [&](int rows, int K, int N, const PreQuantNVFP4* W) {
        Linear::Config c;
        c.batch_rows   = rows;
        c.in_features  = K;
        c.out_features = N;
        set_linear_weight(c, W);
        return std::make_unique<Linear>(c);
    };

    // Image
    I.lin_q   = make_lin(I.rows_img, I.hidden, I.hidden, cfg.W_q);
    I.lin_k   = make_lin(I.rows_img, I.hidden, I.hidden, cfg.W_k);
    I.lin_v   = make_lin(I.rows_img, I.hidden, I.hidden, cfg.W_v);
    I.lin_out = make_lin(I.rows_img, I.hidden, I.hidden, cfg.W_out);
    I.lin_ff_in  = make_lin(I.rows_img, I.hidden, 2 * I.ffn, cfg.W_ff_in);
    I.lin_ff_out = make_lin(I.rows_img, I.ffn,     I.hidden, cfg.W_ff_out);
    // Text
    I.lin_add_q   = make_lin(I.rows_txt, I.hidden, I.hidden, cfg.W_add_q);
    I.lin_add_k   = make_lin(I.rows_txt, I.hidden, I.hidden, cfg.W_add_k);
    I.lin_add_v   = make_lin(I.rows_txt, I.hidden, I.hidden, cfg.W_add_v);
    I.lin_add_out = make_lin(I.rows_txt, I.hidden, I.hidden, cfg.W_add_out);
    I.lin_ff_ctx_in  = make_lin(I.rows_txt, I.hidden, 2 * I.ffn, cfg.W_ff_ctx_in);
    I.lin_ff_ctx_out = make_lin(I.rows_txt, I.ffn,     I.hidden, cfg.W_ff_ctx_out);

    auto check = [&](const std::unique_ptr<Linear>& l, const char* name) {
        if (!l->ok()) { fail(std::string(name) + ": " + l->last_error()); return false; }
        return true;
    };
    if (!check(I.lin_q,        "lin_q"))        return;
    if (!check(I.lin_k,        "lin_k"))        return;
    if (!check(I.lin_v,        "lin_v"))        return;
    if (!check(I.lin_out,      "lin_out"))      return;
    if (!check(I.lin_ff_in,    "lin_ff_in"))    return;
    if (!check(I.lin_ff_out,   "lin_ff_out"))   return;
    if (!check(I.lin_add_q,    "lin_add_q"))    return;
    if (!check(I.lin_add_k,    "lin_add_k"))    return;
    if (!check(I.lin_add_v,    "lin_add_v"))    return;
    if (!check(I.lin_add_out,  "lin_add_out"))  return;
    if (!check(I.lin_ff_ctx_in, "lin_ff_ctx_in"))  return;
    if (!check(I.lin_ff_ctx_out,"lin_ff_ctx_out")) return;

    {
        JointAttention::Config jc{};
        jc.batch    = I.batch;
        jc.seq_txt  = I.seq_txt;
        jc.seq_img  = I.seq_img;
        jc.n_heads  = I.n_heads;
        jc.head_dim = I.head_dim;
        I.jattn = std::make_unique<JointAttention>(jc);
        if (!I.jattn->ok()) { fail(std::string("jattn: ") + I.jattn->last_error()); return; }
    }

    // Norm gains + ones → device.
    const size_t gain_bytes = static_cast<size_t>(I.head_dim) * sizeof(__nv_bfloat16);
    if (cudaMalloc(&I.d_norm_q,       gain_bytes) != cudaSuccess) { fail("malloc"); return; }
    if (cudaMalloc(&I.d_norm_k,       gain_bytes) != cudaSuccess) { fail("malloc"); return; }
    if (cudaMalloc(&I.d_norm_added_q, gain_bytes) != cudaSuccess) { fail("malloc"); return; }
    if (cudaMalloc(&I.d_norm_added_k, gain_bytes) != cudaSuccess) { fail("malloc"); return; }
    cudaMemcpy(I.d_norm_q,       cfg.norm_q_bf16,       gain_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(I.d_norm_k,       cfg.norm_k_bf16,       gain_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(I.d_norm_added_q, cfg.norm_added_q_bf16, gain_bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(I.d_norm_added_k, cfg.norm_added_k_bf16, gain_bytes, cudaMemcpyHostToDevice);
    {
        std::vector<__nv_bfloat16> ones(I.hidden, __float2bfloat16(1.0f));
        const size_t b = ones.size() * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_ones, b) != cudaSuccess) { fail("malloc ones"); return; }
        cudaMemcpy(I.d_ones, ones.data(), b, cudaMemcpyHostToDevice);
    }

    // ---- Workspace plan ----
    auto plan_stream = [&](Impl::StreamWS& s, int rows, size_t& cursor,
                            const std::unique_ptr<Linear>& Q,
                            const std::unique_ptr<Linear>& K,
                            const std::unique_ptr<Linear>& V,
                            const std::unique_ptr<Linear>& OUT,
                            const std::unique_ptr<Linear>& FFIN,
                            const std::unique_ptr<Linear>& FFOUT) {
        const size_t bh        = static_cast<size_t>(rows) * I.hidden  * sizeof(__nv_bfloat16);
        const size_t bf        = static_cast<size_t>(rows) * I.ffn     * sizeof(__nv_bfloat16);
        const size_t b_fused   = static_cast<size_t>(rows) * 2 * I.ffn * sizeof(__nv_bfloat16);

        cursor = align_up(cursor, ALIGN); s.off_norm        = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_mod         = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_q           = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_k           = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_v           = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_attn        = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_proj        = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_norm2       = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_mod2        = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_ff_fused    = cursor; cursor += b_fused;
        cursor = align_up(cursor, ALIGN); s.off_gate        = cursor; cursor += bf;
        cursor = align_up(cursor, ALIGN); s.off_up          = cursor; cursor += bf;
        cursor = align_up(cursor, ALIGN); s.off_act         = cursor; cursor += bf;
        cursor = align_up(cursor, ALIGN); s.off_ff_out_buf  = cursor; cursor += bh;
        cursor = align_up(cursor, ALIGN); s.off_lin_q_ws    = cursor; cursor += Q->workspace_size_bytes();
        cursor = align_up(cursor, ALIGN); s.off_lin_k_ws    = cursor; cursor += K->workspace_size_bytes();
        cursor = align_up(cursor, ALIGN); s.off_lin_v_ws    = cursor; cursor += V->workspace_size_bytes();
        cursor = align_up(cursor, ALIGN); s.off_lin_out_ws  = cursor; cursor += OUT->workspace_size_bytes();
        cursor = align_up(cursor, ALIGN); s.off_lin_ff_in_ws  = cursor; cursor += FFIN->workspace_size_bytes();
        cursor = align_up(cursor, ALIGN); s.off_lin_ff_out_ws = cursor; cursor += FFOUT->workspace_size_bytes();
    };

    size_t cursor = 0;
    plan_stream(I.img_ws, I.rows_img, cursor,
                I.lin_q, I.lin_k, I.lin_v, I.lin_out, I.lin_ff_in, I.lin_ff_out);
    plan_stream(I.txt_ws, I.rows_txt, cursor,
                I.lin_add_q, I.lin_add_k, I.lin_add_v, I.lin_add_out,
                I.lin_ff_ctx_in, I.lin_ff_ctx_out);

    cursor = align_up(cursor, ALIGN);
    I.off_pos_ids = cursor;
    cursor += static_cast<size_t>(I.rows_img) * I.n_heads * sizeof(int32_t);

    cursor = align_up(cursor, ALIGN);
    I.off_pos_ids_txt = cursor;
    cursor += static_cast<size_t>(I.rows_txt) * I.n_heads * sizeof(int32_t);

    cursor = align_up(cursor, ALIGN);
    I.off_joint_ws = cursor;
    cursor += I.jattn->workspace_size_bytes();

    I.total_ws = cursor;
    I.valid = true;
}

DoubleStreamBlock::~DoubleStreamBlock() {
    if (impl_) {
        if (impl_->d_norm_q)        cudaFree(impl_->d_norm_q);
        if (impl_->d_norm_k)        cudaFree(impl_->d_norm_k);
        if (impl_->d_norm_added_q)  cudaFree(impl_->d_norm_added_q);
        if (impl_->d_norm_added_k)  cudaFree(impl_->d_norm_added_k);
        if (impl_->d_ones)          cudaFree(impl_->d_ones);
    }
}

bool        DoubleStreamBlock::ok()         const { return impl_ && impl_->valid; }
const char* DoubleStreamBlock::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t DoubleStreamBlock::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

bool DoubleStreamBlock::forward(void* img, void* txt,
                                const Modulation& mod,
                                const float* rope_cos_img, const float* rope_sin_img,
                                const float* rope_cos_txt, const float* rope_sin_txt,
                                void* workspace, size_t workspace_size,
                                cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.total_ws) { I.err = "workspace too small"; return false; }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    auto p = [&](size_t off) -> void* { return ws + off; };

    // Image-stream pos_ids.
    int32_t* pos_ids = reinterpret_cast<int32_t*>(p(I.off_pos_ids));
    {
        const int n = I.rows_img * I.n_heads;
        const int blk = 256;
        fill_pos_ids_expanded<<<(n + blk - 1)/blk, blk, 0, stream>>>(
            pos_ids, I.rows_img, I.n_heads, I.seq_img);
        if (cudaPeekAtLastError() != cudaSuccess) { I.err = "pos_ids launch"; return false; }
    }
    // Text-stream pos_ids (same per-row=>token mapping, just for txt rows).
    int32_t* pos_ids_txt = reinterpret_cast<int32_t*>(p(I.off_pos_ids_txt));
    {
        const int n = I.rows_txt * I.n_heads;
        const int blk = 256;
        fill_pos_ids_expanded<<<(n + blk - 1)/blk, blk, 0, stream>>>(
            pos_ids_txt, I.rows_txt, I.n_heads, I.seq_txt);
        if (cudaPeekAtLastError() != cudaSuccess) { I.err = "pos_ids_txt launch"; return false; }
    }

    // ===================== ATTENTION SUB-BLOCK =====================

    // Pre-attention norms + modulate for both streams.
    if (!layernorm_bf16(img, p(I.img_ws.off_norm), I.rows_img, I.hidden, I.rms_eps, stream))
        { I.err = "img rmsnorm1"; return false; }
    if (!modulate_bf16(p(I.img_ws.off_norm), p(I.img_ws.off_mod),
                       mod.img_scale_attn, mod.img_shift_attn,
                       I.rows_img, I.hidden, stream))
        { I.err = "img modulate1"; return false; }
    if (!layernorm_bf16(txt, p(I.txt_ws.off_norm), I.rows_txt, I.hidden, I.rms_eps, stream))
        { I.err = "txt rmsnorm1"; return false; }
    if (!modulate_bf16(p(I.txt_ws.off_norm), p(I.txt_ws.off_mod),
                       mod.txt_scale_attn, mod.txt_shift_attn,
                       I.rows_txt, I.hidden, stream))
        { I.err = "txt modulate1"; return false; }

    // QKV projections.
    auto run_lin = [&](const std::unique_ptr<Linear>& l, const void* in, void* out,
                       size_t ws_off, const char* name) {
        if (!l->forward(in, out, p(ws_off), l->workspace_size_bytes(), stream)) {
            I.err = std::string(name) + ": " + l->last_error();
            return false;
        }
        return true;
    };
    if (!run_lin(I.lin_q,     p(I.img_ws.off_mod), p(I.img_ws.off_q), I.img_ws.off_lin_q_ws, "lin_q"))     return false;
    if (!run_lin(I.lin_k,     p(I.img_ws.off_mod), p(I.img_ws.off_k), I.img_ws.off_lin_k_ws, "lin_k"))     return false;
    if (!run_lin(I.lin_v,     p(I.img_ws.off_mod), p(I.img_ws.off_v), I.img_ws.off_lin_v_ws, "lin_v"))     return false;
    if (!run_lin(I.lin_add_q, p(I.txt_ws.off_mod), p(I.txt_ws.off_q), I.txt_ws.off_lin_q_ws, "lin_add_q")) return false;
    if (!run_lin(I.lin_add_k, p(I.txt_ws.off_mod), p(I.txt_ws.off_k), I.txt_ws.off_lin_k_ws, "lin_add_k")) return false;
    if (!run_lin(I.lin_add_v, p(I.txt_ws.off_mod), p(I.txt_ws.off_v), I.txt_ws.off_lin_v_ws, "lin_add_v")) return false;

    // Per-head QK norms (in-place).
    if (!rmsnorm_bf16(p(I.img_ws.off_q), I.d_norm_q, p(I.img_ws.off_q),
                       I.rows_img * I.n_heads, I.head_dim, I.rms_eps, stream))
        { I.err = "img qknorm Q"; return false; }
    if (!rmsnorm_bf16(p(I.img_ws.off_k), I.d_norm_k, p(I.img_ws.off_k),
                       I.rows_img * I.n_heads, I.head_dim, I.rms_eps, stream))
        { I.err = "img qknorm K"; return false; }
    if (!rmsnorm_bf16(p(I.txt_ws.off_q), I.d_norm_added_q, p(I.txt_ws.off_q),
                       I.rows_txt * I.n_heads, I.head_dim, I.rms_eps, stream))
        { I.err = "txt qknorm Q"; return false; }
    if (!rmsnorm_bf16(p(I.txt_ws.off_k), I.d_norm_added_k, p(I.txt_ws.off_k),
                       I.rows_txt * I.n_heads, I.head_dim, I.rms_eps, stream))
        { I.err = "txt qknorm K"; return false; }

    // 4-axis RoPE on Q, K for both streams. Text positions are (0, 0, 0, l) so
    // only the last 32 dims of head_dim actually rotate; the first 96 are
    // identity. We still apply the kernel so behavior matches diffusers exactly.
    if (!rope_4axis_inplace_bf16(p(I.img_ws.off_q),
                                  rope_cos_img, rope_sin_img, pos_ids,
                                  I.rows_img * I.n_heads, I.head_dim, I.seq_img, stream))
        { I.err = "rope img Q"; return false; }
    if (!rope_4axis_inplace_bf16(p(I.img_ws.off_k),
                                  rope_cos_img, rope_sin_img, pos_ids,
                                  I.rows_img * I.n_heads, I.head_dim, I.seq_img, stream))
        { I.err = "rope img K"; return false; }
    if (!rope_4axis_inplace_bf16(p(I.txt_ws.off_q),
                                  rope_cos_txt, rope_sin_txt, pos_ids_txt,
                                  I.rows_txt * I.n_heads, I.head_dim, I.seq_txt, stream))
        { I.err = "rope txt Q"; return false; }
    if (!rope_4axis_inplace_bf16(p(I.txt_ws.off_k),
                                  rope_cos_txt, rope_sin_txt, pos_ids_txt,
                                  I.rows_txt * I.n_heads, I.head_dim, I.seq_txt, stream))
        { I.err = "rope txt K"; return false; }

    // Joint attention: (txt_q, txt_k, txt_v, img_q, img_k, img_v) → (txt_out, img_out).
    if (!I.jattn->forward(p(I.txt_ws.off_q), p(I.txt_ws.off_k), p(I.txt_ws.off_v),
                          p(I.img_ws.off_q), p(I.img_ws.off_k), p(I.img_ws.off_v),
                          p(I.txt_ws.off_attn), p(I.img_ws.off_attn),
                          p(I.off_joint_ws), I.jattn->workspace_size_bytes(), stream))
        { I.err = std::string("jattn: ") + I.jattn->last_error(); return false; }

    // Output projections.
    if (!run_lin(I.lin_out,     p(I.img_ws.off_attn), p(I.img_ws.off_proj),
                                  I.img_ws.off_lin_out_ws, "lin_out"))     return false;
    if (!run_lin(I.lin_add_out, p(I.txt_ws.off_attn), p(I.txt_ws.off_proj),
                                  I.txt_ws.off_lin_out_ws, "lin_add_out")) return false;

    // Gated residuals.
    if (!gated_residual_bf16(img, p(I.img_ws.off_proj), mod.img_gate_attn,
                              I.rows_img, I.hidden, stream))
        { I.err = "img gated_res attn"; return false; }
    if (!gated_residual_bf16(txt, p(I.txt_ws.off_proj), mod.txt_gate_attn,
                              I.rows_txt, I.hidden, stream))
        { I.err = "txt gated_res attn"; return false; }

    // ===================== MLP SUB-BLOCK =====================

    if (!layernorm_bf16(img, p(I.img_ws.off_norm2), I.rows_img, I.hidden, I.rms_eps, stream))
        { I.err = "img rmsnorm2"; return false; }
    if (!modulate_bf16(p(I.img_ws.off_norm2), p(I.img_ws.off_mod2),
                       mod.img_scale_mlp, mod.img_shift_mlp, I.rows_img, I.hidden, stream))
        { I.err = "img modulate2"; return false; }
    if (!layernorm_bf16(txt, p(I.txt_ws.off_norm2), I.rows_txt, I.hidden, I.rms_eps, stream))
        { I.err = "txt rmsnorm2"; return false; }
    if (!modulate_bf16(p(I.txt_ws.off_norm2), p(I.txt_ws.off_mod2),
                       mod.txt_scale_mlp, mod.txt_shift_mlp, I.rows_txt, I.hidden, stream))
        { I.err = "txt modulate2"; return false; }

    // GeGLU MLP per stream.
    if (!run_lin(I.lin_ff_in, p(I.img_ws.off_mod2), p(I.img_ws.off_ff_fused),
                                I.img_ws.off_lin_ff_in_ws, "lin_ff_in")) return false;
    if (!split_half_bf16(p(I.img_ws.off_ff_fused), p(I.img_ws.off_gate), p(I.img_ws.off_up),
                          I.rows_img, I.ffn, stream))
        { I.err = "img split_half"; return false; }
    if (!silu_mul_bf16(p(I.img_ws.off_gate), p(I.img_ws.off_up), p(I.img_ws.off_act),
                        static_cast<size_t>(I.rows_img) * I.ffn, stream))
        { I.err = "img silu_mul"; return false; }
    if (!run_lin(I.lin_ff_out, p(I.img_ws.off_act), p(I.img_ws.off_ff_out_buf),
                                 I.img_ws.off_lin_ff_out_ws, "lin_ff_out")) return false;

    if (!run_lin(I.lin_ff_ctx_in, p(I.txt_ws.off_mod2), p(I.txt_ws.off_ff_fused),
                                    I.txt_ws.off_lin_ff_in_ws, "lin_ff_ctx_in")) return false;
    if (!split_half_bf16(p(I.txt_ws.off_ff_fused), p(I.txt_ws.off_gate), p(I.txt_ws.off_up),
                          I.rows_txt, I.ffn, stream))
        { I.err = "txt split_half"; return false; }
    if (!silu_mul_bf16(p(I.txt_ws.off_gate), p(I.txt_ws.off_up), p(I.txt_ws.off_act),
                        static_cast<size_t>(I.rows_txt) * I.ffn, stream))
        { I.err = "txt silu_mul"; return false; }
    if (!run_lin(I.lin_ff_ctx_out, p(I.txt_ws.off_act), p(I.txt_ws.off_ff_out_buf),
                                     I.txt_ws.off_lin_ff_out_ws, "lin_ff_ctx_out")) return false;

    // Final gated residuals.
    if (!gated_residual_bf16(img, p(I.img_ws.off_ff_out_buf), mod.img_gate_mlp,
                              I.rows_img, I.hidden, stream))
        { I.err = "img gated_res mlp"; return false; }
    if (!gated_residual_bf16(txt, p(I.txt_ws.off_ff_out_buf), mod.txt_gate_mlp,
                              I.rows_txt, I.hidden, stream))
        { I.err = "txt gated_res mlp"; return false; }

    return true;
}

} // namespace f2k::cuda
