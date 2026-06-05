// FluxTransformer — composes every transformer component into one forward call.

#include "backend/cuda/flux_transformer.h"

#include "backend/cuda/embeddings.h"
#include "backend/cuda/modulation_mlp.h"
#include "backend/cuda/double_stream_block.h"
#include "backend/cuda/single_stream_block.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/kernels/seq_ops.h"
#include "backend/cuda/kernels/rope_4axis.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

// Helper: build a weight descriptor from a TensorView. For NVFP4 it carries the
// pre-quantized bits; for MXFP8 it carries the raw BF16 source (the TensorView
// must come from a BF16 F2K, and Linear quantizes to FP8 at construction).
f2k::cuda::PreQuantNVFP4 make_weight_view(const f2k::TensorView* t,
                                          f2k::cuda::Precision prec) {
    f2k::cuda::PreQuantNVFP4 r{};
    r.N = static_cast<int>(t->shape[0]);
    r.K = static_cast<int>(t->shape[1]);
    r.precision = prec;
    if (prec == f2k::cuda::Precision::MXFP8 && t->dtype != f2k::DType::F8_E4M3) {
        // BF16 source → Linear quantizes to MXFP8 at construction.
        r.bf16 = t->data;
    } else {
        // Pre-quantized bits (NVFP4 E2M1+E4M3, or on-disk MXFP8 E4M3+UE8M0).
        r.packed = t->data;
        r.scales = t->scales;
        r.microblock_size = t->microblock_size ? t->microblock_size
                                               : (prec == f2k::cuda::Precision::MXFP8 ? 32 : 16);
        r.tensor_scale = t->tensor_scale ? t->tensor_scale : 1.0f;
    }
    return r;
}

} // anonymous namespace

namespace f2k::cuda {

struct FluxTransformer::Impl {
    Config cfg{};
    bool   valid = false;
    std::string err;

    // PreQuantNVFP4 views (must outlive the sub-objects that reference them).
    PreQuantNVFP4 P_x_embedder, P_ctx_embedder, P_proj_out;
    PreQuantNVFP4 P_time_1, P_time_2, P_double_img, P_double_txt, P_single, P_norm_out;

    // Per-block weight views — built once at construction.
    std::vector<PreQuantNVFP4> dbl_q, dbl_k, dbl_v, dbl_out, dbl_ff_in, dbl_ff_out;
    std::vector<PreQuantNVFP4> dbl_add_q, dbl_add_k, dbl_add_v, dbl_add_out;
    std::vector<PreQuantNVFP4> dbl_ff_ctx_in, dbl_ff_ctx_out;
    std::vector<PreQuantNVFP4> sgl_qkv, sgl_out;

    // Component instances.
    std::unique_ptr<ImageEmbedder>    img_emb;
    std::unique_ptr<ContextEmbedder>  ctx_emb;
    std::unique_ptr<ModulationMLP>    mod_mlp;
    std::vector<std::unique_ptr<DoubleStreamBlock>> dbl_blocks;
    std::vector<std::unique_ptr<SingleStreamBlock>> sgl_blocks;
    std::unique_ptr<FinalProjection>  final_proj;

    // RoPE tables (device). 4-axis FLUX2 RoPE:
    //   dbl: separate img + txt tables (dbl-stream attention is per-stream).
    //   sgl: combined [txt || img] table for the joined sequence.
    float* d_rope_cos_img = nullptr;   // [seq_img,           head_dim/2]
    float* d_rope_sin_img = nullptr;
    float* d_rope_cos_txt = nullptr;   // [seq_txt,           head_dim/2]
    float* d_rope_sin_txt = nullptr;
    float* d_rope_cos_sgl = nullptr;   // [seq_txt + seq_img, head_dim/2]
    float* d_rope_sin_sgl = nullptr;

    // Workspace offsets.
    size_t off_img_hidden  = 0;        // [B*S_img, hidden]
    size_t off_txt_hidden  = 0;        // [B*S_txt, hidden]
    size_t off_combined    = 0;        // [B*(S_img+S_txt), hidden]
    size_t off_mod_ws      = 0;        // ModulationMLP::workspace (persistent)
    size_t off_img_emb_ws  = 0;
    size_t off_ctx_emb_ws  = 0;
    size_t off_dbl_blk_ws  = 0;
    size_t off_sgl_blk_ws  = 0;
    size_t off_final_ws    = 0;
    size_t total_ws = 0;
};

// ============================================================================
// Construction
// ============================================================================

FluxTransformer::FluxTransformer(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;
    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (!cfg.router) { fail("router is null"); return; }
    if (cfg.batch <= 0 || cfg.seq_img <= 0 || cfg.seq_txt <= 0)
        { fail("zero/negative batch or seq"); return; }
    const int hidden = cfg.n_heads * cfg.head_dim;
    const int rows_img = cfg.batch * cfg.seq_img;
    const int rows_txt = cfg.batch * cfg.seq_txt;
    const int rows_combined = cfg.batch * (cfg.seq_img + cfg.seq_txt);
    if (rows_img      % 128 != 0) { fail("batch*seq_img must be % 128"); return; }
    if (rows_txt      % 128 != 0) { fail("batch*seq_txt must be % 128"); return; }
    if (rows_combined % 128 != 0) { fail("batch*(seq_img+seq_txt) must be % 128"); return; }
    if (cfg.num_double_blocks != static_cast<int>(cfg.router->double_blocks().size())) {
        fail("num_double_blocks mismatch with router"); return;
    }
    if (cfg.num_single_blocks != static_cast<int>(cfg.router->single_blocks().size())) {
        fail("num_single_blocks mismatch with router"); return;
    }

    // Local view() binds the configured precision so all the call sites below
    // stay unchanged.
    auto view = [&](const f2k::TensorView* t) {
        return make_weight_view(t, cfg.precision);
    };

    const auto& g = cfg.router->globals();
    I.P_x_embedder   = view(g.x_embedder);
    I.P_ctx_embedder = view(g.context_embedder);
    I.P_proj_out     = view(g.proj_out);
    I.P_time_1       = view(g.time_embed_1);
    I.P_time_2       = view(g.time_embed_2);
    I.P_double_img   = view(g.double_mod_img);
    I.P_double_txt   = view(g.double_mod_txt);
    I.P_single       = view(g.single_mod);
    I.P_norm_out     = view(g.norm_out);

    // ---- Embeddings + Modulation MLP + Final projection ----
    {
        ImageEmbedder::Config c;
        c.batch_rows = rows_img; c.in_channels = cfg.in_channels;
        c.hidden_dim = hidden;   c.W = &I.P_x_embedder;
        I.img_emb = std::make_unique<ImageEmbedder>(c);
        if (!I.img_emb->ok()) { fail(std::string("img_emb: ") + I.img_emb->last_error()); return; }
    }
    {
        ContextEmbedder::Config c;
        c.batch_rows = rows_txt; c.t5_dim = cfg.t5_dim;
        c.hidden_dim = hidden;   c.W = &I.P_ctx_embedder;
        I.ctx_emb = std::make_unique<ContextEmbedder>(c);
        if (!I.ctx_emb->ok()) { fail(std::string("ctx_emb: ") + I.ctx_emb->last_error()); return; }
    }
    {
        ModulationMLP::Config c;
        c.hidden_dim = hidden;
        c.time_dim   = cfg.time_dim;
        c.W_time_1 = &I.P_time_1; c.W_time_2 = &I.P_time_2;
        c.W_double_img = &I.P_double_img; c.W_double_txt = &I.P_double_txt;
        c.W_single = &I.P_single; c.W_norm_out = &I.P_norm_out;
        I.mod_mlp = std::make_unique<ModulationMLP>(c);
        if (!I.mod_mlp->ok()) { fail(std::string("mod_mlp: ") + I.mod_mlp->last_error()); return; }
    }
    {
        FinalProjection::Config c;
        c.batch_rows  = rows_img;
        c.hidden_dim  = hidden;
        c.out_channels = cfg.in_channels;
        c.rms_eps     = cfg.rms_eps;
        c.W_proj_out  = &I.P_proj_out;
        I.final_proj = std::make_unique<FinalProjection>(c);
        if (!I.final_proj->ok()) { fail(std::string("final: ") + I.final_proj->last_error()); return; }
    }

    // ---- Build per-block weight views, then construct each block. ----
    I.dbl_q.reserve(cfg.num_double_blocks); I.dbl_k.reserve(cfg.num_double_blocks);
    I.dbl_v.reserve(cfg.num_double_blocks); I.dbl_out.reserve(cfg.num_double_blocks);
    I.dbl_ff_in.reserve(cfg.num_double_blocks); I.dbl_ff_out.reserve(cfg.num_double_blocks);
    I.dbl_add_q.reserve(cfg.num_double_blocks); I.dbl_add_k.reserve(cfg.num_double_blocks);
    I.dbl_add_v.reserve(cfg.num_double_blocks); I.dbl_add_out.reserve(cfg.num_double_blocks);
    I.dbl_ff_ctx_in.reserve(cfg.num_double_blocks); I.dbl_ff_ctx_out.reserve(cfg.num_double_blocks);
    I.dbl_blocks.reserve(cfg.num_double_blocks);

    for (int i = 0; i < cfg.num_double_blocks; ++i) {
        const auto& dt = cfg.router->double_blocks()[i];
        I.dbl_q.push_back(view(dt.to_q));
        I.dbl_k.push_back(view(dt.to_k));
        I.dbl_v.push_back(view(dt.to_v));
        I.dbl_out.push_back(view(dt.to_out));
        I.dbl_ff_in.push_back(view(dt.ff_in));
        I.dbl_ff_out.push_back(view(dt.ff_out));
        I.dbl_add_q.push_back(view(dt.add_q_proj));
        I.dbl_add_k.push_back(view(dt.add_k_proj));
        I.dbl_add_v.push_back(view(dt.add_v_proj));
        I.dbl_add_out.push_back(view(dt.to_add_out));
        I.dbl_ff_ctx_in.push_back(view(dt.ff_ctx_in));
        I.dbl_ff_ctx_out.push_back(view(dt.ff_ctx_out));

        DoubleStreamBlock::Config c;
        c.batch    = cfg.batch;
        c.seq_img  = cfg.seq_img;
        c.seq_txt  = cfg.seq_txt;
        c.n_heads  = cfg.n_heads;
        c.head_dim = cfg.head_dim;
        c.ffn_dim  = cfg.ffn_dim;
        c.rms_eps  = cfg.rms_eps;
        c.W_q   = &I.dbl_q.back();
        c.W_k   = &I.dbl_k.back();
        c.W_v   = &I.dbl_v.back();
        c.W_out = &I.dbl_out.back();
        c.W_ff_in  = &I.dbl_ff_in.back();
        c.W_ff_out = &I.dbl_ff_out.back();
        c.norm_q_bf16 = dt.norm_q->data;
        c.norm_k_bf16 = dt.norm_k->data;
        c.W_add_q   = &I.dbl_add_q.back();
        c.W_add_k   = &I.dbl_add_k.back();
        c.W_add_v   = &I.dbl_add_v.back();
        c.W_add_out = &I.dbl_add_out.back();
        c.W_ff_ctx_in  = &I.dbl_ff_ctx_in.back();
        c.W_ff_ctx_out = &I.dbl_ff_ctx_out.back();
        c.norm_added_q_bf16 = dt.norm_added_q->data;
        c.norm_added_k_bf16 = dt.norm_added_k->data;

        auto blk = std::make_unique<DoubleStreamBlock>(c);
        if (!blk->ok()) {
            fail("dbl block " + std::to_string(i) + ": " + blk->last_error());
            return;
        }
        I.dbl_blocks.push_back(std::move(blk));
    }

    I.sgl_qkv.reserve(cfg.num_single_blocks); I.sgl_out.reserve(cfg.num_single_blocks);
    I.sgl_blocks.reserve(cfg.num_single_blocks);
    for (int i = 0; i < cfg.num_single_blocks; ++i) {
        const auto& st = cfg.router->single_blocks()[i];
        I.sgl_qkv.push_back(view(st.to_qkv_mlp_proj));
        I.sgl_out.push_back(view(st.to_out));

        SingleStreamBlock::Config c;
        c.batch    = cfg.batch;
        c.seq      = cfg.seq_img + cfg.seq_txt;   // combined
        c.n_heads  = cfg.n_heads;
        c.head_dim = cfg.head_dim;
        c.ffn_dim  = cfg.ffn_dim;
        c.rms_eps  = cfg.rms_eps;
        c.W_qkv_mlp_proj = &I.sgl_qkv.back();
        c.W_out          = &I.sgl_out.back();
        c.norm_q_bf16    = st.norm_q->data;
        c.norm_k_bf16    = st.norm_k->data;

        auto blk = std::make_unique<SingleStreamBlock>(c);
        if (!blk->ok()) {
            fail("sgl block " + std::to_string(i) + ": " + blk->last_error());
            return;
        }
        I.sgl_blocks.push_back(std::move(blk));
    }

    // ---- RoPE tables (4-axis FLUX2 convention, host-built then uploaded). ----
    {
        const std::vector<int> axes_dim = {
            cfg.head_dim / 4, cfg.head_dim / 4,
            cfg.head_dim / 4, cfg.head_dim / 4
        };
        int H_p = cfg.H_patches > 0 ? cfg.H_patches : (int)std::round(std::sqrt((double)cfg.seq_img));
        int W_p = cfg.W_patches > 0 ? cfg.W_patches : H_p;
        if (H_p * W_p != cfg.seq_img) {
            // Non-square seq_img: fall back to a 1×N layout. Used by synthetic
            // numerics tests; real generation should always specify H_patches.
            H_p = 1;
            W_p = cfg.seq_img;
        }

        auto upload = [&](float** dst_cos, float** dst_sin,
                           const std::vector<float>& cos_h,
                           const std::vector<float>& sin_h) {
            const size_t bytes = cos_h.size() * sizeof(float);
            cudaMalloc(reinterpret_cast<void**>(dst_cos), bytes);
            cudaMalloc(reinterpret_cast<void**>(dst_sin), bytes);
            cudaMemcpy(*dst_cos, cos_h.data(), bytes, cudaMemcpyHostToDevice);
            cudaMemcpy(*dst_sin, sin_h.data(), bytes, cudaMemcpyHostToDevice);
        };

        // ---- Image-stream RoPE: pos (0, h_idx, w_idx, 0) ----
        {
            std::vector<std::vector<int>> pos(cfg.seq_img, std::vector<int>(4, 0));
            for (int s = 0; s < cfg.seq_img; ++s) {
                pos[s][1] = s / W_p;
                pos[s][2] = s % W_p;
            }
            std::vector<float> cos_h, sin_h;
            build_rope_4axis_tables(axes_dim, pos, cfg.rope_theta, cos_h, sin_h);
            upload(&I.d_rope_cos_img, &I.d_rope_sin_img, cos_h, sin_h);
        }
        // ---- Text-stream RoPE: pos (0, 0, 0, l_idx) ----
        {
            std::vector<std::vector<int>> pos(cfg.seq_txt, std::vector<int>(4, 0));
            for (int s = 0; s < cfg.seq_txt; ++s) pos[s][3] = s;
            std::vector<float> cos_h, sin_h;
            build_rope_4axis_tables(axes_dim, pos, cfg.rope_theta, cos_h, sin_h);
            upload(&I.d_rope_cos_txt, &I.d_rope_sin_txt, cos_h, sin_h);
        }
        // ---- Combined RoPE for single-stream: [txt || img] ----
        {
            const int S = cfg.seq_txt + cfg.seq_img;
            std::vector<std::vector<int>> pos(S, std::vector<int>(4, 0));
            for (int s = 0; s < cfg.seq_txt; ++s) pos[s][3] = s;
            for (int i = 0; i < cfg.seq_img; ++i) {
                pos[cfg.seq_txt + i][1] = i / W_p;
                pos[cfg.seq_txt + i][2] = i % W_p;
            }
            std::vector<float> cos_h, sin_h;
            build_rope_4axis_tables(axes_dim, pos, cfg.rope_theta, cos_h, sin_h);
            upload(&I.d_rope_cos_sgl, &I.d_rope_sin_sgl, cos_h, sin_h);
        }
    }

    // ---- Workspace plan. ----
    const size_t b_img      = static_cast<size_t>(rows_img)      * hidden * sizeof(__nv_bfloat16);
    const size_t b_txt      = static_cast<size_t>(rows_txt)      * hidden * sizeof(__nv_bfloat16);
    const size_t b_combined = static_cast<size_t>(rows_combined) * hidden * sizeof(__nv_bfloat16);

    size_t c = 0;
    I.off_img_hidden = c;                       c += b_img;
    c = align_up(c, ALIGN); I.off_txt_hidden  = c; c += b_txt;
    c = align_up(c, ALIGN); I.off_combined    = c; c += b_combined;
    c = align_up(c, ALIGN); I.off_mod_ws      = c; c += I.mod_mlp->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_img_emb_ws  = c; c += I.img_emb->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_ctx_emb_ws  = c; c += I.ctx_emb->workspace_size_bytes();
    // Per-block workspaces are reused across blocks of the same type — size to the max.
    size_t dbl_max = 0;
    for (auto& b : I.dbl_blocks) dbl_max = std::max(dbl_max, b->workspace_size_bytes());
    size_t sgl_max = 0;
    for (auto& b : I.sgl_blocks) sgl_max = std::max(sgl_max, b->workspace_size_bytes());
    c = align_up(c, ALIGN); I.off_dbl_blk_ws = c; c += dbl_max;
    c = align_up(c, ALIGN); I.off_sgl_blk_ws = c; c += sgl_max;
    c = align_up(c, ALIGN); I.off_final_ws   = c; c += I.final_proj->workspace_size_bytes();
    I.total_ws = c;

    I.valid = true;
}

FluxTransformer::~FluxTransformer() {
    if (impl_) {
        if (impl_->d_rope_cos_img) cudaFree(impl_->d_rope_cos_img);
        if (impl_->d_rope_sin_img) cudaFree(impl_->d_rope_sin_img);
        if (impl_->d_rope_cos_txt) cudaFree(impl_->d_rope_cos_txt);
        if (impl_->d_rope_sin_txt) cudaFree(impl_->d_rope_sin_txt);
        if (impl_->d_rope_cos_sgl) cudaFree(impl_->d_rope_cos_sgl);
        if (impl_->d_rope_sin_sgl) cudaFree(impl_->d_rope_sin_sgl);
    }
}

bool        FluxTransformer::ok()         const { return impl_ && impl_->valid; }
const char* FluxTransformer::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t FluxTransformer::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

// ============================================================================
// Forward
// ============================================================================

bool FluxTransformer::forward(const void* image_latent, const void* text_emb,
                               const void* timestep_emb,
                               void* image_latent_out,
                               void* workspace, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* img_hidden = ws + I.off_img_hidden;
    void* txt_hidden = ws + I.off_txt_hidden;
    void* combined   = ws + I.off_combined;
    void* mod_ws     = ws + I.off_mod_ws;
    void* img_emb_ws = ws + I.off_img_emb_ws;
    void* ctx_emb_ws = ws + I.off_ctx_emb_ws;
    void* dbl_ws     = ws + I.off_dbl_blk_ws;
    void* sgl_ws     = ws + I.off_sgl_blk_ws;
    void* final_ws   = ws + I.off_final_ws;

    const int hidden = I.cfg.n_heads * I.cfg.head_dim;
    const int rows_img = I.cfg.batch * I.cfg.seq_img;

    // 1. Embeddings.
    if (!I.img_emb->forward(image_latent, img_hidden,
                              img_emb_ws, I.img_emb->workspace_size_bytes(), stream))
        { I.err = std::string("img_emb: ") + I.img_emb->last_error(); return false; }
    if (!I.ctx_emb->forward(text_emb, txt_hidden,
                              ctx_emb_ws, I.ctx_emb->workspace_size_bytes(), stream))
        { I.err = std::string("ctx_emb: ") + I.ctx_emb->last_error(); return false; }

    // 2. Modulation MLP (output pointers live in mod_ws).
    ModulationMLP::Output mod{};
    if (!I.mod_mlp->forward(timestep_emb, mod, mod_ws,
                              I.mod_mlp->workspace_size_bytes(), stream))
        { I.err = std::string("mod_mlp: ") + I.mod_mlp->last_error(); return false; }

    // 3. 8 × DoubleStreamBlock.
    DoubleStreamBlock::Modulation dmod;
    dmod.img_scale_attn = mod.img_scale_attn; dmod.img_shift_attn = mod.img_shift_attn;
    dmod.img_gate_attn  = mod.img_gate_attn;  dmod.img_scale_mlp  = mod.img_scale_mlp;
    dmod.img_shift_mlp  = mod.img_shift_mlp;  dmod.img_gate_mlp   = mod.img_gate_mlp;
    dmod.txt_scale_attn = mod.txt_scale_attn; dmod.txt_shift_attn = mod.txt_shift_attn;
    dmod.txt_gate_attn  = mod.txt_gate_attn;  dmod.txt_scale_mlp  = mod.txt_scale_mlp;
    dmod.txt_shift_mlp  = mod.txt_shift_mlp;  dmod.txt_gate_mlp   = mod.txt_gate_mlp;
    for (int i = 0; i < I.cfg.num_double_blocks; ++i) {
        if (!I.dbl_blocks[i]->forward(img_hidden, txt_hidden, dmod,
                                        I.d_rope_cos_img, I.d_rope_sin_img,
                                        I.d_rope_cos_txt, I.d_rope_sin_txt,
                                        dbl_ws, I.dbl_blocks[i]->workspace_size_bytes(),
                                        stream)) {
            I.err = "dbl block " + std::to_string(i) + ": " + I.dbl_blocks[i]->last_error();
            return false;
        }
    }

    // 4. Concat [txt_hidden || img_hidden] → combined.
    if (!concat_two_streams_bf16(txt_hidden, img_hidden, combined,
                                  I.cfg.batch, I.cfg.seq_txt, I.cfg.seq_img, hidden, stream))
        { I.err = "concat_two_streams"; return false; }

    // 5. 24 × SingleStreamBlock on combined sequence.
    SingleStreamBlock::Modulation smod;
    smod.scale = mod.single_scale; smod.shift = mod.single_shift; smod.gate = mod.single_gate;
    for (int i = 0; i < I.cfg.num_single_blocks; ++i) {
        if (!I.sgl_blocks[i]->forward(combined, smod,
                                        I.d_rope_cos_sgl, I.d_rope_sin_sgl,
                                        sgl_ws, I.sgl_blocks[i]->workspace_size_bytes(),
                                        stream)) {
            I.err = "sgl block " + std::to_string(i) + ": " + I.sgl_blocks[i]->last_error();
            return false;
        }
    }

    // 6. Take the image tail (last seq_img rows per batch) → img_hidden.
    if (!take_tail_bf16(combined, img_hidden,
                         I.cfg.batch, I.cfg.seq_txt, I.cfg.seq_img, hidden, stream))
        { I.err = "take_tail"; return false; }
    (void)rows_img;

    // 7. Final projection.
    if (!I.final_proj->forward(img_hidden, mod.norm_out_scale, mod.norm_out_shift,
                                 image_latent_out,
                                 final_ws, I.final_proj->workspace_size_bytes(), stream))
        { I.err = std::string("final: ") + I.final_proj->last_error(); return false; }

    return true;
}

} // namespace f2k::cuda
