// ModulationMLP — 6-Linear timestep → per-block modulation projection.

#include "backend/cuda/modulation_mlp.h"

#include "backend/cuda/linear.h"
#include "backend/cuda/kernels/silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;
constexpr int    PAD_M = 128;   // Linear M constraint

} // anonymous namespace

namespace f2k::cuda {

struct ModulationMLP::Impl {
    int hidden = 0;
    int time_dim = 0;
    bool valid = false;
    std::string err;

    std::unique_ptr<Linear> lin_t1, lin_t2;
    std::unique_ptr<Linear> lin_dimg, lin_dtxt;
    std::unique_ptr<Linear> lin_single, lin_norm_out;

    // Workspace partitioning.
    size_t off_t_in    = 0;      // [PAD_M, time_dim] BF16
    size_t off_hidden  = 0;      // [PAD_M, hidden]   BF16  — t after linears
    size_t off_dimg    = 0;      // [PAD_M, 6*hidden]
    size_t off_dtxt    = 0;
    size_t off_single  = 0;      // [PAD_M, 3*hidden]
    size_t off_norm    = 0;      // [PAD_M, 2*hidden]
    size_t off_t1_ws   = 0;
    size_t off_t2_ws   = 0;
    size_t off_dimg_ws = 0;
    size_t off_dtxt_ws = 0;
    size_t off_single_ws  = 0;
    size_t off_norm_ws    = 0;
    size_t total_ws = 0;
};

ModulationMLP::ModulationMLP(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.hidden   = cfg.hidden_dim;
    I.time_dim = cfg.time_dim;

    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (I.hidden <= 0 || I.time_dim <= 0)
        { fail("ModulationMLP: zero/negative dim"); return; }
    if (I.hidden % 64 != 0)  { fail("hidden must be %% 64"); return; }
    if (I.time_dim % 64 != 0){ fail("time_dim must be %% 64"); return; }
    if (!cfg.W_time_1 || !cfg.W_time_2 || !cfg.W_double_img || !cfg.W_double_txt ||
        !cfg.W_single || !cfg.W_norm_out)
        { fail("missing PreQuantNVFP4 weight"); return; }

    auto make_lin = [&](int M, int K, int N, const PreQuantNVFP4* W) {
        Linear::Config c;
        c.batch_rows = M; c.in_features = K; c.out_features = N;
        set_linear_weight(c, W);
        return std::make_unique<Linear>(c);
    };

    I.lin_t1       = make_lin(PAD_M, I.time_dim, I.hidden,     cfg.W_time_1);
    I.lin_t2       = make_lin(PAD_M, I.hidden,   I.hidden,     cfg.W_time_2);
    I.lin_dimg     = make_lin(PAD_M, I.hidden,   6 * I.hidden, cfg.W_double_img);
    I.lin_dtxt     = make_lin(PAD_M, I.hidden,   6 * I.hidden, cfg.W_double_txt);
    I.lin_single   = make_lin(PAD_M, I.hidden,   3 * I.hidden, cfg.W_single);
    I.lin_norm_out = make_lin(PAD_M, I.hidden,   2 * I.hidden, cfg.W_norm_out);

    auto check = [&](const std::unique_ptr<Linear>& l, const char* name) {
        if (!l->ok()) { fail(std::string(name) + ": " + l->last_error()); return false; }
        return true;
    };
    if (!check(I.lin_t1,       "lin_t1"))       return;
    if (!check(I.lin_t2,       "lin_t2"))       return;
    if (!check(I.lin_dimg,     "lin_dimg"))     return;
    if (!check(I.lin_dtxt,     "lin_dtxt"))     return;
    if (!check(I.lin_single,   "lin_single"))   return;
    if (!check(I.lin_norm_out, "lin_norm_out")) return;

    const size_t b_time    = static_cast<size_t>(PAD_M) * I.time_dim * sizeof(__nv_bfloat16);
    const size_t b_hidden  = static_cast<size_t>(PAD_M) * I.hidden   * sizeof(__nv_bfloat16);
    const size_t b_6h      = static_cast<size_t>(PAD_M) * 6 * I.hidden * sizeof(__nv_bfloat16);
    const size_t b_3h      = static_cast<size_t>(PAD_M) * 3 * I.hidden * sizeof(__nv_bfloat16);
    const size_t b_2h      = static_cast<size_t>(PAD_M) * 2 * I.hidden * sizeof(__nv_bfloat16);

    size_t c = 0;
    I.off_t_in    = c;                          c += b_time;
    c = align_up(c, ALIGN); I.off_hidden  = c;  c += b_hidden;
    c = align_up(c, ALIGN); I.off_dimg    = c;  c += b_6h;
    c = align_up(c, ALIGN); I.off_dtxt    = c;  c += b_6h;
    c = align_up(c, ALIGN); I.off_single  = c;  c += b_3h;
    c = align_up(c, ALIGN); I.off_norm    = c;  c += b_2h;
    c = align_up(c, ALIGN); I.off_t1_ws   = c;  c += I.lin_t1->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_t2_ws   = c;  c += I.lin_t2->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_dimg_ws = c;  c += I.lin_dimg->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_dtxt_ws = c;  c += I.lin_dtxt->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_single_ws = c; c += I.lin_single->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_norm_ws   = c; c += I.lin_norm_out->workspace_size_bytes();
    I.total_ws = c;

    I.valid = true;
}

ModulationMLP::~ModulationMLP() = default;

bool        ModulationMLP::ok()         const { return impl_ && impl_->valid; }
const char* ModulationMLP::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t ModulationMLP::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

bool ModulationMLP::forward(const void* timestep_emb_bf16,
                             Output& out,
                             void* workspace, size_t workspace_size,
                             cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.total_ws) { I.err = "workspace too small"; return false; }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* t_in     = ws + I.off_t_in;
    void* hidden   = ws + I.off_hidden;
    void* dimg_buf = ws + I.off_dimg;
    void* dtxt_buf = ws + I.off_dtxt;
    void* single_buf = ws + I.off_single;
    void* norm_buf   = ws + I.off_norm;
    void* t1_ws  = ws + I.off_t1_ws;
    void* t2_ws  = ws + I.off_t2_ws;
    void* dimg_ws = ws + I.off_dimg_ws;
    void* dtxt_ws = ws + I.off_dtxt_ws;
    void* single_ws = ws + I.off_single_ws;
    void* norm_ws   = ws + I.off_norm_ws;

    // Pad t_in to [PAD_M, time_dim]: copy input into row 0, zero rows 1..127.
    cudaMemsetAsync(t_in, 0, static_cast<size_t>(PAD_M) * I.time_dim * sizeof(__nv_bfloat16), stream);
    cudaMemcpyAsync(t_in, timestep_emb_bf16,
                    static_cast<size_t>(I.time_dim) * sizeof(__nv_bfloat16),
                    cudaMemcpyDefault, stream);

    // Linear 1 + SiLU
    if (!I.lin_t1->forward(t_in, hidden, t1_ws, I.lin_t1->workspace_size_bytes(), stream))
        { I.err = std::string("lin_t1: ") + I.lin_t1->last_error(); return false; }
    if (!silu_inplace_bf16(hidden, static_cast<size_t>(PAD_M) * I.hidden, stream))
        { I.err = "silu after t1"; return false; }

    // Linear 2 + SiLU
    if (!I.lin_t2->forward(hidden, hidden, t2_ws, I.lin_t2->workspace_size_bytes(), stream))
        { I.err = std::string("lin_t2: ") + I.lin_t2->last_error(); return false; }
    if (!silu_inplace_bf16(hidden, static_cast<size_t>(PAD_M) * I.hidden, stream))
        { I.err = "silu after t2"; return false; }

    // 4 output projections
    if (!I.lin_dimg->forward(hidden, dimg_buf, dimg_ws, I.lin_dimg->workspace_size_bytes(), stream))
        { I.err = std::string("lin_dimg: ") + I.lin_dimg->last_error(); return false; }
    if (!I.lin_dtxt->forward(hidden, dtxt_buf, dtxt_ws, I.lin_dtxt->workspace_size_bytes(), stream))
        { I.err = std::string("lin_dtxt: ") + I.lin_dtxt->last_error(); return false; }
    if (!I.lin_single->forward(hidden, single_buf, single_ws, I.lin_single->workspace_size_bytes(), stream))
        { I.err = std::string("lin_single: ") + I.lin_single->last_error(); return false; }
    if (!I.lin_norm_out->forward(hidden, norm_buf, norm_ws, I.lin_norm_out->workspace_size_bytes(), stream))
        { I.err = std::string("lin_norm_out: ") + I.lin_norm_out->last_error(); return false; }

    // Slice each output's row 0 into named pointers. byte stride = hidden * 2.
    const size_t hb = static_cast<size_t>(I.hidden) * sizeof(__nv_bfloat16);
    uint8_t* di = static_cast<uint8_t*>(dimg_buf);
    uint8_t* dt = static_cast<uint8_t*>(dtxt_buf);
    uint8_t* sg = static_cast<uint8_t*>(single_buf);
    uint8_t* no = static_cast<uint8_t*>(norm_buf);

    // Flux2Modulation emits each (shift, scale, gate) set SHIFT-first — see
    // diffusers transformer_flux2.py Flux2Modulation.split + the block
    // destructuring `(shift_msa, scale_msa, gate_msa)`. So chunk 0 is shift,
    // chunk 1 is scale, chunk 2 is gate. (norm_out below goes through
    // AdaLayerNormContinuous instead, which is scale-first — see normalization.py.)
    out.img_shift_attn = di + 0*hb;
    out.img_scale_attn = di + 1*hb;
    out.img_gate_attn  = di + 2*hb;
    out.img_shift_mlp  = di + 3*hb;
    out.img_scale_mlp  = di + 4*hb;
    out.img_gate_mlp   = di + 5*hb;

    out.txt_shift_attn = dt + 0*hb;
    out.txt_scale_attn = dt + 1*hb;
    out.txt_gate_attn  = dt + 2*hb;
    out.txt_shift_mlp  = dt + 3*hb;
    out.txt_scale_mlp  = dt + 4*hb;
    out.txt_gate_mlp   = dt + 5*hb;

    out.single_shift   = sg + 0*hb;
    out.single_scale   = sg + 1*hb;
    out.single_gate    = sg + 2*hb;

    // AdaLayerNormContinuous: scale, shift = chunk(emb, 2) — scale-first.
    out.norm_out_scale = no + 0*hb;
    out.norm_out_shift = no + 1*hb;

    return true;
}

} // namespace f2k::cuda
