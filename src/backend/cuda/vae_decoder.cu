// VAEDecoder — assembles all the pieces and feeds them in order.

#include "backend/cuda/vae_decoder.h"

#include "backend/cuda/conv2d.h"
#include "backend/cuda/vae_resnet.h"
#include "backend/cuda/vae_attn.h"
#include "backend/cuda/kernels/groupnorm.h"
#include "backend/cuda/kernels/upsample.h"
#include "backend/cuda/kernels/silu_mul.h"

#include "common/f2k_format.h"
#include "common/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstring>
#include <memory>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

// Channel counts at each stage (FLUX.2-klein VAE):
constexpr int CH[5] = {512, 512, 512, 256, 128};
//                     mid   up0   up1   up2  up3
// Spatial multiplier at each stage relative to the latent:
constexpr int SCALE[5] = {1, 2, 4, 8, 8};

} // anonymous namespace

namespace f2k::cuda {

struct VAEDecoder::Impl {
    Config cfg{};
    bool valid = false;
    std::string err;

    int H_out = 0, W_out = 0;

    // post_quant_conv (1×1, 32→32) — applied before conv_in; null if absent.
    std::unique_ptr<Conv2d> post_quant_conv;

    // conv_in
    std::unique_ptr<Conv2d> conv_in;

    // mid block
    std::unique_ptr<ResnetBlock> mid_res0;
    std::unique_ptr<VAEAttention> mid_attn;
    std::unique_ptr<ResnetBlock> mid_res1;

    // up_blocks[0..3]: each has 3 resnets + optional upsampler conv
    std::unique_ptr<ResnetBlock> up_res[4][3];
    std::unique_ptr<Conv2d>      up_conv[4];        // post-upsample conv (3 of them; null for block 3)

    // final norm + conv_out
    void* d_normout_gain = nullptr;
    void* d_normout_bias = nullptr;
    std::unique_ptr<Conv2d> conv_out;

    // Workspace plan.
    size_t off_buf_a = 0;
    size_t off_buf_b = 0;       // alternate buffer for between-stage swaps
    size_t off_upsample = 0;    // pre-upsample-conv NCHW buffer (max needed)
    size_t off_sub_ws  = 0;     // shared sub-component workspace
    size_t buf_bytes_max = 0;
    size_t sub_ws_max    = 0;
    size_t total_ws      = 0;

    // Helpers.
    const f2k::TensorView* T(const std::string& name) const {
        const auto* t = cfg.reader->find(name);
        return t;
    }
    const void* W(const std::string& name) const {
        const auto* t = T(name);
        return t ? t->data : nullptr;
    }
    static int find_groups(int C) {
        // VAE uses 32 unless C < 32 (not the case here).
        return std::min(32, C);
    }
};

VAEDecoder::VAEDecoder(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;
    if (!cfg.reader)      { I.err = "VAEDecoder: null reader"; return; }
    if (cfg.N <= 0 || cfg.H_lat <= 0 || cfg.W_lat <= 0)
        { I.err = "bad dims"; return; }

    I.H_out = cfg.H_lat * 8;
    I.W_out = cfg.W_lat * 8;
    const std::string p = cfg.prefix + ".";

    auto need = [&](const std::string& name) -> const void* {
        const auto* v = I.W(name);
        if (!v) I.err = "missing tensor: " + name;
        return v;
    };

    // --- post_quant_conv (1×1, 32 → 32, no padding) — sibling of `prefix`,
    //     not under it. Optional: skipped if the tensor isn't in the file. ---
    if (!cfg.post_quant_conv_name.empty()) {
        const std::string pq = cfg.post_quant_conv_name + ".";
        if (I.W(pq + "weight")) {
            Conv2d::Config c{};
            c.N = cfg.N; c.C_in = 32; c.H_in = cfg.H_lat; c.W_in = cfg.W_lat;
            c.C_out = 32; c.kH = 1; c.kW = 1; c.stride = 1; c.padding = 0;
            c.W_bf16 = need(pq + "weight"); if (!c.W_bf16) return;
            c.bias_bf16 = need(pq + "bias"); if (!c.bias_bf16) return;
            I.post_quant_conv = std::make_unique<Conv2d>(c);
            if (!I.post_quant_conv->ok())
                { I.err = std::string("post_quant_conv: ") + I.post_quant_conv->last_error(); return; }
        }
    }

    // --- conv_in (32 → 512, 3x3 padding 1) ---
    {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = 32; c.H_in = cfg.H_lat; c.W_in = cfg.W_lat;
        c.C_out = CH[0]; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
        c.W_bf16 = need(p + "conv_in.weight"); if (!c.W_bf16) return;
        c.bias_bf16 = need(p + "conv_in.bias"); if (!c.bias_bf16) return;
        I.conv_in = std::make_unique<Conv2d>(c);
        if (!I.conv_in->ok()) { I.err = std::string("conv_in: ") + I.conv_in->last_error(); return; }
    }

    auto make_resnet = [&](const std::string& prefix, int C_in, int C_out, int H, int W,
                           std::unique_ptr<ResnetBlock>& dst, bool has_shortcut) {
        ResnetBlock::Config rc{};
        rc.N = cfg.N; rc.C_in = C_in; rc.H = H; rc.W = W; rc.C_out = C_out;
        rc.num_groups = 32;
        rc.norm1_gain = need(prefix + "norm1.weight"); if (!rc.norm1_gain) return;
        rc.norm1_bias = need(prefix + "norm1.bias");   if (!rc.norm1_bias) return;
        rc.conv1_W    = need(prefix + "conv1.weight"); if (!rc.conv1_W) return;
        rc.conv1_bias = need(prefix + "conv1.bias");   if (!rc.conv1_bias) return;
        rc.norm2_gain = need(prefix + "norm2.weight"); if (!rc.norm2_gain) return;
        rc.norm2_bias = need(prefix + "norm2.bias");   if (!rc.norm2_bias) return;
        rc.conv2_W    = need(prefix + "conv2.weight"); if (!rc.conv2_W) return;
        rc.conv2_bias = need(prefix + "conv2.bias");   if (!rc.conv2_bias) return;
        if (has_shortcut) {
            rc.shortcut_W    = need(prefix + "conv_shortcut.weight"); if (!rc.shortcut_W) return;
            rc.shortcut_bias = need(prefix + "conv_shortcut.bias");   if (!rc.shortcut_bias) return;
        }
        dst = std::make_unique<ResnetBlock>(rc);
        if (!dst->ok()) I.err = std::string("resnet ") + prefix + ": " + dst->last_error();
    };

    // --- mid block ---
    make_resnet(p + "mid_block.resnets.0.", CH[0], CH[0], cfg.H_lat, cfg.W_lat, I.mid_res0, false);
    if (!I.err.empty()) return;
    {
        VAEAttention::Config ac{};
        ac.N = cfg.N; ac.C = CH[0]; ac.H = cfg.H_lat; ac.W = cfg.W_lat;
        ac.num_groups = 32;
        ac.norm_gain = need(p + "mid_block.attentions.0.group_norm.weight"); if (!ac.norm_gain) return;
        ac.norm_bias = need(p + "mid_block.attentions.0.group_norm.bias");   if (!ac.norm_bias) return;
        ac.to_q_W = need(p + "mid_block.attentions.0.to_q.weight"); if (!ac.to_q_W) return;
        ac.to_q_b = need(p + "mid_block.attentions.0.to_q.bias");   if (!ac.to_q_b) return;
        ac.to_k_W = need(p + "mid_block.attentions.0.to_k.weight"); if (!ac.to_k_W) return;
        ac.to_k_b = need(p + "mid_block.attentions.0.to_k.bias");   if (!ac.to_k_b) return;
        ac.to_v_W = need(p + "mid_block.attentions.0.to_v.weight"); if (!ac.to_v_W) return;
        ac.to_v_b = need(p + "mid_block.attentions.0.to_v.bias");   if (!ac.to_v_b) return;
        ac.to_out_W = need(p + "mid_block.attentions.0.to_out.0.weight"); if (!ac.to_out_W) return;
        ac.to_out_b = need(p + "mid_block.attentions.0.to_out.0.bias");   if (!ac.to_out_b) return;
        I.mid_attn = std::make_unique<VAEAttention>(ac);
        if (!I.mid_attn->ok()) { I.err = std::string("mid_attn: ") + I.mid_attn->last_error(); return; }
    }
    make_resnet(p + "mid_block.resnets.1.", CH[0], CH[0], cfg.H_lat, cfg.W_lat, I.mid_res1, false);
    if (!I.err.empty()) return;

    // --- up_blocks[0..3] ---
    // up_blocks[i] takes channels CH[i] and outputs CH[i+1] (block i's resnets do the transition).
    // Spatial at the *input* of each up block: SCALE[i] × latent. Output of upsample doubles it.
    for (int i = 0; i < 4; ++i) {
        const int C_pre = CH[i];
        const int C_post = CH[i + 1];
        const int H_pre = cfg.H_lat * SCALE[i];
        const int W_pre = cfg.W_lat * SCALE[i];
        // resnet 0: C_pre → C_post (shortcut if channel change)
        // resnets 1,2: C_post → C_post
        for (int r = 0; r < 3; ++r) {
            const int C_in  = (r == 0) ? C_pre  : C_post;
            const int C_out = C_post;
            const bool sc = (r == 0) && (C_in != C_out);
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%sup_blocks.%d.resnets.%d.", p.c_str(), i, r);
            make_resnet(buf, C_in, C_out, H_pre, W_pre, I.up_res[i][r], sc);
            if (!I.err.empty()) return;
        }
        // upsampler conv (3x3, C_post → C_post), only on i ∈ {0,1,2}.
        if (i < 3) {
            char buf_w[128], buf_b[128];
            std::snprintf(buf_w, sizeof(buf_w), "%sup_blocks.%d.upsamplers.0.conv.weight", p.c_str(), i);
            std::snprintf(buf_b, sizeof(buf_b), "%sup_blocks.%d.upsamplers.0.conv.bias",   p.c_str(), i);
            Conv2d::Config c{};
            c.N = cfg.N; c.C_in = C_post; c.H_in = H_pre * 2; c.W_in = W_pre * 2;
            c.C_out = C_post; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
            c.W_bf16 = need(buf_w); if (!c.W_bf16) return;
            c.bias_bf16 = need(buf_b); if (!c.bias_bf16) return;
            I.up_conv[i] = std::make_unique<Conv2d>(c);
            if (!I.up_conv[i]->ok()) { I.err = std::string(buf_w) + ": " + I.up_conv[i]->last_error(); return; }
        }
    }

    // --- final norm + conv_out (128 → 3, 3x3) ---
    {
        const auto* g = I.T(p + "conv_norm_out.weight");
        const auto* b = I.T(p + "conv_norm_out.bias");
        if (!g || !b) { I.err = "missing conv_norm_out"; return; }
        const size_t n_bytes = (size_t)CH[4] * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_normout_gain, n_bytes) != cudaSuccess) { I.err = "malloc normout"; return; }
        if (cudaMalloc(&I.d_normout_bias, n_bytes) != cudaSuccess) { I.err = "malloc normout"; return; }
        cudaMemcpy(I.d_normout_gain, g->data, n_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(I.d_normout_bias, b->data, n_bytes, cudaMemcpyHostToDevice);
    }
    {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = CH[4]; c.H_in = I.H_out; c.W_in = I.W_out;
        c.C_out = 3; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
        c.W_bf16 = need(p + "conv_out.weight"); if (!c.W_bf16) return;
        c.bias_bf16 = need(p + "conv_out.bias"); if (!c.bias_bf16) return;
        I.conv_out = std::make_unique<Conv2d>(c);
        if (!I.conv_out->ok()) { I.err = std::string("conv_out: ") + I.conv_out->last_error(); return; }
    }

    // --- Workspace planning ---
    // We use two scratch tensors A, B and ping-pong between them. Each holds the
    // largest intermediate. The largest intermediate is at the final stage:
    //   N × 128 × (8H_lat) × (8W_lat).
    // But pre-final-upsample we also have N × 256 × (4H_lat) × (4W_lat).
    // We size A/B at the max, plus a separate upsample-source buffer.
    auto stage_bytes = [&](int C, int H, int W) {
        return (size_t)cfg.N * C * H * W * sizeof(__nv_bfloat16);
    };
    size_t max_b = 0;
    max_b = std::max(max_b, stage_bytes(CH[0], cfg.H_lat,      cfg.W_lat));      // mid
    max_b = std::max(max_b, stage_bytes(CH[1], cfg.H_lat * 2,  cfg.W_lat * 2));  // up0 post
    max_b = std::max(max_b, stage_bytes(CH[2], cfg.H_lat * 4,  cfg.W_lat * 4));  // up1 post
    max_b = std::max(max_b, stage_bytes(CH[3], cfg.H_lat * 8,  cfg.W_lat * 8));  // up2 post
    max_b = std::max(max_b, stage_bytes(CH[4], cfg.H_lat * 8,  cfg.W_lat * 8));  // up3
    I.buf_bytes_max = max_b;

    // Sub-component max workspace.
    size_t sub_max = 0;
    if (I.post_quant_conv)
        sub_max = std::max(sub_max, I.post_quant_conv->workspace_size_bytes());
    sub_max = std::max(sub_max, I.conv_in->workspace_size_bytes());
    sub_max = std::max(sub_max, I.mid_res0->workspace_size_bytes());
    sub_max = std::max(sub_max, I.mid_attn->workspace_size_bytes());
    sub_max = std::max(sub_max, I.mid_res1->workspace_size_bytes());
    for (int i = 0; i < 4; ++i) {
        for (int r = 0; r < 3; ++r)
            sub_max = std::max(sub_max, I.up_res[i][r]->workspace_size_bytes());
        if (i < 3)
            sub_max = std::max(sub_max, I.up_conv[i]->workspace_size_bytes());
    }
    sub_max = std::max(sub_max, I.conv_out->workspace_size_bytes());
    I.sub_ws_max = sub_max;

    size_t c = 0;
    c = align_up(c, ALIGN); I.off_buf_a    = c; c += max_b;
    c = align_up(c, ALIGN); I.off_buf_b    = c; c += max_b;
    c = align_up(c, ALIGN); I.off_upsample = c; c += max_b;
    c = align_up(c, ALIGN); I.off_sub_ws   = c; c += sub_max;
    I.total_ws = c;

    I.valid = true;
}

VAEDecoder::~VAEDecoder() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_normout_gain) cudaFree(I.d_normout_gain);
    if (I.d_normout_bias) cudaFree(I.d_normout_bias);
}

bool        VAEDecoder::ok()         const { return impl_ && impl_->valid; }
const char* VAEDecoder::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t VAEDecoder::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }
int VAEDecoder::output_H() const { return impl_ ? impl_->H_out : 0; }
int VAEDecoder::output_W() const { return impl_ ? impl_->W_out : 0; }

bool VAEDecoder::forward(const void* x, void* pixels,
                         void* ws_v, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }
    const auto& cfg = I.cfg;

    uint8_t* ws = static_cast<uint8_t*>(ws_v);
    void* A      = ws + I.off_buf_a;
    void* B      = ws + I.off_buf_b;
    void* UPS    = ws + I.off_upsample;
    void* SUB_WS = ws + I.off_sub_ws;

    // 0. post_quant_conv (1×1, 32→32): x → UPS, then conv_in reads UPS. UPS is
    //    free until the first upsample stage and is sized to the max stage, so
    //    it easily holds [N,32,H,W]. Skipped (x used directly) if not present.
    const void* conv_in_src = x;
    if (I.post_quant_conv) {
        if (!I.post_quant_conv->forward(x, UPS, SUB_WS, I.sub_ws_max, stream))
            { I.err = std::string("post_quant_conv: ") + I.post_quant_conv->last_error(); return false; }
        conv_in_src = UPS;
    }

    // 1. conv_in: [N,32,H,W] → A [N,512,H,W]
    if (!I.conv_in->forward(conv_in_src, A, SUB_WS, I.sub_ws_max, stream))
        { I.err = std::string("conv_in: ") + I.conv_in->last_error(); return false; }

    // 2. mid block: A → B → A → B
    if (!I.mid_res0->forward(A, B, SUB_WS, I.sub_ws_max, stream))
        { I.err = std::string("mid_res0: ") + I.mid_res0->last_error(); return false; }
    if (!I.mid_attn->forward(B, A, SUB_WS, I.sub_ws_max, stream))
        { I.err = std::string("mid_attn: ") + I.mid_attn->last_error(); return false; }
    if (!I.mid_res1->forward(A, B, SUB_WS, I.sub_ws_max, stream))
        { I.err = std::string("mid_res1: ") + I.mid_res1->last_error(); return false; }
    // After mid: data lives in B at [N, 512, H_lat, W_lat].

    // 3. up_blocks: for each block, run 3 resnets (ping-pong), then optional
    //    upsample + 3x3 conv.
    for (int i = 0; i < 4; ++i) {
        const int C_post = CH[i + 1];
        const int H_pre = cfg.H_lat * SCALE[i];
        const int W_pre = cfg.W_lat * SCALE[i];

        // Resnets: alternate B → A → B → A (so after 3 resnets, data is in A).
        if (!I.up_res[i][0]->forward(B, A, SUB_WS, I.sub_ws_max, stream))
            { I.err = "up_res[" + std::to_string(i) + "][0]"; return false; }
        if (!I.up_res[i][1]->forward(A, B, SUB_WS, I.sub_ws_max, stream))
            { I.err = "up_res[" + std::to_string(i) + "][1]"; return false; }
        if (!I.up_res[i][2]->forward(B, A, SUB_WS, I.sub_ws_max, stream))
            { I.err = "up_res[" + std::to_string(i) + "][2]"; return false; }

        // After resnets: data in A, shape [N, C_post, H_pre, W_pre].
        if (i < 3) {
            // Upsample A → UPS (shape [N, C_post, 2*H_pre, 2*W_pre])
            if (!upsample2x_nearest_bf16(A, UPS, cfg.N, C_post, H_pre, W_pre, stream))
                { I.err = "upsample"; return false; }
            // Conv UPS → B
            if (!I.up_conv[i]->forward(UPS, B, SUB_WS, I.sub_ws_max, stream))
                { I.err = "up_conv: " + std::string(I.up_conv[i]->last_error()); return false; }
            // Now B holds [N, C_post, 2*H_pre, 2*W_pre], ready for next block.
        } else {
            // Last block: no upsample. Move A → B for the final stage.
            cudaMemcpyAsync(B, A,
                            (size_t)cfg.N * C_post * H_pre * W_pre * sizeof(__nv_bfloat16),
                            cudaMemcpyDeviceToDevice, stream);
        }
    }

    // 4. conv_norm_out + SiLU + conv_out
    // B: [N, 128, 8H, 8W] → groupnorm → A → silu → A → conv → pixels
    if (!groupnorm_bf16(B, A, I.d_normout_gain, I.d_normout_bias,
                        cfg.N, CH[4], I.H_out, I.W_out, 32, 1e-6f, stream))
        { I.err = "conv_norm_out"; return false; }
    if (!silu_inplace_bf16(A,
                           (size_t)cfg.N * CH[4] * I.H_out * I.W_out, stream))
        { I.err = "silu_out"; return false; }
    if (!I.conv_out->forward(A, pixels, SUB_WS, I.sub_ws_max, stream))
        { I.err = std::string("conv_out: ") + I.conv_out->last_error(); return false; }

    return true;
}

} // namespace f2k::cuda
