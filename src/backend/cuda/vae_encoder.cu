// VAEEncoder — assembles the FLUX.2-klein VAE encoder and runs it in order.
// Mirror of vae_decoder.cu; the only new primitive is the asymmetric (0,1,0,1)
// downsample pad, implemented by pad_bottom_right_bf16 below.

#include "backend/cuda/vae_encoder.h"

#include "backend/cuda/conv2d.h"
#include "backend/cuda/vae_resnet.h"
#include "backend/cuda/vae_attn.h"
#include "backend/cuda/kernels/groupnorm.h"
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

// Channel counts (block_out_channels) of the FLUX.2-klein VAE encoder.
constexpr int BLK[4] = {128, 256, 512, 512};

// dst[N,C,H+1,W+1] = src[N,C,H,W] with the extra bottom row and right column
// zero-filled. Matches F.pad(x, (0,1,0,1)) before a stride-2 conv (padding=0).
__global__ void pad_br_kernel(const __nv_bfloat16* __restrict__ src,
                              __nv_bfloat16* __restrict__ dst,
                              int N, int C, int H, int W) {
    const int Hd = H + 1, Wd = W + 1;
    const long total = (long)N * C * Hd * Wd;
    for (long idx = blockIdx.x * (long)blockDim.x + threadIdx.x;
         idx < total; idx += gridDim.x * (long)blockDim.x) {
        const int w = idx % Wd;
        long t = idx / Wd;
        const int h = t % Hd; t /= Hd;
        const int c = t % C;
        const int n = t / C;
        __nv_bfloat16 v = __float2bfloat16(0.0f);
        if (h < H && w < W) v = src[(((long)n * C + c) * H + h) * W + w];
        dst[idx] = v;
    }
}

bool pad_bottom_right_bf16(const void* src, void* dst, int N, int C, int H, int W,
                           cudaStream_t stream) {
    const long total = (long)N * C * (H + 1) * (W + 1);
    const int block = 256;
    int grid = (int)std::min<long>((total + block - 1) / block, 65535);
    if (grid < 1) grid = 1;
    pad_br_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(src), static_cast<__nv_bfloat16*>(dst), N, C, H, W);
    return cudaGetLastError() == cudaSuccess;
}

} // anonymous namespace

namespace f2k::cuda {

struct VAEEncoder::Impl {
    Config cfg{};
    bool valid = false;
    std::string err;

    int H_lat = 0, W_lat = 0;

    std::unique_ptr<Conv2d> conv_in;                 // 3 → 128
    std::unique_ptr<ResnetBlock> down_res[4][2];     // [block][resnet]
    std::unique_ptr<Conv2d>      down_samp[3];       // stride-2 conv (blocks 0,1,2)

    std::unique_ptr<ResnetBlock> mid_res0;
    std::unique_ptr<VAEAttention> mid_attn;
    std::unique_ptr<ResnetBlock> mid_res1;

    void* d_normout_gain = nullptr;
    void* d_normout_bias = nullptr;
    std::unique_ptr<Conv2d> conv_out;                // 512 → 64
    std::unique_ptr<Conv2d> quant_conv;              // 64 → 64 (1x1); nullable

    // Workspace plan.
    size_t off_a = 0, off_b = 0, off_pad = 0, off_sub = 0;
    size_t buf_max = 0, pad_max = 0, sub_max = 0, total_ws = 0;

    const f2k::TensorView* T(const std::string& n) const { return cfg.reader->find(n); }
    const void* W(const std::string& n) const { const auto* t = T(n); return t ? t->data : nullptr; }
};

VAEEncoder::VAEEncoder(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;
    if (!cfg.reader) { I.err = "VAEEncoder: null reader"; return; }
    if (cfg.N <= 0 || cfg.H_in <= 0 || cfg.W_in <= 0) { I.err = "bad dims"; return; }
    if (cfg.H_in % 8 != 0 || cfg.W_in % 8 != 0) { I.err = "H/W must be divisible by 8"; return; }
    I.H_lat = cfg.H_in / 8;
    I.W_lat = cfg.W_in / 8;
    const std::string p = cfg.prefix + ".";

    auto need = [&](const std::string& name) -> const void* {
        const auto* v = I.W(name);
        if (!v) I.err = "missing tensor: " + name;
        return v;
    };

    // --- conv_in (3 → 128, 3x3 pad 1) ---
    {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = 3; c.H_in = cfg.H_in; c.W_in = cfg.W_in;
        c.C_out = BLK[0]; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
        c.W_bf16 = need(p + "conv_in.weight"); if (!c.W_bf16) return;
        c.bias_bf16 = need(p + "conv_in.bias"); if (!c.bias_bf16) return;
        I.conv_in = std::make_unique<Conv2d>(c);
        if (!I.conv_in->ok()) { I.err = std::string("conv_in: ") + I.conv_in->last_error(); return; }
    }

    auto make_resnet = [&](const std::string& prefix, int C_in, int C_out, int H, int W,
                           std::unique_ptr<ResnetBlock>& dst, bool has_shortcut) {
        ResnetBlock::Config rc{};
        rc.N = cfg.N; rc.C_in = C_in; rc.H = H; rc.W = W; rc.C_out = C_out; rc.num_groups = 32;
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

    // --- down_blocks[0..3] ---
    for (int i = 0; i < 4; ++i) {
        const int C_in  = (i == 0) ? BLK[0] : BLK[i - 1];
        const int C_out = BLK[i];
        const int H_pre = cfg.H_in >> i;   // spatial at the input of this block
        const int W_pre = cfg.W_in >> i;
        for (int r = 0; r < 2; ++r) {
            const int rc_in  = (r == 0) ? C_in : C_out;
            const bool sc = (r == 0) && (rc_in != C_out);
            char buf[128];
            std::snprintf(buf, sizeof(buf), "%sdown_blocks.%d.resnets.%d.", p.c_str(), i, r);
            make_resnet(buf, rc_in, C_out, H_pre, W_pre, I.down_res[i][r], sc);
            if (!I.err.empty()) return;
        }
        // downsampler (3x3 stride 2, padding 0; input is asymmetrically pre-padded)
        if (i < 3) {
            char bw[128], bb[128];
            std::snprintf(bw, sizeof(bw), "%sdown_blocks.%d.downsamplers.0.conv.weight", p.c_str(), i);
            std::snprintf(bb, sizeof(bb), "%sdown_blocks.%d.downsamplers.0.conv.bias",   p.c_str(), i);
            Conv2d::Config c{};
            c.N = cfg.N; c.C_in = C_out; c.H_in = H_pre + 1; c.W_in = W_pre + 1;
            c.C_out = C_out; c.kH = 3; c.kW = 3; c.stride = 2; c.padding = 0;
            c.W_bf16 = need(bw); if (!c.W_bf16) return;
            c.bias_bf16 = need(bb); if (!c.bias_bf16) return;
            I.down_samp[i] = std::make_unique<Conv2d>(c);
            if (!I.down_samp[i]->ok()) { I.err = std::string(bw) + ": " + I.down_samp[i]->last_error(); return; }
        }
    }

    // --- mid block (all at latent spatial, 512 ch) ---
    make_resnet(p + "mid_block.resnets.0.", BLK[3], BLK[3], I.H_lat, I.W_lat, I.mid_res0, false);
    if (!I.err.empty()) return;
    {
        VAEAttention::Config ac{};
        ac.N = cfg.N; ac.C = BLK[3]; ac.H = I.H_lat; ac.W = I.W_lat; ac.num_groups = 32;
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
    make_resnet(p + "mid_block.resnets.1.", BLK[3], BLK[3], I.H_lat, I.W_lat, I.mid_res1, false);
    if (!I.err.empty()) return;

    // --- conv_norm_out (GroupNorm 32 over 512) — device gain/bias ---
    {
        const auto* g = I.T(p + "conv_norm_out.weight");
        const auto* b = I.T(p + "conv_norm_out.bias");
        if (!g || !b) { I.err = "missing conv_norm_out"; return; }
        const size_t n = (size_t)BLK[3] * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_normout_gain, n) != cudaSuccess) { I.err = "malloc normout"; return; }
        if (cudaMalloc(&I.d_normout_bias, n) != cudaSuccess) { I.err = "malloc normout"; return; }
        cudaMemcpy(I.d_normout_gain, g->data, n, cudaMemcpyHostToDevice);
        cudaMemcpy(I.d_normout_bias, b->data, n, cudaMemcpyHostToDevice);
    }

    // --- conv_out (512 → 64, 3x3 pad 1) ---
    {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = BLK[3]; c.H_in = I.H_lat; c.W_in = I.W_lat;
        c.C_out = 64; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
        c.W_bf16 = need(p + "conv_out.weight"); if (!c.W_bf16) return;
        c.bias_bf16 = need(p + "conv_out.bias"); if (!c.bias_bf16) return;
        I.conv_out = std::make_unique<Conv2d>(c);
        if (!I.conv_out->ok()) { I.err = std::string("conv_out: ") + I.conv_out->last_error(); return; }
    }

    // --- quant_conv (64 → 64, 1x1) — sibling of `prefix`. Optional. ---
    if (!cfg.quant_conv_name.empty()) {
        const std::string q = cfg.quant_conv_name + ".";
        if (I.W(q + "weight")) {
            Conv2d::Config c{};
            c.N = cfg.N; c.C_in = 64; c.H_in = I.H_lat; c.W_in = I.W_lat;
            c.C_out = 64; c.kH = 1; c.kW = 1; c.stride = 1; c.padding = 0;
            c.W_bf16 = need(q + "weight"); if (!c.W_bf16) return;
            c.bias_bf16 = need(q + "bias"); if (!c.bias_bf16) return;
            I.quant_conv = std::make_unique<Conv2d>(c);
            if (!I.quant_conv->ok()) { I.err = std::string("quant_conv: ") + I.quant_conv->last_error(); return; }
        }
    }

    // --- Workspace planning ---
    auto stage = [&](int C, int H, int W) { return (size_t)cfg.N * C * H * W * sizeof(__nv_bfloat16); };
    size_t mb = 0;
    mb = std::max(mb, stage(BLK[0], cfg.H_in,      cfg.W_in));        // conv_in / down0 resnets
    mb = std::max(mb, stage(BLK[1], cfg.H_in >> 1, cfg.W_in >> 1));   // down1 resnets
    mb = std::max(mb, stage(BLK[2], cfg.H_in >> 2, cfg.W_in >> 2));   // down2 resnets
    mb = std::max(mb, stage(BLK[3], cfg.H_in >> 3, cfg.W_in >> 3));   // down3 / mid
    I.buf_max = mb;

    size_t pm = 0;
    for (int i = 0; i < 3; ++i)
        pm = std::max(pm, stage(BLK[i], (cfg.H_in >> i) + 1, (cfg.W_in >> i) + 1));
    I.pad_max = pm;

    size_t sm = I.conv_in->workspace_size_bytes();
    for (int i = 0; i < 4; ++i) {
        for (int r = 0; r < 2; ++r) sm = std::max(sm, I.down_res[i][r]->workspace_size_bytes());
        if (i < 3) sm = std::max(sm, I.down_samp[i]->workspace_size_bytes());
    }
    sm = std::max(sm, I.mid_res0->workspace_size_bytes());
    sm = std::max(sm, I.mid_attn->workspace_size_bytes());
    sm = std::max(sm, I.mid_res1->workspace_size_bytes());
    sm = std::max(sm, I.conv_out->workspace_size_bytes());
    if (I.quant_conv) sm = std::max(sm, I.quant_conv->workspace_size_bytes());
    I.sub_max = sm;

    size_t c = 0;
    c = align_up(c, ALIGN); I.off_a   = c; c += I.buf_max;
    c = align_up(c, ALIGN); I.off_b   = c; c += I.buf_max;
    c = align_up(c, ALIGN); I.off_pad = c; c += I.pad_max;
    c = align_up(c, ALIGN); I.off_sub = c; c += I.sub_max;
    I.total_ws = c;

    I.valid = true;
}

VAEEncoder::~VAEEncoder() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_normout_gain) cudaFree(I.d_normout_gain);
    if (I.d_normout_bias) cudaFree(I.d_normout_bias);
}

bool        VAEEncoder::ok()         const { return impl_ && impl_->valid; }
const char* VAEEncoder::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t VAEEncoder::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }
int VAEEncoder::latent_H() const { return impl_ ? impl_->H_lat : 0; }
int VAEEncoder::latent_W() const { return impl_ ? impl_->W_lat : 0; }
int VAEEncoder::output_C() const { return 64; }

bool VAEEncoder::forward(const void* image, void* moments,
                         void* ws_v, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }
    const auto& cfg = I.cfg;

    uint8_t* ws = static_cast<uint8_t*>(ws_v);
    void* A   = ws + I.off_a;
    void* B   = ws + I.off_b;
    void* PAD = ws + I.off_pad;
    void* SUB = ws + I.off_sub;

    // 1. conv_in: image → A [N,128,H,W]
    if (!I.conv_in->forward(image, A, SUB, I.sub_max, stream))
        { I.err = std::string("conv_in: ") + I.conv_in->last_error(); return false; }

    void* cur = A;   // live data
    void* alt = B;
    auto swap = [&]{ void* t = cur; cur = alt; alt = t; };

    // 2. down_blocks
    for (int i = 0; i < 4; ++i) {
        const int C_out = BLK[i];
        const int H_pre = cfg.H_in >> i;
        const int W_pre = cfg.W_in >> i;
        for (int r = 0; r < 2; ++r) {
            if (!I.down_res[i][r]->forward(cur, alt, SUB, I.sub_max, stream))
                { I.err = "down_res[" + std::to_string(i) + "][" + std::to_string(r) + "]"; return false; }
            swap();
        }
        if (i < 3) {
            // asymmetric (0,1,0,1) pad → stride-2 conv
            if (!pad_bottom_right_bf16(cur, PAD, cfg.N, C_out, H_pre, W_pre, stream))
                { I.err = "downsample pad"; return false; }
            if (!I.down_samp[i]->forward(PAD, alt, SUB, I.sub_max, stream))
                { I.err = "downsample conv: " + std::string(I.down_samp[i]->last_error()); return false; }
            swap();
        }
    }
    // cur: [N, 512, H_lat, W_lat]

    // 3. mid block
    if (!I.mid_res0->forward(cur, alt, SUB, I.sub_max, stream))
        { I.err = std::string("mid_res0: ") + I.mid_res0->last_error(); return false; }
    swap();
    if (!I.mid_attn->forward(cur, alt, SUB, I.sub_max, stream))
        { I.err = std::string("mid_attn: ") + I.mid_attn->last_error(); return false; }
    swap();
    if (!I.mid_res1->forward(cur, alt, SUB, I.sub_max, stream))
        { I.err = std::string("mid_res1: ") + I.mid_res1->last_error(); return false; }
    swap();
    // cur: [N, 512, H_lat, W_lat]

    // 4. conv_norm_out + SiLU + conv_out → 64 ch
    if (!groupnorm_bf16(cur, alt, I.d_normout_gain, I.d_normout_bias,
                        cfg.N, BLK[3], I.H_lat, I.W_lat, 32, 1e-6f, stream))
        { I.err = "conv_norm_out"; return false; }
    if (!silu_inplace_bf16(alt, (size_t)cfg.N * BLK[3] * I.H_lat * I.W_lat, stream))
        { I.err = "silu_out"; return false; }
    // conv_out writes to `cur` (reused; large enough for 64ch) unless quant_conv
    // follows, in which case conv_out → cur then quant_conv → moments.
    void* conv_out_dst = I.quant_conv ? cur : moments;
    if (!I.conv_out->forward(alt, conv_out_dst, SUB, I.sub_max, stream))
        { I.err = std::string("conv_out: ") + I.conv_out->last_error(); return false; }

    // 5. quant_conv (1x1, 64→64) → moments
    if (I.quant_conv) {
        if (!I.quant_conv->forward(conv_out_dst, moments, SUB, I.sub_max, stream))
            { I.err = std::string("quant_conv: ") + I.quant_conv->last_error(); return false; }
    }
    return true;
}

} // namespace f2k::cuda
