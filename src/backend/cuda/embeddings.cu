// Embedding / un-embedding modules. Thin wrappers over Linear; FinalProjection
// adds the RMSNorm(no-affine) + modulate(norm_out_scale/shift) stage.

#include "backend/cuda/embeddings.h"

#include "backend/cuda/linear.h"
#include "backend/cuda/kernels/rmsnorm.h"
#include "backend/cuda/kernels/modulation.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

} // anonymous namespace

namespace f2k::cuda {

// ---------------------------------------------------------------------------
// ImageEmbedder
// ---------------------------------------------------------------------------

struct ImageEmbedder::Impl {
    bool valid = false;
    std::string err;
    std::unique_ptr<Linear> lin;
};

ImageEmbedder::ImageEmbedder(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    if (cfg.batch_rows <= 0 || cfg.in_channels <= 0 || cfg.hidden_dim <= 0 || !cfg.W) {
        I.err = "ImageEmbedder: bad config"; return;
    }
    Linear::Config c;
    c.batch_rows = cfg.batch_rows;
    c.in_features  = cfg.in_channels;
    c.out_features = cfg.hidden_dim;
    set_linear_weight(c, cfg.W);
    I.lin = std::make_unique<Linear>(c);
    if (!I.lin->ok()) { I.err = std::string("lin: ") + I.lin->last_error(); return; }
    I.valid = true;
}
ImageEmbedder::~ImageEmbedder() = default;
bool        ImageEmbedder::ok()         const { return impl_ && impl_->valid; }
const char* ImageEmbedder::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t ImageEmbedder::workspace_size_bytes() const {
    return impl_ && impl_->lin ? impl_->lin->workspace_size_bytes() : 0;
}
bool ImageEmbedder::forward(const void* x_latent, void* x_hidden,
                             void* workspace, size_t ws_size, cudaStream_t stream) {
    if (!impl_->valid) return false;
    return impl_->lin->forward(x_latent, x_hidden, workspace, ws_size, stream);
}

// ---------------------------------------------------------------------------
// ContextEmbedder
// ---------------------------------------------------------------------------

struct ContextEmbedder::Impl {
    bool valid = false;
    std::string err;
    std::unique_ptr<Linear> lin;
};

ContextEmbedder::ContextEmbedder(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    if (cfg.batch_rows <= 0 || cfg.t5_dim <= 0 || cfg.hidden_dim <= 0 || !cfg.W) {
        I.err = "ContextEmbedder: bad config"; return;
    }
    Linear::Config c;
    c.batch_rows   = cfg.batch_rows;
    c.in_features  = cfg.t5_dim;
    c.out_features = cfg.hidden_dim;
    set_linear_weight(c, cfg.W);
    I.lin = std::make_unique<Linear>(c);
    if (!I.lin->ok()) { I.err = std::string("lin: ") + I.lin->last_error(); return; }
    I.valid = true;
}
ContextEmbedder::~ContextEmbedder() = default;
bool        ContextEmbedder::ok()         const { return impl_ && impl_->valid; }
const char* ContextEmbedder::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t ContextEmbedder::workspace_size_bytes() const {
    return impl_ && impl_->lin ? impl_->lin->workspace_size_bytes() : 0;
}
bool ContextEmbedder::forward(const void* txt_emb, void* txt_hidden,
                               void* workspace, size_t ws_size, cudaStream_t stream) {
    if (!impl_->valid) return false;
    return impl_->lin->forward(txt_emb, txt_hidden, workspace, ws_size, stream);
}

// ---------------------------------------------------------------------------
// FinalProjection (norm_out + proj_out)
// ---------------------------------------------------------------------------

struct FinalProjection::Impl {
    bool valid = false;
    std::string err;
    int rows = 0, hidden = 0, out_ch = 0;
    float rms_eps = 1e-6f;
    std::unique_ptr<Linear> lin_proj;
    void* d_ones = nullptr;
    size_t off_norm = 0, off_mod = 0, off_lin_ws = 0, total_ws = 0;
};

FinalProjection::FinalProjection(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    if (cfg.batch_rows <= 0 || cfg.hidden_dim <= 0 || cfg.out_channels <= 0 || !cfg.W_proj_out) {
        I.err = "FinalProjection: bad config"; return;
    }
    I.rows = cfg.batch_rows;
    I.hidden = cfg.hidden_dim;
    I.out_ch = cfg.out_channels;
    I.rms_eps = cfg.rms_eps;

    Linear::Config lc;
    lc.batch_rows   = I.rows;
    lc.in_features  = I.hidden;
    lc.out_features = I.out_ch;
    set_linear_weight(lc, cfg.W_proj_out);
    I.lin_proj = std::make_unique<Linear>(lc);
    if (!I.lin_proj->ok()) { I.err = std::string("lin_proj: ") + I.lin_proj->last_error(); return; }

    {
        std::vector<__nv_bfloat16> ones(I.hidden, __float2bfloat16(1.0f));
        if (cudaMalloc(&I.d_ones, ones.size() * 2) != cudaSuccess) {
            I.err = "malloc ones"; return;
        }
        cudaMemcpy(I.d_ones, ones.data(), ones.size() * 2, cudaMemcpyHostToDevice);
    }

    const size_t bh = static_cast<size_t>(I.rows) * I.hidden * sizeof(__nv_bfloat16);
    size_t c = 0;
    I.off_norm = c;                                c += bh;
    c = align_up(c, ALIGN); I.off_mod    = c;      c += bh;
    c = align_up(c, ALIGN); I.off_lin_ws = c;      c += I.lin_proj->workspace_size_bytes();
    I.total_ws = c;

    I.valid = true;
}

FinalProjection::~FinalProjection() {
    if (impl_ && impl_->d_ones) cudaFree(impl_->d_ones);
}

bool        FinalProjection::ok()         const { return impl_ && impl_->valid; }
const char* FinalProjection::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t FinalProjection::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

bool FinalProjection::forward(const void* x_hidden,
                               const void* norm_out_scale,
                               const void* norm_out_shift,
                               void* x_latent_out,
                               void* workspace, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* norm_buf = ws + I.off_norm;
    void* mod_buf  = ws + I.off_mod;
    void* lin_ws   = ws + I.off_lin_ws;

    if (!rmsnorm_bf16(x_hidden, I.d_ones, norm_buf, I.rows, I.hidden, I.rms_eps, stream))
        { I.err = "rmsnorm"; return false; }
    if (!modulate_bf16(norm_buf, mod_buf, norm_out_scale, norm_out_shift, I.rows, I.hidden, stream))
        { I.err = "modulate"; return false; }
    if (!I.lin_proj->forward(mod_buf, x_latent_out, lin_ws,
                              I.lin_proj->workspace_size_bytes(), stream))
        { I.err = std::string("lin_proj: ") + I.lin_proj->last_error(); return false; }
    return true;
}

} // namespace f2k::cuda
