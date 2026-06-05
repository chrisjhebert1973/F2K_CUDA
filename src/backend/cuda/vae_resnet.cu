// VAE ResnetBlock — composes 2 GroupNorm + 2 Conv2d + SiLU + skip add.

#include "backend/cuda/vae_resnet.h"

#include "backend/cuda/conv2d.h"
#include "backend/cuda/kernels/groupnorm.h"
#include "backend/cuda/kernels/silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

// Elementwise y = a + b (BF16).
__global__ void add_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                const __nv_bfloat16* __restrict__ a,
                                const __nv_bfloat16* __restrict__ b,
                                size_t n) {
    const size_t i = static_cast<size_t>(blockIdx.x) * 256 + threadIdx.x;
    if (i >= n) return;
    const float av = __bfloat162float(a[i]);
    const float bv = __bfloat162float(b[i]);
    y[i] = __float2bfloat16(av + bv);
}

} // anonymous namespace

namespace f2k::cuda {

struct ResnetBlock::Impl {
    Config cfg{};
    bool valid = false;
    std::string err;

    // Device-resident gains/biases (the norm parameters; convs own their own).
    void* d_norm1_gain = nullptr;
    void* d_norm1_bias = nullptr;
    void* d_norm2_gain = nullptr;
    void* d_norm2_bias = nullptr;

    std::unique_ptr<Conv2d> conv1, conv2, shortcut;
    bool have_shortcut = false;

    // Workspace plan.
    size_t off_h1 = 0;     // [N, C_in,  H, W] post-norm1+silu
    size_t off_h2 = 0;     // [N, C_out, H, W] post-conv1
    size_t off_h3 = 0;     // [N, C_out, H, W] post-norm2+silu
    size_t off_h4 = 0;     // [N, C_out, H, W] post-conv2
    size_t off_skip = 0;   // [N, C_out, H, W] shortcut output (only if shortcut)
    size_t off_conv1_ws = 0;
    size_t off_conv2_ws = 0;
    size_t off_shortcut_ws = 0;
    size_t total_ws = 0;
};

ResnetBlock::ResnetBlock(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;

    if (cfg.N <= 0 || cfg.C_in <= 0 || cfg.H <= 0 || cfg.W <= 0 || cfg.C_out <= 0)
        { I.err = "ResnetBlock: bad dims"; return; }
    if (cfg.C_in  % cfg.num_groups != 0) { I.err = "C_in  not %% num_groups"; return; }
    if (cfg.C_out % cfg.num_groups != 0) { I.err = "C_out not %% num_groups"; return; }

    I.have_shortcut = (cfg.C_in != cfg.C_out);
    if (I.have_shortcut) {
        if (!cfg.shortcut_W) { I.err = "channel change needs shortcut weight"; return; }
    }

    // Upload norm params (convs do their own upload).
    auto upload = [&](void** dst, const void* host, int n_elem) -> bool {
        const size_t b = static_cast<size_t>(n_elem) * sizeof(__nv_bfloat16);
        if (cudaMalloc(dst, b) != cudaSuccess) return false;
        cudaMemcpy(*dst, host, b, cudaMemcpyHostToDevice);
        return true;
    };
    if (!upload(&I.d_norm1_gain, cfg.norm1_gain, cfg.C_in))  { I.err = "malloc norm1 gain"; return; }
    if (!upload(&I.d_norm1_bias, cfg.norm1_bias, cfg.C_in))  { I.err = "malloc norm1 bias"; return; }
    if (!upload(&I.d_norm2_gain, cfg.norm2_gain, cfg.C_out)) { I.err = "malloc norm2 gain"; return; }
    if (!upload(&I.d_norm2_bias, cfg.norm2_bias, cfg.C_out)) { I.err = "malloc norm2 bias"; return; }

    // conv1: C_in → C_out, 3x3
    {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = cfg.C_in; c.H_in = cfg.H; c.W_in = cfg.W;
        c.C_out = cfg.C_out; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
        c.W_bf16 = cfg.conv1_W; c.bias_bf16 = cfg.conv1_bias;
        I.conv1 = std::make_unique<Conv2d>(c);
        if (!I.conv1->ok()) { I.err = std::string("conv1: ") + I.conv1->last_error(); return; }
    }
    // conv2: C_out → C_out, 3x3
    {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = cfg.C_out; c.H_in = cfg.H; c.W_in = cfg.W;
        c.C_out = cfg.C_out; c.kH = 3; c.kW = 3; c.stride = 1; c.padding = 1;
        c.W_bf16 = cfg.conv2_W; c.bias_bf16 = cfg.conv2_bias;
        I.conv2 = std::make_unique<Conv2d>(c);
        if (!I.conv2->ok()) { I.err = std::string("conv2: ") + I.conv2->last_error(); return; }
    }
    if (I.have_shortcut) {
        Conv2d::Config c{};
        c.N = cfg.N; c.C_in = cfg.C_in; c.H_in = cfg.H; c.W_in = cfg.W;
        c.C_out = cfg.C_out; c.kH = 1; c.kW = 1; c.stride = 1; c.padding = 0;
        c.W_bf16 = cfg.shortcut_W; c.bias_bf16 = cfg.shortcut_bias;
        I.shortcut = std::make_unique<Conv2d>(c);
        if (!I.shortcut->ok()) { I.err = std::string("shortcut: ") + I.shortcut->last_error(); return; }
    }

    // Workspace plan.
    const size_t in_bytes  = static_cast<size_t>(cfg.N) * cfg.C_in  * cfg.H * cfg.W * sizeof(__nv_bfloat16);
    const size_t out_bytes = static_cast<size_t>(cfg.N) * cfg.C_out * cfg.H * cfg.W * sizeof(__nv_bfloat16);
    size_t c = 0;
    c = align_up(c, ALIGN); I.off_h1 = c; c += in_bytes;
    c = align_up(c, ALIGN); I.off_h2 = c; c += out_bytes;
    c = align_up(c, ALIGN); I.off_h3 = c; c += out_bytes;
    c = align_up(c, ALIGN); I.off_h4 = c; c += out_bytes;
    if (I.have_shortcut) { c = align_up(c, ALIGN); I.off_skip = c; c += out_bytes; }
    c = align_up(c, ALIGN); I.off_conv1_ws = c; c += I.conv1->workspace_size_bytes();
    c = align_up(c, ALIGN); I.off_conv2_ws = c; c += I.conv2->workspace_size_bytes();
    if (I.have_shortcut) {
        c = align_up(c, ALIGN); I.off_shortcut_ws = c; c += I.shortcut->workspace_size_bytes();
    }
    I.total_ws = c;

    I.valid = true;
}

ResnetBlock::~ResnetBlock() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_norm1_gain) cudaFree(I.d_norm1_gain);
    if (I.d_norm1_bias) cudaFree(I.d_norm1_bias);
    if (I.d_norm2_gain) cudaFree(I.d_norm2_gain);
    if (I.d_norm2_bias) cudaFree(I.d_norm2_bias);
}

bool        ResnetBlock::ok()         const { return impl_ && impl_->valid; }
const char* ResnetBlock::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t ResnetBlock::workspace_size_bytes() const { return impl_ ? impl_->total_ws : 0; }

bool ResnetBlock::forward(const void* x, void* y,
                          void* workspace, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.total_ws) { I.err = "workspace too small"; return false; }
    const auto& cfg = I.cfg;

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* h1 = ws + I.off_h1;
    void* h2 = ws + I.off_h2;
    void* h3 = ws + I.off_h3;
    void* h4 = ws + I.off_h4;
    void* skip = I.have_shortcut ? (ws + I.off_skip) : const_cast<void*>(x);
    void* conv1_ws = ws + I.off_conv1_ws;
    void* conv2_ws = ws + I.off_conv2_ws;
    void* shortcut_ws = I.have_shortcut ? ws + I.off_shortcut_ws : nullptr;

    // norm1 + silu
    if (!groupnorm_bf16(x, h1, I.d_norm1_gain, I.d_norm1_bias,
                         cfg.N, cfg.C_in, cfg.H, cfg.W, cfg.num_groups, cfg.eps, stream))
        { I.err = "norm1"; return false; }
    if (!silu_inplace_bf16(h1, static_cast<size_t>(cfg.N)*cfg.C_in*cfg.H*cfg.W, stream))
        { I.err = "silu1"; return false; }

    // conv1
    if (!I.conv1->forward(h1, h2, conv1_ws, I.conv1->workspace_size_bytes(), stream))
        { I.err = std::string("conv1: ") + I.conv1->last_error(); return false; }

    // norm2 + silu
    if (!groupnorm_bf16(h2, h3, I.d_norm2_gain, I.d_norm2_bias,
                         cfg.N, cfg.C_out, cfg.H, cfg.W, cfg.num_groups, cfg.eps, stream))
        { I.err = "norm2"; return false; }
    if (!silu_inplace_bf16(h3, static_cast<size_t>(cfg.N)*cfg.C_out*cfg.H*cfg.W, stream))
        { I.err = "silu2"; return false; }

    // conv2
    if (!I.conv2->forward(h3, h4, conv2_ws, I.conv2->workspace_size_bytes(), stream))
        { I.err = std::string("conv2: ") + I.conv2->last_error(); return false; }

    // shortcut (if channel change)
    if (I.have_shortcut) {
        if (!I.shortcut->forward(x, skip, shortcut_ws, I.shortcut->workspace_size_bytes(), stream))
            { I.err = std::string("shortcut: ") + I.shortcut->last_error(); return false; }
    }

    // y = skip + h4
    const size_t n_elem = static_cast<size_t>(cfg.N) * cfg.C_out * cfg.H * cfg.W;
    const int BLOCK = 256;
    add_bf16_kernel<<<static_cast<int>((n_elem + BLOCK - 1) / BLOCK), BLOCK, 0, stream>>>(
        static_cast<__nv_bfloat16*>(y),
        static_cast<const __nv_bfloat16*>(skip),
        static_cast<const __nv_bfloat16*>(h4),
        n_elem);
    if (cudaPeekAtLastError() != cudaSuccess) { I.err = "add"; return false; }
    return true;
}

} // namespace f2k::cuda
