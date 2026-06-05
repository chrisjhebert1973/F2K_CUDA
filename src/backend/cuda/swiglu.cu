// SwiGLU MLP — orchestrates three Linears + a fused silu*mul kernel.

#include "backend/cuda/swiglu.h"
#include "backend/cuda/linear.h"
#include "backend/cuda/kernels/silu_mul.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <memory>
#include <string>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

} // anonymous namespace

namespace f2k::cuda {

struct SwiGLU::Impl {
    int  batch_rows = 0;
    int  hidden_dim = 0;
    int  ffn_dim    = 0;
    bool valid      = false;
    std::string err;

    std::unique_ptr<Linear> gate;
    std::unique_ptr<Linear> up;
    std::unique_ptr<Linear> down;

    // Workspace partitioning:
    //   gate_out [batch_rows, ffn_dim]   BF16
    //   up_out   [batch_rows, ffn_dim]   BF16
    //   fused    [batch_rows, ffn_dim]   BF16  (silu(gate) * up; reuses gate_out via overlap? keep separate)
    //   gate_ws  Linear::workspace
    //   up_ws    Linear::workspace
    //   down_ws  Linear::workspace
    size_t off_gate_out = 0, off_up_out = 0, off_fused = 0;
    size_t off_gate_ws  = 0, off_up_ws  = 0, off_down_ws = 0;
    size_t bytes_intermediate = 0;
    size_t total_workspace = 0;
};

SwiGLU::SwiGLU(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.batch_rows = cfg.batch_rows;
    I.hidden_dim = cfg.hidden_dim;
    I.ffn_dim    = cfg.ffn_dim;

    if (I.batch_rows % 128 != 0 || I.ffn_dim % 128 != 0 || I.hidden_dim % 64 != 0) {
        I.err = "SwiGLU: batch_rows,ffn_dim must be /128; hidden_dim /64";
        return;
    }
    if (!cfg.W_gate_bf16 || !cfg.W_up_bf16 || !cfg.W_down_bf16) {
        I.err = "SwiGLU: a weight pointer is null";
        return;
    }

    {
        Linear::Config c;
        c.batch_rows   = I.batch_rows;
        c.in_features  = I.hidden_dim;
        c.out_features = I.ffn_dim;
        c.W_bf16       = cfg.W_gate_bf16;
        I.gate = std::make_unique<Linear>(c);
        if (!I.gate->ok()) { I.err = std::string("gate: ") + I.gate->last_error(); return; }
    }
    {
        Linear::Config c;
        c.batch_rows   = I.batch_rows;
        c.in_features  = I.hidden_dim;
        c.out_features = I.ffn_dim;
        c.W_bf16       = cfg.W_up_bf16;
        I.up = std::make_unique<Linear>(c);
        if (!I.up->ok()) { I.err = std::string("up: ") + I.up->last_error(); return; }
    }
    {
        Linear::Config c;
        c.batch_rows   = I.batch_rows;
        c.in_features  = I.ffn_dim;
        c.out_features = I.hidden_dim;
        c.W_bf16       = cfg.W_down_bf16;
        I.down = std::make_unique<Linear>(c);
        if (!I.down->ok()) { I.err = std::string("down: ") + I.down->last_error(); return; }
    }

    const size_t ffn_buf =
        static_cast<size_t>(I.batch_rows) * I.ffn_dim * sizeof(__nv_bfloat16);
    I.off_gate_out = 0;
    I.off_up_out   = align_up(I.off_gate_out + ffn_buf, ALIGN);
    I.off_fused    = align_up(I.off_up_out   + ffn_buf, ALIGN);
    I.off_gate_ws  = align_up(I.off_fused    + ffn_buf, ALIGN);
    I.off_up_ws    = align_up(I.off_gate_ws  + I.gate->workspace_size_bytes(), ALIGN);
    I.off_down_ws  = align_up(I.off_up_ws    + I.up->workspace_size_bytes(),   ALIGN);
    I.total_workspace = I.off_down_ws + I.down->workspace_size_bytes();
    I.bytes_intermediate = 3 * ffn_buf;
    I.valid = true;
}

SwiGLU::~SwiGLU() = default;

bool        SwiGLU::ok()          const { return impl_ && impl_->valid; }
const char* SwiGLU::last_error()  const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t SwiGLU::workspace_size_bytes() const {
    return impl_ ? impl_->total_workspace : 0;
}

bool SwiGLU::forward(const void* x_bf16, void* y_bf16,
                     void* workspace, size_t workspace_size,
                     cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.total_workspace) {
        I.err = "workspace too small";
        return false;
    }
    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* gate_out = ws + I.off_gate_out;
    void* up_out   = ws + I.off_up_out;
    void* fused    = ws + I.off_fused;
    void* gate_ws  = ws + I.off_gate_ws;
    void* up_ws    = ws + I.off_up_ws;
    void* down_ws  = ws + I.off_down_ws;

    if (!I.gate->forward(x_bf16, gate_out, gate_ws,
                         I.gate->workspace_size_bytes(), stream)) {
        I.err = std::string("gate.forward: ") + I.gate->last_error();
        return false;
    }
    if (!I.up->forward(x_bf16, up_out, up_ws,
                       I.up->workspace_size_bytes(), stream)) {
        I.err = std::string("up.forward: ") + I.up->last_error();
        return false;
    }

    const size_t n_elem = static_cast<size_t>(I.batch_rows) * I.ffn_dim;
    if (!silu_mul_bf16(gate_out, up_out, fused, n_elem, stream)) {
        I.err = "silu_mul launch";
        return false;
    }

    if (!I.down->forward(fused, y_bf16, down_ws,
                         I.down->workspace_size_bytes(), stream)) {
        I.err = std::string("down.forward: ") + I.down->last_error();
        return false;
    }
    return true;
}

} // namespace f2k::cuda
