// JointAttention: concat-text-and-image → attention → split.

#include "backend/cuda/joint_attention.h"
#include "backend/cuda/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <memory>
#include <string>

namespace {

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }
constexpr size_t ALIGN = 256;

// Concatenate two [batch, S, H, D] tensors along S into a [batch, S_a+S_b, H, D] output.
__global__ void concat_seq_kernel(const __nv_bfloat16* __restrict__ A,
                                  const __nv_bfloat16* __restrict__ B,
                                  __nv_bfloat16* __restrict__ out,
                                  int batch, int S_a, int S_b, int HD) {
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    if (s >= S_a + S_b || b >= batch) return;
    const size_t row_out = (static_cast<size_t>(b) * (S_a + S_b) + s) * HD;
    if (s < S_a) {
        const size_t row_in = (static_cast<size_t>(b) * S_a + s) * HD;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) {
            out[row_out + i] = A[row_in + i];
        }
    } else {
        const size_t row_in = (static_cast<size_t>(b) * S_b + (s - S_a)) * HD;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) {
            out[row_out + i] = B[row_in + i];
        }
    }
}

// Inverse: split a [batch, S_a+S_b, H, D] tensor into two outputs of
// [batch, S_a, H, D] and [batch, S_b, H, D].
__global__ void split_seq_kernel(const __nv_bfloat16* __restrict__ in,
                                 __nv_bfloat16* __restrict__ A,
                                 __nv_bfloat16* __restrict__ B,
                                 int batch, int S_a, int S_b, int HD) {
    const int s = blockIdx.x;
    const int b = blockIdx.y;
    if (s >= S_a + S_b || b >= batch) return;
    const size_t row_in = (static_cast<size_t>(b) * (S_a + S_b) + s) * HD;
    if (s < S_a) {
        const size_t row_out = (static_cast<size_t>(b) * S_a + s) * HD;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) {
            A[row_out + i] = in[row_in + i];
        }
    } else {
        const size_t row_out = (static_cast<size_t>(b) * S_b + (s - S_a)) * HD;
        for (int i = threadIdx.x; i < HD; i += blockDim.x) {
            B[row_out + i] = in[row_in + i];
        }
    }
}

} // anonymous namespace

namespace f2k::cuda {

struct JointAttention::Impl {
    int batch = 0, seq_txt = 0, seq_img = 0, n_heads = 0, head_dim = 0;
    bool valid = false;
    std::string err;

    std::unique_ptr<Attention> attn;

    size_t off_Q = 0, off_K = 0, off_V = 0, off_O = 0;
    size_t total_ws = 0;
};

JointAttention::JointAttention(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    auto fail = [&](std::string s) { I.err = std::move(s); };

    if (cfg.batch <= 0 || cfg.seq_txt <= 0 || cfg.seq_img <= 0 ||
        cfg.n_heads <= 0 || cfg.head_dim <= 0) {
        fail("JointAttention: zero/negative dim"); return;
    }

    I.batch    = cfg.batch;
    I.seq_txt  = cfg.seq_txt;
    I.seq_img  = cfg.seq_img;
    I.n_heads  = cfg.n_heads;
    I.head_dim = cfg.head_dim;

    Attention::Config ac{};
    ac.batch    = cfg.batch;
    ac.seq      = cfg.seq_txt + cfg.seq_img;
    ac.n_heads  = cfg.n_heads;
    ac.head_dim = cfg.head_dim;
    ac.scale    = cfg.scale;
    I.attn = std::make_unique<Attention>(ac);
    if (!I.attn->ok()) { fail(std::string("inner attn: ") + I.attn->last_error()); return; }

    const size_t bsxhd =
        static_cast<size_t>(cfg.batch) * (cfg.seq_txt + cfg.seq_img) *
        cfg.n_heads * cfg.head_dim * sizeof(__nv_bfloat16);
    I.off_Q = 0;
    I.off_K = align_up(I.off_Q + bsxhd, ALIGN);
    I.off_V = align_up(I.off_K + bsxhd, ALIGN);
    I.off_O = align_up(I.off_V + bsxhd, ALIGN);
    I.total_ws = I.off_O + bsxhd;

    I.valid = true;
}

JointAttention::~JointAttention() = default;

bool        JointAttention::ok()         const { return impl_ && impl_->valid; }
const char* JointAttention::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t JointAttention::workspace_size_bytes() const {
    return impl_ ? impl_->total_ws : 0;
}

bool JointAttention::forward(const void* Q_txt, const void* K_txt, const void* V_txt,
                             const void* Q_img, const void* K_img, const void* V_img,
                             void* O_txt, void* O_img,
                             void* workspace, size_t workspace_size,
                             cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.total_ws) { I.err = "workspace too small"; return false; }

    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* Q_concat = ws + I.off_Q;
    void* K_concat = ws + I.off_K;
    void* V_concat = ws + I.off_V;
    void* O_concat = ws + I.off_O;

    const int HD = I.n_heads * I.head_dim;
    const int S_total = I.seq_txt + I.seq_img;
    dim3 grid(S_total, I.batch), block(256);

    auto concat = [&](const void* A, const void* B, void* out) {
        concat_seq_kernel<<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(A),
            static_cast<const __nv_bfloat16*>(B),
            static_cast<      __nv_bfloat16*>(out),
            I.batch, I.seq_txt, I.seq_img, HD);
        return cudaPeekAtLastError() == cudaSuccess;
    };
    if (!concat(Q_txt, Q_img, Q_concat)) { I.err = "concat Q"; return false; }
    if (!concat(K_txt, K_img, K_concat)) { I.err = "concat K"; return false; }
    if (!concat(V_txt, V_img, V_concat)) { I.err = "concat V"; return false; }

    if (!I.attn->forward(Q_concat, K_concat, V_concat, O_concat, stream)) {
        I.err = std::string("inner attn forward: ") + I.attn->last_error();
        return false;
    }

    split_seq_kernel<<<grid, block, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(O_concat),
        static_cast<      __nv_bfloat16*>(O_txt),
        static_cast<      __nv_bfloat16*>(O_img),
        I.batch, I.seq_txt, I.seq_img, HD);
    if (cudaPeekAtLastError() != cudaSuccess) { I.err = "split O"; return false; }
    return true;
}

} // namespace f2k::cuda
