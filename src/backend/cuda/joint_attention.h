// Joint attention: attention over the concatenation of two sequences
// (typically text + image in an MMDiT block). Internally a thin wrapper
// around Attention with seq = seq_txt + seq_img: copy each modality into
// the concat workspace, run attention, split the output back.
//
// Each tensor is BF16 [batch, seq, n_heads, head_dim]. The user is
// responsible for applying RoPE (or any per-modality preprocessing) to
// Q_img/K_img before calling this — the joint attention itself is identical
// for the two streams once they're laid out together.

#pragma once

#include <cstddef>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class JointAttention {
public:
    struct Config {
        int   batch    = 0;
        int   seq_txt  = 0;
        int   seq_img  = 0;
        int   n_heads  = 0;
        int   head_dim = 0;
        float scale    = 0.0f;   // 0 ⇒ 1/sqrt(head_dim)
    };

    explicit JointAttention(const Config& cfg);
    ~JointAttention();

    bool        ok()         const;
    const char* last_error() const;

    size_t workspace_size_bytes() const;

    // All device pointers.
    //   Q_txt/K_txt/V_txt: [batch, seq_txt, n_heads, head_dim] BF16
    //   Q_img/K_img/V_img: [batch, seq_img, n_heads, head_dim] BF16
    //   O_txt:             [batch, seq_txt, n_heads, head_dim] BF16
    //   O_img:             [batch, seq_img, n_heads, head_dim] BF16
    bool forward(const void* Q_txt, const void* K_txt, const void* V_txt,
                 const void* Q_img, const void* K_img, const void* V_img,
                 void* O_txt, void* O_img,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
