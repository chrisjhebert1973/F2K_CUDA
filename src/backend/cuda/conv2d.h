// Conv2d — thin wrapper around cuDNN's convolution forward + bias add for
// the FLUX.2 VAE decoder.
//
// All tensors NCHW BF16; weights [C_out, C_in, kH, kW] BF16; bias [C_out] BF16.
// Single-stride, single-padding, dilation=1, no grouped conv. 3x3 and 1x1 are
// the only kernel sizes the VAE uses.
//
// Owned at construction: cuDNN descriptors (tensor / filter / conv), chosen
// algorithm, and the device-side workspace size. The weight + bias themselves
// are uploaded to device buffers we own.

#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

class Conv2d {
public:
    struct Config {
        int N = 0, C_in = 0, H_in = 0, W_in = 0;
        int C_out = 0;
        int kH = 3, kW = 3;
        int stride = 1;
        int padding = 1;
        const void* W_bf16    = nullptr;   // [C_out, C_in, kH, kW] BF16, host
        const void* bias_bf16 = nullptr;   // [C_out] BF16, host (optional)
    };

    explicit Conv2d(const Config& cfg);
    ~Conv2d();

    Conv2d(const Conv2d&)            = delete;
    Conv2d& operator=(const Conv2d&) = delete;

    bool        ok()         const;
    const char* last_error() const;

    int H_out() const;
    int W_out() const;
    size_t workspace_size_bytes() const;

    // x: [N, C_in, H_in, W_in]  BF16 device
    // y: [N, C_out, H_out, W_out] BF16 device
    bool forward(const void* x_bf16, void* y_bf16,
                 void* workspace, size_t workspace_size,
                 cudaStream_t stream = nullptr);

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace f2k::cuda
