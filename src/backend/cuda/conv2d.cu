// Conv2d — cuDNN conv forward + bias add, BF16 NCHW.
//
// Uses the legacy cuDNN v8/v9 convolution API (cudnnConvolutionForward) — the
// simplest path for a static set of conv shapes. Modern graph API would buy
// us fusions (conv+bias+activation in one node) but adds significant code.

#include "backend/cuda/conv2d.h"

#include <cudnn.h>
#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

namespace f2k::cuda {

#define CUDNN_CHECK(x, label)                                                       \
    do {                                                                            \
        cudnnStatus_t _s = (x);                                                     \
        if (_s != CUDNN_STATUS_SUCCESS) {                                           \
            I.err = std::string(label) + ": " + cudnnGetErrorString(_s);            \
            return;                                                                 \
        }                                                                           \
    } while (0)

struct Conv2d::Impl {
    Config cfg{};
    bool   valid = false;
    std::string err;

    cudnnHandle_t handle = nullptr;
    cudnnTensorDescriptor_t  x_desc = nullptr;
    cudnnTensorDescriptor_t  y_desc = nullptr;
    cudnnTensorDescriptor_t  b_desc = nullptr;
    cudnnFilterDescriptor_t  w_desc = nullptr;
    cudnnConvolutionDescriptor_t conv_desc = nullptr;
    cudnnConvolutionFwdAlgo_t algo{};
    size_t workspace_bytes = 0;

    int H_out = 0, W_out = 0;
    bool have_bias = false;

    void* d_W    = nullptr;
    void* d_bias = nullptr;
};

Conv2d::Conv2d(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.cfg = cfg;

    if (cfg.N <= 0 || cfg.C_in <= 0 || cfg.H_in <= 0 || cfg.W_in <= 0 ||
        cfg.C_out <= 0 || cfg.kH <= 0 || cfg.kW <= 0) {
        I.err = "Conv2d: bad dims"; return;
    }
    if (!cfg.W_bf16) { I.err = "Conv2d: W is null"; return; }
    I.have_bias = (cfg.bias_bf16 != nullptr);

    I.H_out = (cfg.H_in + 2 * cfg.padding - cfg.kH) / cfg.stride + 1;
    I.W_out = (cfg.W_in + 2 * cfg.padding - cfg.kW) / cfg.stride + 1;

    CUDNN_CHECK(cudnnCreate(&I.handle), "cudnnCreate");
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&I.x_desc), "create x desc");
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&I.y_desc), "create y desc");
    CUDNN_CHECK(cudnnCreateTensorDescriptor(&I.b_desc), "create b desc");
    CUDNN_CHECK(cudnnCreateFilterDescriptor(&I.w_desc), "create w desc");
    CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&I.conv_desc), "create conv desc");

    CUDNN_CHECK(cudnnSetTensor4dDescriptor(I.x_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_BFLOAT16,
                                            cfg.N, cfg.C_in, cfg.H_in, cfg.W_in),
                "set x desc");
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(I.y_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_BFLOAT16,
                                            cfg.N, cfg.C_out, I.H_out, I.W_out),
                "set y desc");
    CUDNN_CHECK(cudnnSetTensor4dDescriptor(I.b_desc, CUDNN_TENSOR_NCHW,
                                            CUDNN_DATA_BFLOAT16,
                                            1, cfg.C_out, 1, 1),
                "set b desc");
    CUDNN_CHECK(cudnnSetFilter4dDescriptor(I.w_desc, CUDNN_DATA_BFLOAT16,
                                            CUDNN_TENSOR_NCHW,
                                            cfg.C_out, cfg.C_in, cfg.kH, cfg.kW),
                "set w desc");
    CUDNN_CHECK(cudnnSetConvolution2dDescriptor(I.conv_desc,
                                                  cfg.padding, cfg.padding,
                                                  cfg.stride, cfg.stride,
                                                  /*dilation*/ 1, 1,
                                                  CUDNN_CROSS_CORRELATION,
                                                  CUDNN_DATA_FLOAT),
                "set conv desc");
    // BF16 compute can be enabled but FP32 accumulator + BF16 IO is the standard
    // pattern; we keep math_type at the default which already promotes to fp32.
    cudnnSetConvolutionMathType(I.conv_desc, CUDNN_TENSOR_OP_MATH);

    // Pick algorithm: ask cuDNN to recommend the fastest for this configuration.
    int returned = 0;
    cudnnConvolutionFwdAlgoPerf_t perf[8];
    CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            I.handle, I.x_desc, I.w_desc, I.conv_desc, I.y_desc,
            /*requested*/ 8, &returned, perf),
        "get algo");
    if (returned <= 0) { I.err = "no conv algo"; return; }
    I.algo = perf[0].algo;
    I.workspace_bytes = perf[0].memory;

    // Upload W + bias.
    const size_t w_bytes = static_cast<size_t>(cfg.C_out) * cfg.C_in * cfg.kH * cfg.kW * sizeof(__nv_bfloat16);
    if (cudaMalloc(&I.d_W, w_bytes) != cudaSuccess) { I.err = "malloc W"; return; }
    cudaMemcpy(I.d_W, cfg.W_bf16, w_bytes, cudaMemcpyHostToDevice);
    if (I.have_bias) {
        const size_t b_bytes = static_cast<size_t>(cfg.C_out) * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.d_bias, b_bytes) != cudaSuccess) { I.err = "malloc bias"; return; }
        cudaMemcpy(I.d_bias, cfg.bias_bf16, b_bytes, cudaMemcpyHostToDevice);
    }

    I.valid = true;
}

Conv2d::~Conv2d() {
    if (!impl_) return;
    Impl& I = *impl_;
    if (I.d_W)         cudaFree(I.d_W);
    if (I.d_bias)      cudaFree(I.d_bias);
    if (I.x_desc)      cudnnDestroyTensorDescriptor(I.x_desc);
    if (I.y_desc)      cudnnDestroyTensorDescriptor(I.y_desc);
    if (I.b_desc)      cudnnDestroyTensorDescriptor(I.b_desc);
    if (I.w_desc)      cudnnDestroyFilterDescriptor(I.w_desc);
    if (I.conv_desc)   cudnnDestroyConvolutionDescriptor(I.conv_desc);
    if (I.handle)      cudnnDestroy(I.handle);
}

bool        Conv2d::ok()         const { return impl_ && impl_->valid; }
const char* Conv2d::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
int    Conv2d::H_out() const { return impl_ ? impl_->H_out : 0; }
int    Conv2d::W_out() const { return impl_ ? impl_->W_out : 0; }
size_t Conv2d::workspace_size_bytes() const { return impl_ ? impl_->workspace_bytes : 0; }

bool Conv2d::forward(const void* x, void* y,
                     void* workspace, size_t ws_size, cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (ws_size < I.workspace_bytes) { I.err = "workspace too small"; return false; }

    cudnnSetStream(I.handle, stream);

    const float alpha = 1.0f;
    const float beta  = 0.0f;
    cudnnStatus_t st = cudnnConvolutionForward(
        I.handle,
        &alpha,
        I.x_desc, x,
        I.w_desc, I.d_W,
        I.conv_desc, I.algo,
        workspace, I.workspace_bytes,
        &beta,
        I.y_desc, y);
    if (st != CUDNN_STATUS_SUCCESS) {
        I.err = std::string("cudnnConvolutionForward: ") + cudnnGetErrorString(st);
        return false;
    }

    if (I.have_bias) {
        const float alpha2 = 1.0f;
        const float beta2  = 1.0f;
        st = cudnnAddTensor(I.handle, &alpha2, I.b_desc, I.d_bias,
                             &beta2,  I.y_desc, y);
        if (st != CUDNN_STATUS_SUCCESS) {
            I.err = std::string("cudnnAddTensor(bias): ") + cudnnGetErrorString(st);
            return false;
        }
    }
    return true;
}

} // namespace f2k::cuda
