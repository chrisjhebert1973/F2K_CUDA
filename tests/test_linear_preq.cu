// Linear pre-quantized weight path test.
//
// Strategy: pick a single random BF16 W. Quantize it once via in-memory NVFP4
// (same logic as the converter). Build TWO Linear instances:
//   A: from BF16 → quantizes internally
//   B: from PreQuantNVFP4 (the bytes we just produced) → skips quantization
//
// Run the same activation through both. Outputs must be bit-identical: both
// paths route the same source bytes through the same CuTe-tensor write into
// the SFB layout, and the GEMM is deterministic.

#include "backend/cuda/linear.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <random>
#include <string>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

struct PreQ {
    std::vector<uint8_t> packed;
    std::vector<uint8_t> scales;
    int N, K;
};

// Replicates the converter's quantization logic for B (column-major weight).
// Output: [N, K/2] row-major packed FP4 + [N, K/16] row-major E4M3 scales.
PreQ quantize_for_storage(const std::vector<__nv_bfloat16>& W_bf16, int N, int K) {
    PreQ p;
    p.N = N; p.K = K;
    p.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
    p.scales.assign(static_cast<size_t>(N) * (K / 16), 0);
    constexpr float E2M1_MAX = 6.0f;
    const int n_kblocks = K / 16;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            float absmax = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float v = bf16_to_fp32(W_bf16[static_cast<size_t>(n) * K + kb * 16 + i]);
                absmax = std::max(absmax, std::fabs(v));
            }
            const float scale = (absmax > 1e-30f) ? (absmax / E2M1_MAX) : 1.0f;
            cutlass::float_ue4m3_t sf(scale);
            p.scales[static_cast<size_t>(n) * n_kblocks + kb] = static_cast<uint8_t>(sf.raw());
            for (int i = 0; i < 16; ++i) {
                const int k = kb * 16 + i;
                const float v = bf16_to_fp32(W_bf16[static_cast<size_t>(n) * K + k]);
                cutlass::float_e2m1_t e2m1(v / scale);
                const uint8_t nibble = static_cast<uint8_t>(e2m1.raw()) & 0xFu;
                const size_t byte_idx = static_cast<size_t>(n) * (K / 2) + (k >> 1);
                if ((k & 1) == 0) {
                    p.packed[byte_idx] = (p.packed[byte_idx] & 0xF0u) | nibble;
                } else {
                    p.packed[byte_idx] = (p.packed[byte_idx] & 0x0Fu) | static_cast<uint8_t>(nibble << 4);
                }
            }
        }
    }
    return p;
}

bool run_case(int M, int N, int K) {
    std::printf("LinearPreQ M=%-4d N=%-4d K=%-4d  ", M, N, K);

    std::mt19937 rng(0xCAFEFEEDULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hW(static_cast<size_t>(N) * K),
                                 hx(static_cast<size_t>(M) * K);
    for (auto& v : hW) v = fp32_to_bf16(d(rng));
    for (auto& v : hx) v = fp32_to_bf16(d(rng));

    // Path A: Linear from BF16 (existing path).
    f2k::cuda::Linear::Config cfgA;
    cfgA.batch_rows = M; cfgA.in_features = K; cfgA.out_features = N;
    cfgA.W_bf16 = hW.data();
    f2k::cuda::Linear linA(cfgA);
    if (!linA.ok()) { std::printf("FAIL A ctor: %s\n", linA.last_error()); return false; }

    // Path B: pre-quantize once and feed in via W_preq.
    PreQ pq = quantize_for_storage(hW, N, K);
    f2k::cuda::PreQuantNVFP4 preq{};
    preq.packed = pq.packed.data();
    preq.scales = pq.scales.data();
    preq.N = N; preq.K = K;
    f2k::cuda::Linear::Config cfgB;
    cfgB.batch_rows = M; cfgB.in_features = K; cfgB.out_features = N;
    cfgB.W_preq = &preq;
    f2k::cuda::Linear linB(cfgB);
    if (!linB.ok()) { std::printf("FAIL B ctor: %s\n", linB.last_error()); return false; }

    // Run both with the same activation.
    void *d_x, *d_yA, *d_yB, *d_wsA, *d_wsB;
    const size_t x_bytes = hx.size() * 2;
    const size_t y_bytes = static_cast<size_t>(M) * N * 2;
    cudaMalloc(&d_x,   x_bytes);
    cudaMalloc(&d_yA,  y_bytes);
    cudaMalloc(&d_yB,  y_bytes);
    cudaMalloc(&d_wsA, linA.workspace_size_bytes());
    cudaMalloc(&d_wsB, linB.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), x_bytes, cudaMemcpyHostToDevice);

    const bool okA = linA.forward(d_x, d_yA, d_wsA, linA.workspace_size_bytes());
    const bool okB = linB.forward(d_x, d_yB, d_wsB, linB.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!okA || !okB) {
        std::printf("FAIL forward (A=%d B=%d)\n", okA, okB);
        return false;
    }

    std::vector<__nv_bfloat16> hyA(static_cast<size_t>(M) * N), hyB(hyA.size());
    cudaMemcpy(hyA.data(), d_yA, y_bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hyB.data(), d_yB, y_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_yA); cudaFree(d_yB); cudaFree(d_wsA); cudaFree(d_wsB);

    // Outputs must be byte-identical: same source bits → same GEMM result.
    const bool bit_identical =
        std::memcmp(hyA.data(), hyB.data(), y_bytes) == 0;
    double max_abs_diff = 0;
    for (size_t i = 0; i < hyA.size(); ++i) {
        max_abs_diff = std::max(max_abs_diff,
            (double)std::fabs(bf16_to_fp32(hyA[i]) - bf16_to_fp32(hyB[i])));
    }
    std::printf("%s  bit_identical=%s  max_abs_diff=%.6f\n",
                bit_identical ? "PASS" : "FAIL",
                bit_identical ? "yes" : "no",
                max_abs_diff);
    return bit_identical;
}

} // namespace

// Real-model smoke: load one Linear weight from F2K, run forward, verify
// non-NaN and sane magnitudes. Skipped if the converted shards don't exist.
bool real_model_smoke() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("Real-model smoke: SKIPPED (converted shards not present)\n");
        return true;
    }
    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string())) {
        std::fprintf(stderr, "loader failed: %s\n", ld.last_error().c_str());
        return false;
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) {
        std::fprintf(stderr, "router failed: %s\n", router.last_error().c_str());
        return false;
    }
    const auto* w = router.double_blocks()[0].to_q;   // [4096, 4096] NVFP4

    f2k::cuda::PreQuantNVFP4 preq{};
    preq.packed = w->data;
    preq.scales = w->scales;
    preq.N = static_cast<int>(w->shape[0]);
    preq.K = static_cast<int>(w->shape[1]);
    preq.microblock_size = w->microblock_size ? w->microblock_size : 16;
    preq.tensor_scale = w->tensor_scale;

    f2k::cuda::Linear::Config cfg;
    cfg.batch_rows = 128;
    cfg.in_features  = preq.K;
    cfg.out_features = preq.N;
    cfg.W_preq = &preq;
    f2k::cuda::Linear lin(cfg);
    if (!lin.ok()) { std::fprintf(stderr, "real-W ctor: %s\n", lin.last_error()); return false; }

    std::mt19937 rng(0xBADF00DULL);
    std::uniform_real_distribution<float> d(-1.0f, 1.0f);
    std::vector<__nv_bfloat16> hx(128 * preq.K);
    for (auto& v : hx) v = fp32_to_bf16(d(rng));

    void *d_x, *d_y, *d_ws;
    const size_t x_bytes = hx.size() * 2;
    const size_t y_bytes = 128 * preq.N * 2;
    cudaMalloc(&d_x, x_bytes);
    cudaMalloc(&d_y, y_bytes);
    cudaMalloc(&d_ws, lin.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), x_bytes, cudaMemcpyHostToDevice);
    const bool ok = lin.forward(d_x, d_y, d_ws, lin.workspace_size_bytes());
    cudaDeviceSynchronize();
    std::vector<__nv_bfloat16> hy(128 * preq.N);
    cudaMemcpy(hy.data(), d_y, y_bytes, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
    if (!ok) { std::fprintf(stderr, "real-W forward: %s\n", lin.last_error()); return false; }

    int n_bad = 0;
    double max_abs = 0, sum_abs = 0;
    for (auto v : hy) {
        const float f = bf16_to_fp32(v);
        if (!std::isfinite(f)) ++n_bad;
        max_abs = std::max(max_abs, (double)std::fabs(f));
        sum_abs += std::fabs(f);
    }
    const double mean_abs = sum_abs / hy.size();
    const bool pass = n_bad == 0 && mean_abs > 0.01 && max_abs < 100.0;
    std::printf("Real-model smoke [transformer_blocks.0.attn.to_q, %dx%d]:  %s "
                "nans=%d max=%.3f mean=%.3f\n",
                preq.N, preq.K, pass ? "PASS" : "FAIL", n_bad, max_abs, mean_abs);
    return pass;
}

int main() {
    bool ok = true;
    ok &= run_case(128, 128, 64);
    ok &= run_case(128, 256, 128);
    ok &= run_case(256, 512, 256);
    ok &= real_model_smoke();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
