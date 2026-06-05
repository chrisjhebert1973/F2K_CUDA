// Linear class smoke test.
//
// Current state (2026-06-03): Linear's plumbing (GEMM call, layout setup,
// quant write paths, bias add) is verified for inputs where each NVFP4
// microblock contains constant values. Random per-element data within a
// microblock produces uncorrelated output — a CUTLASS layout/byte convention
// I haven't fully reverse-engineered. See [[feedback-cutlass-nvfp4-linear-gap]]
// in memory for the diagnostic ladder we ran.
//
// Until that's fixed, this test exercises the cases that DO work end-to-end:
//   - All ones
//   - W constant along K with one varying row
//   - W constant within microblocks but varying between microblocks

#include "backend/cuda/linear.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <random>
#include <vector>

namespace {

inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

bool run_one(const char* label, int M, int N, int K,
             const std::vector<__nv_bfloat16>& hx,
             const std::vector<__nv_bfloat16>& hW,
             float expected_val, float tol) {
    std::printf("%-44s ", label);
    f2k::cuda::Linear::Config cfg;
    cfg.batch_rows = M; cfg.in_features = K; cfg.out_features = N;
    cfg.W_bf16 = hW.data();

    f2k::cuda::Linear lin(cfg);
    if (!lin.ok()) { std::printf("FAIL (ctor: %s)\n", lin.last_error()); return false; }

    void *d_x, *d_y, *d_ws;
    cudaMalloc(&d_x,  hx.size() * 2);
    cudaMalloc(&d_y,  static_cast<size_t>(M) * N * 2);
    cudaMalloc(&d_ws, lin.workspace_size_bytes());
    cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
    const bool ok = lin.forward(d_x, d_y, d_ws, lin.workspace_size_bytes());
    cudaDeviceSynchronize();
    if (!ok) {
        std::printf("FAIL (forward: %s)\n", lin.last_error());
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
        return false;
    }
    std::vector<__nv_bfloat16> hy(static_cast<size_t>(M) * N);
    cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);

    // Check uniformity AND magnitude. We expect every output cell to be
    // expected_val within tol (allowing for FP4 scale-rounding overshoot).
    double max_dev = 0.0;
    for (auto v : hy) max_dev = std::max(max_dev, (double)std::fabs(bf16_to_fp32(v) - expected_val));
    const bool pass = max_dev < tol;
    std::printf("%s  y=%.2f exp=%.2f max_dev=%.3f tol=%.2f\n",
                pass ? "PASS" : "FAIL",
                bf16_to_fp32(hy[0]), expected_val, max_dev, tol);
    return pass;
}

} // namespace

int main() {
    bool ok = true;

    // Case 0a: x varies within microblock (K-axis alternation in x)
    {
        const int M = 128, N = 128, K = 128;
        std::vector<__nv_bfloat16> hx(M * K), hW(N * K, fp32_to_bf16(1.0f));
        for (int m = 0; m < M; ++m)
            for (int k = 0; k < K; ++k)
                hx[m*K + k] = fp32_to_bf16((k & 1) ? 0.5f : 1.0f);
        ok &= run_one("Linear x-alternating-within-block (K-axis)",
                      M, N, K, hx, hW, 0.75f * K, /*tol*/12.0f);
    }

    // Case 0b: W varies within microblock (K-axis alternation in W)
    //   x[m, k] = 1, W[n, k even] = 1, W[n, k odd] = 0.5
    //   Same expected = 0.75 * K = 96
    {
        const int M = 128, N = 128, K = 128;
        std::vector<__nv_bfloat16> hx(M * K, fp32_to_bf16(1.0f));
        std::vector<__nv_bfloat16> hW(N * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                hW[n*K + k] = fp32_to_bf16((k & 1) ? 0.5f : 1.0f);
        ok &= run_one("Linear W-alternating-within-block (K-axis)",
                      M, N, K, hx, hW, 0.75f * K, /*tol*/12.0f);
    }

    // Case 0c: W varies within microblock along N (n alternation)
    //   x[m, k] = 1, W[n even, k] = 1, W[n odd, k] = 0.5
    //   For each m: y[m, n even] = K (= sum of 1s) = 128
    //               y[m, n odd ] = 0.5 * K = 64
    {
        const int M = 128, N = 128, K = 128;
        std::vector<__nv_bfloat16> hx(M * K, fp32_to_bf16(1.0f));
        std::vector<__nv_bfloat16> hW(N * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                hW[n*K + k] = fp32_to_bf16((n & 1) ? 0.5f : 1.0f);
        f2k::cuda::Linear::Config cfg;
        cfg.batch_rows = M; cfg.in_features = K; cfg.out_features = N;
        cfg.W_bf16 = hW.data();
        f2k::cuda::Linear lin(cfg);
        void *d_x, *d_y, *d_ws;
        cudaMalloc(&d_x,  hx.size() * 2);
        cudaMalloc(&d_y,  static_cast<size_t>(M) * N * 2);
        cudaMalloc(&d_ws, lin.workspace_size_bytes());
        cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        lin.forward(d_x, d_y, d_ws, lin.workspace_size_bytes());
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> hy(static_cast<size_t>(M) * N);
        cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);
        const float y00 = bf16_to_fp32(hy[0]);
        const float y01 = bf16_to_fp32(hy[1]);
        const float y02 = bf16_to_fp32(hy[2]);
        const float y03 = bf16_to_fp32(hy[3]);
        std::printf("%-44s y[*,0..3]=%.2f %.2f %.2f %.2f (exp 128 64 128 64)\n",
            "Linear W-alternating (N-axis)", y00, y01, y02, y03);
        const bool pass =
            std::fabs(y00 - 128.0f) < 12.0f && std::fabs(y01 - 64.0f) < 8.0f &&
            std::fabs(y02 - 128.0f) < 12.0f && std::fabs(y03 - 64.0f) < 8.0f;
        ok &= pass;
    }

    // Case 1: all ones → y[m, n] = K (per element FP4 noise allowed).
    {
        const int M = 128, N = 128, K = 128;
        std::vector<__nv_bfloat16> hx(M * K, fp32_to_bf16(1.0f));
        std::vector<__nv_bfloat16> hW(N * K, fp32_to_bf16(1.0f));
        ok &= run_one("Linear all-ones M=N=K=128", M, N, K, hx, hW,
                      /*expect*/128.0f, /*tol*/12.0f);  // ~9% overhead
    }

    // Case 2: row-varying W (W[n=0]=2, others=1), x=1 → y[m, 0]=2K, else K.
    //   Run twice via two Linear instances, since each tests a single cell.
    //   We check the dominant value at n=0 and at n=5 separately.
    {
        const int M = 128, N = 128, K = 128;
        std::vector<__nv_bfloat16> hx(M * K, fp32_to_bf16(1.0f));
        std::vector<__nv_bfloat16> hW(N * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                hW[n*K + k] = fp32_to_bf16((n == 0) ? 2.0f : 1.0f);
        f2k::cuda::Linear::Config cfg;
        cfg.batch_rows = M; cfg.in_features = K; cfg.out_features = N;
        cfg.W_bf16 = hW.data();
        f2k::cuda::Linear lin(cfg);
        void *d_x, *d_y, *d_ws;
        cudaMalloc(&d_x,  hx.size() * 2);
        cudaMalloc(&d_y,  static_cast<size_t>(M) * N * 2);
        cudaMalloc(&d_ws, lin.workspace_size_bytes());
        cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        lin.forward(d_x, d_y, d_ws, lin.workspace_size_bytes());
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> hy(static_cast<size_t>(M) * N);
        cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);

        const float y00 = bf16_to_fp32(hy[0]);
        const float y05 = bf16_to_fp32(hy[5]);
        const bool pass = std::fabs(y00 - 256.0f) < 24.0f &&
                          std::fabs(y05 - 128.0f) < 12.0f;
        std::printf("%-44s %s  y[*,0]=%.2f (exp 256) y[*,5]=%.2f (exp 128)\n",
            "Linear row-varying W (W[0]=2)",
            pass ? "PASS" : "FAIL", y00, y05);
        ok &= pass;
    }

    // Case 3: K-axis-varying W: W[*, 16..31]=2, else 1, x=1
    //   y[m, n] = 16*1 + 16*2 + 16*1 + 16*1 + ... = K + 16 (extra 16 from k 16..31)
    {
        const int M = 128, N = 128, K = 128;
        std::vector<__nv_bfloat16> hx(M * K, fp32_to_bf16(1.0f));
        std::vector<__nv_bfloat16> hW(N * K);
        for (int n = 0; n < N; ++n)
            for (int k = 0; k < K; ++k)
                hW[n*K + k] = fp32_to_bf16((k >= 16 && k < 32) ? 2.0f : 1.0f);
        const float expect = static_cast<float>(K) + 16.0f;  // = 144 for K=128
        ok &= run_one("Linear K-axis-varying W (W[k 16..31]=2)",
                      M, N, K, hx, hW, expect, /*tol*/16.0f);
    }

    // ---- RANDOM-DATA tests (the original failing ones) ----
    auto run_random = [](int M, int N, int K, bool with_bias) {
        std::mt19937 rng(0xfeedfaceULL);
        std::uniform_real_distribution<float> dx(-1.0f, 1.0f);
        std::vector<__nv_bfloat16> hx(static_cast<size_t>(M) * K),
                                   hW(static_cast<size_t>(N) * K),
                                   hbias(N);
        for (auto& v : hx)    v = fp32_to_bf16(dx(rng));
        for (auto& v : hW)    v = fp32_to_bf16(dx(rng));
        for (auto& v : hbias) v = fp32_to_bf16(dx(rng) * 0.5f);

        f2k::cuda::Linear::Config cfg;
        cfg.batch_rows = M; cfg.in_features = K; cfg.out_features = N;
        cfg.W_bf16 = hW.data();
        cfg.bias_bf16 = with_bias ? hbias.data() : nullptr;
        f2k::cuda::Linear lin(cfg);
        if (!lin.ok()) { std::printf("FAIL ctor: %s\n", lin.last_error()); return false; }

        void *d_x, *d_y, *d_ws;
        cudaMalloc(&d_x,  hx.size() * 2);
        cudaMalloc(&d_y,  static_cast<size_t>(M) * N * 2);
        cudaMalloc(&d_ws, lin.workspace_size_bytes());
        cudaMemcpy(d_x, hx.data(), hx.size() * 2, cudaMemcpyHostToDevice);
        lin.forward(d_x, d_y, d_ws, lin.workspace_size_bytes());
        cudaDeviceSynchronize();
        std::vector<__nv_bfloat16> hy(static_cast<size_t>(M) * N);
        cudaMemcpy(hy.data(), d_y, hy.size() * 2, cudaMemcpyDeviceToHost);
        cudaFree(d_x); cudaFree(d_y); cudaFree(d_ws);

        // FP32 reference y_ref[m, n] = sum_k x[m,k] * W[n,k] + bias[n].
        std::vector<float> y_ref(static_cast<size_t>(M) * N, 0.0f);
        for (int m = 0; m < M; ++m)
            for (int n = 0; n < N; ++n) {
                double acc = 0;
                for (int k = 0; k < K; ++k)
                    acc += bf16_to_fp32(hx[m*K+k]) * bf16_to_fp32(hW[n*K+k]);
                if (with_bias) acc += bf16_to_fp32(hbias[n]);
                y_ref[m*N+n] = acc;
            }

        double dot=0, na=0, nb=0, sse=0;
        for (size_t i = 0; i < hy.size(); ++i) {
            const double a = bf16_to_fp32(hy[i]);
            const double b = y_ref[i];
            dot += a*b; na += a*a; nb += b*b;
            sse += (a-b)*(a-b);
        }
        const double cos = dot / std::sqrt(na*nb + 1e-30);
        const double rel = std::sqrt(sse / (nb + 1e-30));
        std::printf("Linear random M=%-4d N=%-4d K=%-4d bias=%s  cos=%.4f rel_l2=%.4f\n",
                    M, N, K, with_bias?"yes":"no ", cos, rel);
        return cos > 0.95 && rel < 0.30;
    };

    ok &= run_random(128,  128,  64,   false);
    ok &= run_random(128,  256,  128,  true);
    ok &= run_random(256,  512,  256,  false);
    ok &= run_random(256,  256,  512,  true);

    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
