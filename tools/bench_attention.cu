// Microbenchmark for the WMMA flash-attention kernel at transformer scale.
// Reports latency, achieved HBM bandwidth (K/V re-read traffic dominates), and
// effective TFLOPS, so we can tell whether the kernel is bandwidth-bound (→
// larger query tiles / more K/V reuse pays off) or utilization-bound.

#include "backend/cuda/attention.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdio>
#include <random>
#include <vector>

namespace f2k::cuda {
// Experimental register-O occupancy probe defined in attention.cu (D=128,
// aligned S only).
bool launch_wmma_regO_probe(const void* Q, const void* K, const void* V, void* O,
                            int B, int S, int H, float scale, bool report,
                            cudaStream_t stream);
bool launch_mma_probe(const void* Q, const void* K, const void* V, void* O,
                      int B, int S, int H, float scale, bool report,
                      cudaStream_t stream);
}

namespace {

using ProbeFn = bool (*)(const void*, const void*, const void*, void*,
                         int, int, int, float, bool, cudaStream_t);

// Validate a probe kernel against the production forward() and benchmark it.
void bench_probe(ProbeFn probe, const char* /*label*/, int B, int S, int H, int D, int iters) {
    if (D != 128 || S % 64 != 0 || S % 48 != 0) {
        std::printf("probe skipped (needs D=128, S%%64==0, S%%48==0)\n");
        return;
    }
    const float scale = 1.0f / std::sqrt((float)D);
    const size_t N = (size_t)B * S * H * D, bytes = N * sizeof(__nv_bfloat16);
    std::vector<__nv_bfloat16> h(N);
    std::mt19937 rng(0xBEEF1234u);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& v : h) v = __float2bfloat16(d(rng));

    void *dQ, *dK, *dV, *dO, *dRef;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes); cudaMalloc(&dV, bytes);
    cudaMalloc(&dO, bytes); cudaMalloc(&dRef, bytes);
    cudaMemcpy(dQ, h.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, h.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, h.data(), bytes, cudaMemcpyHostToDevice);

    // Reference = the production kernel (already validated cos=1.0 vs FP32).
    f2k::cuda::Attention::Config cfg{};
    cfg.batch = B; cfg.seq = S; cfg.n_heads = H; cfg.head_dim = D;
    f2k::cuda::Attention att(cfg);
    att.forward(dQ, dK, dV, dRef);
    probe(dQ, dK, dV, dO, B, S, H, scale, true, 0);
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> hO(N), hR(N);
    cudaMemcpy(hO.data(), dO, bytes, cudaMemcpyDeviceToHost);
    cudaMemcpy(hR.data(), dRef, bytes, cudaMemcpyDeviceToHost);
    double dot = 0, na = 0, nb = 0;
    for (size_t i = 0; i < N; ++i) {
        const double a = __bfloat162float(hO[i]), bb = __bfloat162float(hR[i]);
        dot += a * bb; na += a * a; nb += bb * bb;
    }
    const double cos = dot / std::sqrt(na * nb + 1e-30);

    for (int i = 0; i < 3; ++i) probe(dQ, dK, dV, dO, B, S, H, scale, false, 0);
    cudaDeviceSynchronize();
    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) probe(dQ, dK, dV, dO, B, S, H, scale, false, 0);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;

    const int BM = 64;
    const long ctas = (long)((S + BM - 1) / BM) * H * B;
    const double kv = (double)ctas * S * D * 2.0 * sizeof(__nv_bfloat16);
    std::printf("    S=%-5d  %8.3f ms   %6.1f GB/s   cos_vs_prod=%.5f\n",
                S, ms, (kv + 2.0 * bytes) / (ms * 1e-3) / 1e9, cos);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO); cudaFree(dRef);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

void bench(int B, int S, int H, int D, int iters) {
    f2k::cuda::Attention::Config cfg{};
    cfg.batch = B; cfg.seq = S; cfg.n_heads = H; cfg.head_dim = D;
    f2k::cuda::Attention att(cfg);
    if (!att.ok()) { std::printf("ctor fail: %s\n", att.last_error()); return; }

    const size_t N = (size_t)B * S * H * D;
    const size_t bytes = N * sizeof(__nv_bfloat16);
    std::vector<__nv_bfloat16> h(N);
    std::mt19937 rng(0xBEEF1234u);
    std::uniform_real_distribution<float> d(-0.5f, 0.5f);
    for (auto& v : h) v = __float2bfloat16(d(rng));

    void *dQ, *dK, *dV, *dO;
    cudaMalloc(&dQ, bytes); cudaMalloc(&dK, bytes);
    cudaMalloc(&dV, bytes); cudaMalloc(&dO, bytes);
    cudaMemcpy(dQ, h.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dK, h.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dV, h.data(), bytes, cudaMemcpyHostToDevice);

    for (int i = 0; i < 3; ++i) att.forward(dQ, dK, dV, dO);  // warmup
    cudaDeviceSynchronize();

    cudaEvent_t t0, t1; cudaEventCreate(&t0); cudaEventCreate(&t1);
    cudaEventRecord(t0);
    for (int i = 0; i < iters; ++i) att.forward(dQ, dK, dV, dO);
    cudaEventRecord(t1); cudaEventSynchronize(t1);
    float ms = 0; cudaEventElapsedTime(&ms, t0, t1); ms /= iters;

    // K/V HBM traffic: each CTA (BM=64 queries) re-reads all S keys+values.
    // grid.x = ceil(S/64) CTAs per (head,batch); K and V each S*D bf16.
    const int BM = 64;
    const long ctas = (long)((S + BM - 1) / BM) * H * B;
    const double kv_bytes = (double)ctas * S * D * 2.0 /*K,V*/ * sizeof(__nv_bfloat16);
    const double qo_bytes = 2.0 * bytes;  // Q read + O write (once each)
    const double gbps = (kv_bytes + qo_bytes) / (ms * 1e-3) / 1e9;
    // FLOPs: QK^T (2*S*S*D) + PV (2*S*S*D) per (head,batch).
    const double flop = 4.0 * (double)S * S * D * H * B;
    const double tflops = flop / (ms * 1e-3) / 1e12;

    std::printf("S=%-5d H=%-3d D=%-3d  %8.3f ms   %6.1f GB/s   %6.1f TFLOPS  "
                "(KV reread %.2f GB, %ld CTAs)\n",
                S, H, D, ms, gbps, tflops, kv_bytes / 1e9, ctas);

    cudaFree(dQ); cudaFree(dK); cudaFree(dV); cudaFree(dO);
    cudaEventDestroy(t0); cudaEventDestroy(t1);
}

} // namespace

int main() {
    int dev = 0; cudaDeviceProp p; cudaGetDeviceProperties(&p, dev);
    std::printf("Device: %s  sm_%d%d  smem/block(optin)=%d KiB  %d SMs  "
                "(GB10 LPDDR5X peak ~273 GB/s)\n\n",
                p.name, p.major, p.minor,
                (int)(p.sharedMemPerBlockOptin / 1024), p.multiProcessorCount);

    // Transformer joint attention at several resolutions (B=1, H=32, D=128):
    //   256px  S=768   512px S=1536   1024px S=4608.
    bench(1, 768,  32, 128, 50);
    bench(1, 1536, 32, 128, 50);
    bench(1, 4608, 32, 128, 30);
    // VAE mid-block (D=512) for reference — uses the warp-per-query flash path.
    bench(1, 16384, 1, 512, 10);

    std::printf("\n--- regO probe (register-resident O wmma) ---\n");
    bench_probe(f2k::cuda::launch_wmma_regO_probe, "regO", 1, 768,  32, 128, 50);
    bench_probe(f2k::cuda::launch_wmma_regO_probe, "regO", 1, 4608, 32, 128, 30);
    std::printf("\n--- mma.sync probe (rescale-tax-free) ---\n");
    bench_probe(f2k::cuda::launch_mma_probe, "mma", 1, 768,  32, 128, 50);
    bench_probe(f2k::cuda::launch_mma_probe, "mma", 1, 4608, 32, 128, 30);
    return 0;
}
