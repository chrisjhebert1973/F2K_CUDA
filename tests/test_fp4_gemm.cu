// Exercises the Blackwell NVFP4 GEMM via the wrapper. Runs three problem
// sizes — small, medium, and a transformer-scale shape — verifies each
// against CUTLASS's host reference, prints sustained TFLOPS.

#include "backend/cuda/fp4_gemm.h"

#include <cstdio>
#include <cstdlib>

namespace {

bool run_case(int m, int n, int k, int iters) {
    auto r = f2k::cuda::run_fp4_gemm_smoke(m, n, k, iters);
    std::printf("M=%-5d N=%-5d K=%-5d  ", m, n, k);
    if (!r.ok) {
        std::printf("FAIL (%s)\n", r.error_msg ? r.error_msg : "unknown");
        return false;
    }
    std::printf("verify=%s max_err=%.4f mean_err=%.4f  %.3f ms  %.1f TFLOPS\n",
                r.verify_passed ? "PASS" : "FAIL",
                r.max_abs_err, r.mean_abs_err,
                r.avg_runtime_ms, r.tflops);
    return r.verify_passed;
}

} // namespace

int main() {
    bool all_ok = true;
    // Small — quick sanity check, exercises the kernel launch path.
    all_ok &= run_case(256,  256,  256,  10);
    // Medium — representative of an attention QKV projection at moderate seq.
    all_ok &= run_case(2048, 2048, 2048, 10);
    // Transformer-scale — roughly an MMDiT FFN at hidden=4096, ffn=10752.
    all_ok &= run_case(4096, 4096, 1024, 5);

    std::printf("%s\n", all_ok ? "ALL OK" : "FAILURES");
    return all_ok ? 0 : 1;
}
