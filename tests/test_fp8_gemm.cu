// Exercises the Blackwell MXFP8 GEMM via the wrapper. Mirrors test_fp4_gemm:
// three problem sizes, verified against CUTLASS's host reference, prints TFLOPS.

#include "backend/cuda/fp8_gemm.h"

#include <cstdio>
#include <cstdlib>

namespace {

bool run_case(int m, int n, int k, int iters) {
    auto r = f2k::cuda::run_fp8_gemm_smoke(m, n, k, iters);
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
    all_ok &= run_case(256,  256,  256,  10);
    all_ok &= run_case(2048, 2048, 2048, 10);
    all_ok &= run_case(4096, 4096, 1024, 5);

    std::printf("%s\n", all_ok ? "ALL OK" : "FAILURES");
    return all_ok ? 0 : 1;
}
