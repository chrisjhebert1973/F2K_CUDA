// Standalone validation of the m16n8k16 bf16 mma.sync primitives with MANUAL
// fragment loads (no ldmatrix). Proves the documented thread→element mapping
// before we build the flash-attention kernel on top of it.
//
//   build:  nvcc -arch=sm_121a -o /tmp/mma_unit tools/mma_unit.cu && /tmp/mma_unit
//
// mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32:
//   A: 16x16 row-major, regs a0..a3 (2 bf16 each)
//   B: 16x8  col-major, regs b0,b1
//   C/D: 16x8 fp32, regs d0..d3
//   gid = lane/4, tid4 = lane%4
//   a0=A[gid][2t+0..1]   a1=A[gid+8][2t..]   a2=A[gid][2t+8..]   a3=A[gid+8][2t+8..]
//   b0=B[2t..1][gid]     b1=B[2t+8..9][gid]   (k=row, n=col=gid)
//   d0=C[gid][2t]  d1=C[gid][2t+1]  d2=C[gid+8][2t]  d3=C[gid+8][2t+1]

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdio>
#include <cmath>
#include <random>
#include <vector>

__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    uint16_t l, h;
    __nv_bfloat16 a = lo, b = hi;
    l = *reinterpret_cast<uint16_t*>(&a);
    h = *reinterpret_cast<uint16_t*>(&b);
    return (uint32_t(h) << 16) | uint32_t(l);
}

__device__ __forceinline__ uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}
__device__ __forceinline__ void ldmatrix_x2(uint32_t (&r)[2], const void* p) {
    uint32_t a = smem_u32(p);
    asm volatile("ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}
__device__ __forceinline__ void ldmatrix_x2_trans(uint32_t (&r)[2], const void* p) {
    uint32_t a = smem_u32(p);
    asm volatile("ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0,%1}, [%2];"
                 : "=r"(r[0]), "=r"(r[1]) : "r"(a));
}

__device__ __forceinline__ void mma_m16n8k16(float (&d)[4],
                                             const uint32_t (&a)[4],
                                             const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3},{%4,%5,%6,%7},{%8,%9},{%0,%1,%2,%3};"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]),
          "r"(b[0]), "r"(b[1]));
}

// As: [16][16] row-major bf16. Bs: B[k][n], 16x8, stored row-major [16][8].
// Computes C[16][8] = As @ Bs, writes row-major to Cout (global).
__global__ void test_kernel(const __nv_bfloat16* As, const __nv_bfloat16* Bs,
                            float* Cout) {
    const int lane = threadIdx.x & 31;
    const int gid = lane >> 2, t = lane & 3;

    uint32_t a[4], b[2];
    a[0] = pack2(As[gid * 16 + 2 * t + 0],       As[gid * 16 + 2 * t + 1]);
    a[1] = pack2(As[(gid + 8) * 16 + 2 * t + 0], As[(gid + 8) * 16 + 2 * t + 1]);
    a[2] = pack2(As[gid * 16 + 2 * t + 8 + 0],       As[gid * 16 + 2 * t + 8 + 1]);
    a[3] = pack2(As[(gid + 8) * 16 + 2 * t + 8 + 0], As[(gid + 8) * 16 + 2 * t + 8 + 1]);
    b[0] = pack2(Bs[(2 * t + 0) * 8 + gid], Bs[(2 * t + 1) * 8 + gid]);
    b[1] = pack2(Bs[(2 * t + 8) * 8 + gid], Bs[(2 * t + 9) * 8 + gid]);

    float d[4] = {0, 0, 0, 0};
    mma_m16n8k16(d, a, b);

    Cout[gid * 8 + 2 * t + 0]       = d[0];
    Cout[gid * 8 + 2 * t + 1]       = d[1];
    Cout[(gid + 8) * 8 + 2 * t + 0] = d[2];
    Cout[(gid + 8) * 8 + 2 * t + 1] = d[3];
}

__device__ __forceinline__ void load_A(uint32_t (&a)[4], const __nv_bfloat16* As, int ld) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, t = lane & 3;
    a[0] = pack2(As[gid*ld + 2*t],       As[gid*ld + 2*t + 1]);
    a[1] = pack2(As[(gid+8)*ld + 2*t],   As[(gid+8)*ld + 2*t + 1]);
    a[2] = pack2(As[gid*ld + 2*t + 8],   As[gid*ld + 2*t + 9]);
    a[3] = pack2(As[(gid+8)*ld + 2*t + 8], As[(gid+8)*ld + 2*t + 9]);
}
__device__ __forceinline__ void store_C(float* C, const float (&d)[4], int ld) {
    const int lane = threadIdx.x & 31, gid = lane >> 2, t = lane & 3;
    C[gid*ld + 2*t] = d[0]; C[gid*ld + 2*t + 1] = d[1];
    C[(gid+8)*ld + 2*t] = d[2]; C[(gid+8)*ld + 2*t + 1] = d[3];
}

// Load A manually (validated), load B 16x8 via ldmatrix per (sel, trans), mma, store C.
// src is 128 bf16 in smem with leading dim ldS. sel/trans sweep the address recipe.
__global__ void test_B(const __nv_bfloat16* As, const __nv_bfloat16* src, float* C,
                       int ldS, int sel, int trans) {
    __shared__ __nv_bfloat16 s[256];
    const int lane = threadIdx.x & 31, ll = lane & 15;
    for (int i = lane; i < 128; i += 32) s[i] = src[i];
    __syncwarp();
    uint32_t a[4]; load_A(a, As, 16);
    const int off = (sel == 0) ? (ll % 8) * ldS + (ll / 8) * 8     // matrices split along cols(+8)
                               : ll * ldS;                          // matrices split along rows(+8 rows)
    uint32_t b[2];
    if (trans) ldmatrix_x2_trans(b, &s[off]); else ldmatrix_x2(b, &s[off]);
    float d[4] = {0, 0, 0, 0};
    mma_m16n8k16(d, a, b);
    store_C(C, d, 8);
}

// Sweep recipes; report which (sel,trans) reproduces ref. mode 0=QK (B[k][n]=src[n][k],
// src=[8 key][16 hd] ldS=16), mode 1=PV (B[k][n]=src[k][n], src=[16 key][8 hd] ldS=8).
static int sweep_B(const char* name, int mode) {
    const int M = 16, K = 16, NN = 8;
    std::mt19937 rng(0xABCDEF12u + mode);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<__nv_bfloat16> hA(M * K), hSrc(128);
    for (auto& v : hA) v = __float2bfloat16(dist(rng));
    for (auto& v : hSrc) v = __float2bfloat16(dist(rng));
    auto Bval = [&](int k, int n) -> float {
        return (mode == 0) ? __bfloat162float(hSrc[n * 16 + k])    // src[key=n][hd=k]
                           : __bfloat162float(hSrc[k * 8 + n]);    // src[key=k][hd=n]
    };
    std::vector<float> ref(M * NN);
    for (int m = 0; m < M; ++m) for (int n = 0; n < NN; ++n) {
        double acc = 0; for (int k = 0; k < K; ++k) acc += __bfloat162float(hA[m*K+k]) * Bval(k, n);
        ref[m*NN+n] = acc;
    }
    __nv_bfloat16 *dA, *dS; float* dC;
    cudaMalloc(&dA, hA.size()*2); cudaMalloc(&dS, hSrc.size()*2); cudaMalloc(&dC, M*NN*4);
    cudaMemcpy(dA, hA.data(), hA.size()*2, cudaMemcpyHostToDevice);
    cudaMemcpy(dS, hSrc.data(), hSrc.size()*2, cudaMemcpyHostToDevice);
    const int ldS = (mode == 0) ? 16 : 8;
    int found = -1;
    for (int sel = 0; sel < 2; ++sel) for (int trans = 0; trans < 2; ++trans) {
        test_B<<<1, 32>>>(dA, dS, dC, ldS, sel, trans);
        cudaDeviceSynchronize();
        std::vector<float> hC(M*NN);
        cudaMemcpy(hC.data(), dC, M*NN*4, cudaMemcpyDeviceToHost);
        double me = 0; for (int i = 0; i < M*NN; ++i) me = fmax(me, fabs(hC[i]-ref[i]));
        std::printf("  %s sel=%d trans=%d  max_err=%.4f  %s\n", name, sel, trans, me,
                    me < 0.05 ? "<== MATCH" : "");
        if (me < 0.05 && found < 0) found = sel * 10 + trans;
    }
    cudaFree(dA); cudaFree(dS); cudaFree(dC);
    return found;
}

int main() {
    const int M = 16, N = 8, K = 16;
    std::vector<__nv_bfloat16> hA(M * K), hB(K * N);
    std::mt19937 rng(0xC0FFEE12u);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    for (auto& v : hA) v = __float2bfloat16(dist(rng));
    for (auto& v : hB) v = __float2bfloat16(dist(rng));

    __nv_bfloat16 *dA, *dB; float* dC;
    cudaMalloc(&dA, hA.size() * 2); cudaMalloc(&dB, hB.size() * 2);
    cudaMalloc(&dC, M * N * 4);
    cudaMemcpy(dA, hA.data(), hA.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dB, hB.data(), hB.size() * 2, cudaMemcpyHostToDevice);
    test_kernel<<<1, 32>>>(dA, dB, dC);
    cudaDeviceSynchronize();
    cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::printf("CUDA err: %s\n", cudaGetErrorString(e)); return 1; }

    std::vector<float> hC(M * N);
    cudaMemcpy(hC.data(), dC, M * N * 4, cudaMemcpyDeviceToHost);

    double max_e = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double ref = 0;
            for (int k = 0; k < K; ++k)
                ref += (double)__bfloat162float(hA[m * K + k]) *
                       (double)__bfloat162float(hB[k * N + n]);
            max_e = fmax(max_e, fabs(ref - hC[m * N + n]));
        }
    std::printf("m16n8k16 manual-load mma: max_abs_err=%.5f  %s\n",
                max_e, max_e < 0.05 ? "PASS" : "FAIL");

    std::printf("\nldmatrix B-load sweep (QK^T: col-major K source):\n");
    int qk = sweep_B("QK", 0);
    std::printf("ldmatrix B-load sweep (P@V: row-major V source):\n");
    int pv = sweep_B("PV", 1);
    std::printf("\nWinners: QK sel/trans=%d  PV sel/trans=%d\n", qk, pv);
    return (max_e < 0.05 && qk >= 0 && pv >= 0) ? 0 : 1;
}
