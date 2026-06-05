// Linear (BF16 → NVFP4-quantize → FP4 GEMM → BF16 + bias).
//
// The CUTLASS type setup mirrors fp4_gemm.cu exactly — re-declared here so
// the activation/weight quantization kernels can write the SFA/SFB scale
// tables in the same CuTe layouts the CUTLASS kernel reads from.

#include "backend/cuda/linear.h"
#include "backend/cuda/fp4_gemm.h"
#include "backend/cuda/fp8_gemm.h"

#include "cutlass/cutlass.h"
#include "cute/tensor.hpp"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"
#include "cutlass/util/packed_stride.hpp"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstring>
#include <string>
#include <vector>

#if defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED)

using namespace cute;

// Named (not anonymous) namespace: nvcc's cudafe stub generator gives the
// same mangled internal name to file-scope anonymous namespaces in some TUs,
// which collides with cute's own anonymous namespaces. A unique name avoids it.
namespace f2k_linear_internal {

// ---- CUTLASS type config (mirror of fp4_gemm.cu) ---------------------------

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
// ColumnMajor (CUTLASS's only supported layout for NVFP4 B on Blackwell).
// Bytes are laid out k*(N/2) + (n/2): two consecutive N values share a byte.
// Our user-facing W input is [N, K] row-major; quant_for_B_kernel handles
// the transposed packing.
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 32;

using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using ElementD   = cutlass::bfloat16_t;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

// ---- Quantization kernels --------------------------------------------------
//
// One CTA == one 16-element microblock of one row. 16 threads per CTA, all in
// the lower half of a warp (mask 0xFFFF). Active threads compute absmax,
// derive a per-block E4M3 scale (= absmax / E2M1_MAX), quantize their element
// to E2M1, and pack into 8 bytes. Thread 0 emits the scale via the CuTe layout.

constexpr float E2M1_MAX = 6.0f;
constexpr unsigned WARP16_MASK = 0x0000FFFFu;

__device__ __forceinline__ float subwarp_absmax_16(float v) {
    v = fmaxf(v, __shfl_xor_sync(WARP16_MASK, v, 1));
    v = fmaxf(v, __shfl_xor_sync(WARP16_MASK, v, 2));
    v = fmaxf(v, __shfl_xor_sync(WARP16_MASK, v, 4));
    v = fmaxf(v, __shfl_xor_sync(WARP16_MASK, v, 8));
    return v;
}

template <class LayoutSF>
__global__ void quant_for_A_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ packed,
    cutlass::float_ue4m3_t* __restrict__ sf,
    LayoutSF layout_sf,
    int rows, int K)
{
    const int kblock = blockIdx.x;
    const int row    = blockIdx.y;
    const int tid    = threadIdx.x;
    if (row >= rows) return;

    const int k_idx = kblock * 16 + tid;
    if (k_idx >= K) return;

    const float v = __bfloat162float(x[static_cast<size_t>(row) * K + k_idx]);
    const float absmax = subwarp_absmax_16(fabsf(v));
    const float scale  = (absmax > 1e-30f) ? (absmax / E2M1_MAX) : 1.0f;

    // Quantize element to E2M1; mask to lower 4 bits.
    const cutlass::float_e2m1_t q(v / scale);
    const uint8_t nibble = static_cast<uint8_t>(q.raw()) & 0xFu;

    // Pair adjacent lanes: even lane writes the byte (low|high).
    const uint8_t other_nibble = static_cast<uint8_t>(
        __shfl_xor_sync(WARP16_MASK, static_cast<unsigned>(nibble), 1) & 0xFu);
    if ((tid & 1) == 0) {
        const uint8_t byte = nibble | static_cast<uint8_t>(other_nibble << 4);
        const size_t byte_idx =
            static_cast<size_t>(row) * (K / 2) + static_cast<size_t>(kblock) * 8 + (tid >> 1);
        packed[byte_idx] = byte;
    }

    // One scale per microblock — written via the CuTe SFA layout.
    // Per gett.hpp the K coord is the *element* index that starts the
    // microblock (0, 16, 32, ...), not the microblock index itself.
    if (tid == 0) {
        const auto idx = layout_sf(row, kblock * 16, 0);
        sf[idx] = cutlass::float_ue4m3_t(scale);
    }
}

// quant_for_B_kernel removed (2026-06-03): used naive byte_idx for ColumnMajor B
// which doesn't match CUTLASS's tiled FP4 layout. Weight quant stays on the
// host (runs once at load time) and uses CuTe tensors. If/when GPU weight
// quant is needed (e.g. for online LoRA application), it must use CuTe
// tensors on device — see feedback-cutlass-nvfp4-linear-gap memory.

// ---- Bias add (BF16, broadcasts [N] across rows) ---------------------------

__global__ void bias_add_bf16_kernel(__nv_bfloat16* __restrict__ y,
                                     const __nv_bfloat16* __restrict__ bias,
                                     int M, int N) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    const int m = blockIdx.y;
    if (n >= N || m >= M) return;
    const size_t off = static_cast<size_t>(m) * N + n;
    const float yv = __bfloat162float(y[off]);
    const float bv = __bfloat162float(bias[n]);
    y[off] = __float2bfloat16(yv + bv);
}

// ---- Misc ------------------------------------------------------------------

inline size_t align_up(size_t x, size_t a) { return (x + a - 1) / a * a; }

} // namespace f2k_linear_internal

// ============================================================================
// MXFP8 path: E4M3 data + per-32 UE8M0 block scales. Mirrors fp8_gemm.cu's
// type config so the activation/weight quant can write SFA/SFB in the layouts
// the CUTLASS kernel reads. Separate named namespace to keep the two
// block-scaled type configs from colliding.
// ============================================================================
namespace f2k_linear_fp8 {

using ElementA   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 16;

using ElementB   = cutlass::mx_float8_t<cutlass::float_e4m3_t>;
using LayoutBTag = cutlass::layout::ColumnMajor;
constexpr int AlignmentB = 16;

using ElementC   = cutlass::bfloat16_t;
using LayoutCTag = cutlass::layout::RowMajor;
using ElementD   = cutlass::bfloat16_t;
using LayoutDTag = cutlass::layout::RowMajor;
constexpr int AlignmentC = 128 / cutlass::sizeof_bits<ElementC>::value;
constexpr int AlignmentD = 128 / cutlass::sizeof_bits<ElementD>::value;

using ElementAccumulator = float;
using ArchTag            = cutlass::arch::Sm120;
using OperatorClass      = cutlass::arch::OpClassBlockScaledTensorOp;

using ThreadBlockShape = Shape<_128, _128, _128>;
using ClusterShape     = Shape<_1, _1, _1>;

using CollectiveEpilogue = typename cutlass::epilogue::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ThreadBlockShape, ClusterShape,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementAccumulator,
    ElementC, LayoutCTag, AlignmentC,
    ElementD, LayoutDTag, AlignmentD,
    cutlass::epilogue::collective::EpilogueScheduleAuto
>::CollectiveOp;

using CollectiveMainloop = typename cutlass::gemm::collective::CollectiveBuilder<
    ArchTag, OperatorClass,
    ElementA, LayoutATag, AlignmentA,
    ElementB, LayoutBTag, AlignmentB,
    ElementAccumulator,
    ThreadBlockShape, ClusterShape,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue::SharedStorage))>,
    cutlass::gemm::collective::KernelScheduleAuto
>::CollectiveOp;

using GemmKernel = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop,
    CollectiveEpilogue,
    void>;

using Gemm = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel>;

using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

constexpr float E4M3_MAX = 448.0f;
constexpr int   MX_VEC   = 32;

// Largest power-of-two scale X = 2^ceil(log2(absmax/448)) s.t. absmax/X <= 448,
// maximizing E4M3 resolution within the block.
__device__ __host__ __forceinline__ float mx_scale_from_absmax(float absmax) {
    if (!(absmax > 0.0f)) return 1.0f;
    const int e = static_cast<int>(ceilf(log2f(absmax / E4M3_MAX)));
    return ldexpf(1.0f, e);
}

// One CTA == one 32-element microblock of one row. 32 threads (full warp).
// Each lane quantizes its element to E4M3 (row-major, 1 byte each); lane 0
// emits the UE8M0 block scale via the CuTe SFA layout.
template <class LayoutSF>
__global__ void quant_for_A_fp8_kernel(
    const __nv_bfloat16* __restrict__ x,
    uint8_t* __restrict__ data,
    cutlass::float_ue8m0_t* __restrict__ sf,
    LayoutSF layout_sf,
    int rows, int K)
{
    const int kblock = blockIdx.x;
    const int row    = blockIdx.y;
    const int tid    = threadIdx.x;
    if (row >= rows) return;
    const int k_idx = kblock * MX_VEC + tid;
    if (k_idx >= K) return;

    const float v = __bfloat162float(x[static_cast<size_t>(row) * K + k_idx]);
    float a = fabsf(v);
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 1));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 2));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 4));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 8));
    a = fmaxf(a, __shfl_xor_sync(0xFFFFFFFFu, a, 16));

    const float X = mx_scale_from_absmax(a);
    const cutlass::float_e4m3_t q(v / X);
    data[static_cast<size_t>(row) * K + k_idx] = static_cast<uint8_t>(q.raw());

    if (tid == 0) {
        const auto idx = layout_sf(row, kblock * MX_VEC, 0);
        sf[idx] = cutlass::float_ue8m0_t(X);
    }
}

} // namespace f2k_linear_fp8

namespace f2k::cuda {

using namespace f2k_linear_internal;


// ----------------------------------------------------------------------------
// Linear::Impl
// ----------------------------------------------------------------------------

struct Linear::Impl {
    int M = 0, N = 0, K = 0;
    bool        valid = false;
    std::string err;
    Linear::Precision prec = Linear::Precision::NVFP4;

    // Device-resident quantized weight + optional bias. For NVFP4 the data is
    // packed FP4 ([N,K/2]); for MXFP8 it is E4M3 ([N,K], 1 byte/elem).
    void* W_data_dev = nullptr;
    void* W_sfb_dev  = nullptr;   // SFB-laid-out block scales (E4M3 or UE8M0)
    void* bias_dev   = nullptr;   // [N] BF16 or null

    size_t W_data_bytes = 0;
    size_t W_sfb_bytes  = 0;
    size_t bias_bytes   = 0;

    // Cached layout instances (only the active precision's are populated).
    LayoutSFA layout_sfa{};
    LayoutSFB layout_sfb{};
    f2k_linear_fp8::LayoutSFA layout_sfa8{};
    f2k_linear_fp8::LayoutSFB layout_sfb8{};

    // Workspace partitioning (offsets into the caller-provided workspace).
    size_t off_A_data = 0;        // quantized activation (FP4 packed or E4M3)
    size_t off_SFA   = 0;
    size_t off_gemm  = 0;
    size_t sfa_bytes = 0;
    size_t total_workspace = 0;

    // Exactly one GEMM is constructed, per precision.
    std::unique_ptr<FP4Gemm> gemm;
    std::unique_ptr<FP8Gemm> gemm8;
};

// ----------------------------------------------------------------------------
// Constructor: validate, allocate, quantize W (and copy bias if any).
// ----------------------------------------------------------------------------

Linear::Linear(const Config& cfg) : impl_(std::make_unique<Impl>()) {
    Impl& I = *impl_;
    I.M = cfg.batch_rows;
    I.N = cfg.out_features;
    I.K = cfg.in_features;
    I.prec = cfg.precision;
    const bool fp8 = (I.prec == Linear::Precision::MXFP8);

    const int k_align = fp8 ? 128 : 64;   // MXFP8 vec=32, tile K=128
    if (I.M % 128 != 0 || I.N % 128 != 0 || I.K % k_align != 0) {
        I.err = "shape constraints: M,N % 128; K % (64 NVFP4 / 128 MXFP8)";
        return;
    }
    const bool have_bf16 = (cfg.W_bf16 != nullptr);
    const bool have_preq = (cfg.W_preq != nullptr);
    if (have_bf16 == have_preq) {
        I.err = "Config: exactly one of W_bf16 / W_preq must be set";
        return;
    }
    if (have_preq) {
        if (cfg.W_preq->N != I.N || cfg.W_preq->K != I.K) {
            I.err = "W_preq shape mismatch with (N, K)";
            return;
        }
        const int want_mb = fp8 ? 32 : 16;
        if (cfg.W_preq->microblock_size != want_mb) {
            I.err = "W_preq microblock_size mismatch (want 16 NVFP4 / 32 MXFP8)";
            return;
        }
        if (!cfg.W_preq->packed || !cfg.W_preq->scales) {
            I.err = "W_preq packed/scales pointer is null";
            return;
        }
    }

    size_t A_data_bytes = 0, gemm_ws = 0;

    if (!fp8) {
        // ===================== NVFP4 path =====================
        I.gemm = std::make_unique<FP4Gemm>(I.M, I.N, I.K);
        if (!I.gemm->ok()) {
            I.err = std::string("FP4Gemm init: ") + I.gemm->last_error();
            return;
        }
        I.layout_sfa =
            Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(I.M, I.N, I.K, 1));
        I.layout_sfb =
            Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(I.M, I.N, I.K, 1));

        I.W_data_bytes = static_cast<size_t>(I.N) * I.K / 2;     // packed FP4
        I.W_sfb_bytes  = static_cast<size_t>(size(filter_zeros(I.layout_sfb)));
        if (cudaMalloc(&I.W_data_dev, I.W_data_bytes) != cudaSuccess) { I.err = "cudaMalloc(W_fp4)"; return; }
        if (cudaMalloc(&I.W_sfb_dev,  I.W_sfb_bytes)  != cudaSuccess) { I.err = "cudaMalloc(W_sfb)"; return; }

        std::vector<uint8_t> W_packed(I.W_data_bytes, 0);
        std::vector<cutlass::float_ue4m3_t> W_sf(I.W_sfb_bytes);
        auto sf_tensor = cute::make_tensor(W_sf.data(), I.layout_sfb);
        auto stride_B = cutlass::make_cute_packed_stride(
            typename Gemm::GemmKernel::StrideB{}, {I.N, I.K, 1});
        auto layout_B = cute::make_layout(cute::make_shape(I.N, I.K, 1), stride_B);
        auto w_tensor = cute::make_tensor(
            cute::recast_ptr<cutlass::float_e2m1_t>(W_packed.data()), layout_B);
        const int n_kblocks = I.K / 16;

        if (have_bf16) {
            const __nv_bfloat16* W_h = static_cast<const __nv_bfloat16*>(cfg.W_bf16);
            for (int n = 0; n < I.N; ++n)
                for (int kb = 0; kb < n_kblocks; ++kb) {
                    float absmax = 0.0f;
                    for (int i = 0; i < 16; ++i)
                        absmax = std::max(absmax,
                            std::fabs(__bfloat162float(W_h[static_cast<size_t>(n) * I.K + kb * 16 + i])));
                    const float scale = (absmax > 1e-30f) ? (absmax / 6.0f) : 1.0f;
                    sf_tensor(n, kb * 16, 0) = cutlass::float_ue4m3_t(scale);
                    for (int i = 0; i < 16; ++i) {
                        const int k = kb * 16 + i;
                        const float v = __bfloat162float(W_h[static_cast<size_t>(n) * I.K + k]);
                        w_tensor(n, k, 0) = cutlass::float_e2m1_t(v / scale);
                    }
                }
        } else {
            const uint8_t* src_packed = static_cast<const uint8_t*>(cfg.W_preq->packed);
            const uint8_t* src_scales = static_cast<const uint8_t*>(cfg.W_preq->scales);
            for (int n = 0; n < I.N; ++n)
                for (int kb = 0; kb < n_kblocks; ++kb) {
                    cutlass::float_ue4m3_t s;
                    s.raw() = src_scales[static_cast<size_t>(n) * n_kblocks + kb];
                    sf_tensor(n, kb * 16, 0) = s;
                    for (int i = 0; i < 16; ++i) {
                        const int k = kb * 16 + i;
                        const size_t byte_idx = static_cast<size_t>(n) * (I.K / 2) + (k >> 1);
                        const uint8_t byte = src_packed[byte_idx];
                        const uint8_t nibble = (k & 1) ? (byte >> 4) : (byte & 0xFu);
                        cutlass::float_e2m1_t e2m1; e2m1.raw() = nibble;
                        w_tensor(n, k, 0) = e2m1;
                    }
                }
        }
        cudaMemcpy(I.W_data_dev, W_packed.data(), I.W_data_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(I.W_sfb_dev,  W_sf.data(),
                   I.W_sfb_bytes * sizeof(cutlass::float_ue4m3_t), cudaMemcpyHostToDevice);

        A_data_bytes = static_cast<size_t>(I.M) * I.K / 2;
        I.sfa_bytes  = static_cast<size_t>(size(filter_zeros(I.layout_sfa)));
        gemm_ws      = I.gemm->workspace_size_bytes();
    } else {
        // ===================== MXFP8 path =====================
        namespace fp8ns = f2k_linear_fp8;
        I.gemm8 = std::make_unique<FP8Gemm>(I.M, I.N, I.K);
        if (!I.gemm8->ok()) {
            I.err = std::string("FP8Gemm init: ") + I.gemm8->last_error();
            return;
        }
        I.layout_sfa8 =
            fp8ns::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(I.M, I.N, I.K, 1));
        I.layout_sfb8 =
            fp8ns::Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(I.M, I.N, I.K, 1));

        I.W_data_bytes = static_cast<size_t>(I.N) * I.K;          // E4M3, 1 byte/elem
        I.W_sfb_bytes  = static_cast<size_t>(size(filter_zeros(I.layout_sfb8)));
        if (cudaMalloc(&I.W_data_dev, I.W_data_bytes) != cudaSuccess) { I.err = "cudaMalloc(W_e4m3)"; return; }
        if (cudaMalloc(&I.W_sfb_dev,  I.W_sfb_bytes)  != cudaSuccess) { I.err = "cudaMalloc(W_sfb8)"; return; }

        std::vector<uint8_t> W_packed(I.W_data_bytes, 0);
        std::vector<cutlass::float_ue8m0_t> W_sf(I.W_sfb_bytes);
        auto sf_tensor = cute::make_tensor(W_sf.data(), I.layout_sfb8);
        auto stride_B = cutlass::make_cute_packed_stride(
            typename fp8ns::Gemm::GemmKernel::StrideB{}, {I.N, I.K, 1});
        auto layout_B = cute::make_layout(cute::make_shape(I.N, I.K, 1), stride_B);
        auto w_tensor = cute::make_tensor(
            cute::recast_ptr<cutlass::float_e4m3_t>(W_packed.data()), layout_B);
        const int n_kblocks = I.K / fp8ns::MX_VEC;

        if (have_bf16) {
            // Quantize from a BF16 source at construction.
            const __nv_bfloat16* W_h = static_cast<const __nv_bfloat16*>(cfg.W_bf16);
            for (int n = 0; n < I.N; ++n)
                for (int kb = 0; kb < n_kblocks; ++kb) {
                    float absmax = 0.0f;
                    for (int i = 0; i < fp8ns::MX_VEC; ++i)
                        absmax = std::max(absmax,
                            std::fabs(__bfloat162float(W_h[static_cast<size_t>(n) * I.K + kb * fp8ns::MX_VEC + i])));
                    const float X = fp8ns::mx_scale_from_absmax(absmax);
                    sf_tensor(n, kb * fp8ns::MX_VEC, 0) = cutlass::float_ue8m0_t(X);
                    for (int i = 0; i < fp8ns::MX_VEC; ++i) {
                        const int k = kb * fp8ns::MX_VEC + i;
                        const float v = __bfloat162float(W_h[static_cast<size_t>(n) * I.K + k]);
                        w_tensor(n, k, 0) = cutlass::float_e4m3_t(v / X);
                    }
                }
        } else {
            // Copy pre-quantized MXFP8 bits directly from F2K: E4M3 data [N,K]
            // row-major + UE8M0 scales [N,K/32] row-major.
            const uint8_t* src_packed = static_cast<const uint8_t*>(cfg.W_preq->packed);
            const uint8_t* src_scales = static_cast<const uint8_t*>(cfg.W_preq->scales);
            for (int n = 0; n < I.N; ++n)
                for (int kb = 0; kb < n_kblocks; ++kb) {
                    cutlass::float_ue8m0_t s;
                    s.raw() = src_scales[static_cast<size_t>(n) * n_kblocks + kb];
                    sf_tensor(n, kb * fp8ns::MX_VEC, 0) = s;
                    for (int i = 0; i < fp8ns::MX_VEC; ++i) {
                        const int k = kb * fp8ns::MX_VEC + i;
                        cutlass::float_e4m3_t e;
                        e.raw() = src_packed[static_cast<size_t>(n) * I.K + k];
                        w_tensor(n, k, 0) = e;
                    }
                }
        }
        cudaMemcpy(I.W_data_dev, W_packed.data(), I.W_data_bytes, cudaMemcpyHostToDevice);
        cudaMemcpy(I.W_sfb_dev,  W_sf.data(),
                   I.W_sfb_bytes * sizeof(cutlass::float_ue8m0_t), cudaMemcpyHostToDevice);

        A_data_bytes = static_cast<size_t>(I.M) * I.K;            // E4M3 activation
        I.sfa_bytes  = static_cast<size_t>(size(filter_zeros(I.layout_sfa8)));
        gemm_ws      = I.gemm8->workspace_size_bytes();
    }

    if (cfg.bias_bf16) {
        I.bias_bytes = static_cast<size_t>(I.N) * sizeof(__nv_bfloat16);
        if (cudaMalloc(&I.bias_dev, I.bias_bytes) != cudaSuccess) {
            I.err = "cudaMalloc(bias)";
            return;
        }
        cudaMemcpy(I.bias_dev, cfg.bias_bf16, I.bias_bytes, cudaMemcpyHostToDevice);
    }

    // Workspace plan: A_data | SFA | GEMM workspace, each 256-byte aligned.
    I.off_A_data = 0;
    I.off_SFA    = align_up(I.off_A_data + A_data_bytes, 256);
    I.off_gemm   = align_up(I.off_SFA    + I.sfa_bytes,  256);
    I.total_workspace = I.off_gemm + gemm_ws;

    I.valid = true;
}

Linear::~Linear() {
    if (impl_) {
        if (impl_->W_data_dev) cudaFree(impl_->W_data_dev);
        if (impl_->W_sfb_dev)  cudaFree(impl_->W_sfb_dev);
        if (impl_->bias_dev)   cudaFree(impl_->bias_dev);
    }
}

Linear::Linear(Linear&&) noexcept = default;
Linear& Linear::operator=(Linear&&) noexcept = default;

bool        Linear::ok()         const { return impl_ && impl_->valid; }
const char* Linear::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
size_t Linear::workspace_size_bytes() const { return impl_ ? impl_->total_workspace : 0; }

bool Linear::forward(const void* x_bf16, void* y_bf16,
                     void* workspace, size_t workspace_size,
                     cudaStream_t stream) {
    Impl& I = *impl_;
    if (!I.valid) return false;
    if (workspace_size < I.total_workspace) {
        I.err = "workspace too small";
        return false;
    }
    uint8_t* ws = static_cast<uint8_t*>(workspace);
    void* A_data = ws + I.off_A_data;
    void* SFA    = ws + I.off_SFA;
    void* gws    = ws + I.off_gemm;

    if (I.prec == Linear::Precision::NVFP4) {
        // 1. Quantize x → (A_fp4, SFA) on device.
        const int n_kblocks = I.K / 16;
        dim3 grid(n_kblocks, I.M), block(16);
        quant_for_A_kernel<LayoutSFA><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_bf16),
            static_cast<uint8_t*>(A_data),
            static_cast<cutlass::float_ue4m3_t*>(SFA),
            I.layout_sfa, I.M, I.K);
        if (cudaPeekAtLastError() != cudaSuccess) {
            I.err = "activation quant launch failed"; return false;
        }
        // 2. GEMM. C = nullptr (beta = 0).
        if (!I.gemm->run(A_data, SFA, I.W_data_dev, I.W_sfb_dev,
                         nullptr, y_bf16, gws, 1.0f, 0.0f, stream)) {
            I.err = std::string("gemm.run: ") + I.gemm->last_error(); return false;
        }
    } else {
        // 1. Quantize x → (A_e4m3, SFA) on device (MXFP8, per-32 UE8M0).
        namespace fp8ns = f2k_linear_fp8;
        const int n_kblocks = I.K / fp8ns::MX_VEC;
        dim3 grid(n_kblocks, I.M), block(fp8ns::MX_VEC);
        fp8ns::quant_for_A_fp8_kernel<fp8ns::LayoutSFA><<<grid, block, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(x_bf16),
            static_cast<uint8_t*>(A_data),
            static_cast<cutlass::float_ue8m0_t*>(SFA),
            I.layout_sfa8, I.M, I.K);
        if (cudaPeekAtLastError() != cudaSuccess) {
            I.err = "fp8 activation quant launch failed"; return false;
        }
        // 2. GEMM.
        if (!I.gemm8->run(A_data, SFA, I.W_data_dev, I.W_sfb_dev,
                          nullptr, y_bf16, gws, 1.0f, 0.0f, stream)) {
            I.err = std::string("fp8 gemm.run: ") + I.gemm8->last_error(); return false;
        }
    }

    // 3. Optional bias.
    if (I.bias_dev) {
        constexpr int BLK = 256;
        dim3 bgrid((I.N + BLK - 1) / BLK, I.M), bblock(BLK);
        bias_add_bf16_kernel<<<bgrid, bblock, 0, stream>>>(
            static_cast<__nv_bfloat16*>(y_bf16),
            static_cast<const __nv_bfloat16*>(I.bias_dev),
            I.M, I.N);
        if (cudaPeekAtLastError() != cudaSuccess) {
            I.err = "bias add launch failed";
            return false;
        }
    }
    return true;
}

} // namespace f2k::cuda

#else  // ---------------------------- arch unsupported ---------------------

namespace f2k::cuda {
struct Linear::Impl { const char* err = "CUTLASS built without SM120/SM121 NVFP4 support"; };
Linear::Linear(const Config&) : impl_(std::make_unique<Impl>()) {}
Linear::~Linear() = default;
Linear::Linear(Linear&&) noexcept = default;
Linear& Linear::operator=(Linear&&) noexcept = default;
bool        Linear::ok()         const { return false; }
const char* Linear::last_error() const { return impl_ ? impl_->err : "no impl"; }
size_t Linear::workspace_size_bytes() const { return 0; }
bool Linear::forward(const void*, void*, void*, size_t, cudaStream_t) { return false; }
} // namespace f2k::cuda

#endif
