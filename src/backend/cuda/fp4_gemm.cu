// Blackwell NVFP4 → BF16 GEMM. PIMPL'd so the heavy CUTLASS templates stay
// out of the header.

#include "backend/cuda/fp4_gemm.h"

#include "cutlass/cutlass.h"

#include "cute/tensor.hpp"
#include "cutlass/tensor_ref.h"
#include "cutlass/epilogue/thread/linear_combination.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/detail/sm100_blockscaled_layout.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler_params.h"

#include "cutlass/util/distribution.h"
#include "cutlass/util/host_tensor.h"
#include "cutlass/util/packed_stride.hpp"
#include "cutlass/util/reference/host/gett.hpp"
#include "cutlass/util/reference/host/tensor_fill.h"
#include "cutlass/util/reference/host/tensor_norm.h"
#include "cutlass/util/reference/host/tensor_compare.h"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>

#if !(defined(CUTLASS_ARCH_MMA_SM120_SUPPORTED) || defined(CUTLASS_ARCH_MMA_SM121_SUPPORTED))
namespace f2k::cuda {

struct FP4Gemm::Impl { const char* err = "CUTLASS built without SM120/SM121 NVFP4 support"; };
FP4Gemm::FP4Gemm(int, int, int) : impl_(std::make_unique<Impl>()) {}
FP4Gemm::~FP4Gemm() = default;
FP4Gemm::FP4Gemm(FP4Gemm&&) noexcept = default;
FP4Gemm& FP4Gemm::operator=(FP4Gemm&&) noexcept = default;
bool FP4Gemm::ok() const { return false; }
const char* FP4Gemm::last_error() const { return impl_ ? impl_->err : "no impl"; }
int FP4Gemm::m() const { return 0; }
int FP4Gemm::n() const { return 0; }
int FP4Gemm::k() const { return 0; }
size_t FP4Gemm::a_size_bytes() const { return 0; }
size_t FP4Gemm::b_size_bytes() const { return 0; }
size_t FP4Gemm::c_size_bytes() const { return 0; }
size_t FP4Gemm::d_size_bytes() const { return 0; }
size_t FP4Gemm::sfa_size_bytes() const { return 0; }
size_t FP4Gemm::sfb_size_bytes() const { return 0; }
size_t FP4Gemm::workspace_size_bytes() const { return 0; }
bool FP4Gemm::run(const void*, const void*, const void*, const void*,
                  const void*, void*, void*, float, float, cudaStream_t) {
    return false;
}

FP4GemmResult run_fp4_gemm_smoke(int, int, int, int) {
    FP4GemmResult r;
    r.error_msg = "CUTLASS built without SM120/SM121 NVFP4 support";
    return r;
}

} // namespace f2k::cuda

#else  // ---------------------------- SM120/SM121 supported ------------------

using namespace cute;

namespace {

// --- Type configuration: matches example 79a exactly. ---------------------

using ElementA   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
using LayoutATag = cutlass::layout::RowMajor;
constexpr int AlignmentA = 32;

using ElementB   = cutlass::nv_float4_t<cutlass::float_e2m1_t>;
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

using StrideA   = typename Gemm::GemmKernel::StrideA;
using StrideB   = typename Gemm::GemmKernel::StrideB;
using StrideC   = typename Gemm::GemmKernel::StrideC;
using StrideD   = typename Gemm::GemmKernel::StrideD;
using LayoutSFA = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFA;
using LayoutSFB = typename Gemm::GemmKernel::CollectiveMainloop::LayoutSFB;
using Sm1xxBlkScaledConfig =
    typename Gemm::GemmKernel::CollectiveMainloop::Sm1xxBlkScaledConfig;

template <typename T>
auto make_iter(T* p) {
    return cute::recast_ptr<T>(p);
}

// CUTLASS reports byte sizes for FP4 differently across versions; compute
// from raw shape to be defensive.
inline size_t fp4_packed_bytes(int outer, int inner) {
    // 4-bit elements, two per byte, no padding.
    const size_t bits = static_cast<size_t>(outer) * static_cast<size_t>(inner) * 4;
    return (bits + 7) / 8;
}

} // anonymous namespace

namespace f2k::cuda {

// ----------------------------------------------------------------------------
// Impl
// ----------------------------------------------------------------------------

struct FP4Gemm::Impl {
    int m = 0, n = 0, k = 0;
    bool valid = false;
    std::string err;

    StrideA   stride_A{};
    StrideB   stride_B{};
    StrideC   stride_C{};
    StrideD   stride_D{};
    LayoutSFA layout_SFA{};
    LayoutSFB layout_SFB{};

    size_t sfa_bytes = 0;
    size_t sfb_bytes = 0;
    size_t workspace_bytes = 0;

    Gemm gemm;  // CUTLASS kernel handle. We re-initialize() per launch with
                // current pointers because device addresses may change.
};

// ----------------------------------------------------------------------------
// Class methods
// ----------------------------------------------------------------------------

FP4Gemm::FP4Gemm(int m, int n, int k) : impl_(std::make_unique<Impl>()) {
    if (m % 128 != 0 || n % 128 != 0 || k % 64 != 0) {
        impl_->err = "m,n must be multiples of 128; k a multiple of 64";
        return;
    }
    impl_->m = m;
    impl_->n = n;
    impl_->k = k;

    impl_->stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    impl_->stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    impl_->stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    impl_->stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    impl_->layout_SFA =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    impl_->layout_SFB =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    // Scale buffer sizes: number of E4M3 elements per the layout, times one byte each.
    impl_->sfa_bytes = static_cast<size_t>(size(filter_zeros(impl_->layout_SFA)));
    impl_->sfb_bytes = static_cast<size_t>(size(filter_zeros(impl_->layout_SFB)));

    // Workspace: query with placeholder args. The actual addresses don't matter
    // because workspace size depends only on problem shape, not data pointers.
    typename Gemm::Arguments probe{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {m, n, k, 1},
        { nullptr, impl_->stride_A,
          nullptr, impl_->stride_B,
          nullptr, impl_->layout_SFA,
          nullptr, impl_->layout_SFB },
        { {1.0f, 0.0f},
          nullptr, impl_->stride_C,
          nullptr, impl_->stride_D }
    };
    impl_->workspace_bytes = Gemm::get_workspace_size(probe);

    impl_->valid = true;
}

FP4Gemm::~FP4Gemm() = default;
FP4Gemm::FP4Gemm(FP4Gemm&&) noexcept = default;
FP4Gemm& FP4Gemm::operator=(FP4Gemm&&) noexcept = default;

bool        FP4Gemm::ok()         const { return impl_ && impl_->valid; }
const char* FP4Gemm::last_error() const {
    return impl_ ? (impl_->err.empty() ? "" : impl_->err.c_str()) : "no impl";
}
int FP4Gemm::m() const { return impl_->m; }
int FP4Gemm::n() const { return impl_->n; }
int FP4Gemm::k() const { return impl_->k; }

size_t FP4Gemm::a_size_bytes() const { return fp4_packed_bytes(impl_->m, impl_->k); }
size_t FP4Gemm::b_size_bytes() const { return fp4_packed_bytes(impl_->n, impl_->k); }
size_t FP4Gemm::c_size_bytes() const {
    return static_cast<size_t>(impl_->m) * impl_->n * sizeof(cutlass::bfloat16_t);
}
size_t FP4Gemm::d_size_bytes()         const { return c_size_bytes(); }
size_t FP4Gemm::sfa_size_bytes()       const { return impl_->sfa_bytes; }
size_t FP4Gemm::sfb_size_bytes()       const { return impl_->sfb_bytes; }
size_t FP4Gemm::workspace_size_bytes() const { return impl_->workspace_bytes; }

bool FP4Gemm::run(const void* A, const void* SFA,
                  const void* B, const void* SFB,
                  const void* C, void* D,
                  void* workspace,
                  float alpha, float beta,
                  cudaStream_t stream) {
    if (!impl_ || !impl_->valid) return false;

    typename Gemm::Arguments args{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {impl_->m, impl_->n, impl_->k, 1},
        {
            static_cast<const typename ElementA::DataType*>(A), impl_->stride_A,
            static_cast<const typename ElementB::DataType*>(B), impl_->stride_B,
            static_cast<const typename ElementA::ScaleFactorType*>(SFA), impl_->layout_SFA,
            static_cast<const typename ElementB::ScaleFactorType*>(SFB), impl_->layout_SFB
        },
        {
            {alpha, beta},
            static_cast<const ElementC*>(C), impl_->stride_C,
            static_cast<ElementD*>(D),       impl_->stride_D
        }
    };

    if (impl_->gemm.can_implement(args) != cutlass::Status::kSuccess) {
        impl_->err = "gemm.can_implement() rejected";
        return false;
    }
    if (impl_->gemm.initialize(args, workspace, stream) != cutlass::Status::kSuccess) {
        impl_->err = "gemm.initialize() failed";
        return false;
    }
    if (impl_->gemm.run(stream) != cutlass::Status::kSuccess) {
        impl_->err = "gemm.run() failed";
        return false;
    }
    return true;
}

// ----------------------------------------------------------------------------
// Smoke test built on top of FP4Gemm.
// ----------------------------------------------------------------------------

namespace {

template <typename Element, typename Layout>
void fill_random(cutlass::TensorView<Element, Layout> view, uint64_t seed) {
    double lo, hi;
    constexpr int bits = cutlass::sizeof_bits<Element>::value;
    if      constexpr (bits <= 6) { lo = -2; hi = 2; }
    else if constexpr (bits <= 8) {
        if constexpr (cute::is_same_v<Element, cutlass::float_ue8m0_t>) { lo = 1; hi = 4; }
        else                                                           { lo = -1; hi = 1; }
    } else { lo = -4; hi = 4; }
    cutlass::reference::host::TensorFillRandomUniform(view, seed, hi, lo, 0);
}

class GpuTimer {
public:
    GpuTimer()  { cudaEventCreate(&s_); cudaEventCreate(&e_); }
    ~GpuTimer() { cudaEventDestroy(s_); cudaEventDestroy(e_); }
    void start(cudaStream_t s = nullptr) { cudaEventRecord(s_, s); }
    void stop (cudaStream_t s = nullptr) { cudaEventRecord(e_,  s); cudaEventSynchronize(e_); }
    float ms() const { float v = 0.f; cudaEventElapsedTime(&v, s_, e_); return v; }
private:
    cudaEvent_t s_{}, e_{};
};

} // anonymous namespace

FP4GemmResult run_fp4_gemm_smoke(int m, int n, int k, int iterations) {
    FP4GemmResult res;

    FP4Gemm gemm(m, n, k);
    if (!gemm.ok()) {
        res.error_msg = gemm.last_error();
        return res;
    }

    // Recompute layouts locally for the host-tensor side (needed for fill+ref).
    auto stride_A = cutlass::make_cute_packed_stride(StrideA{}, {m, k, 1});
    auto stride_B = cutlass::make_cute_packed_stride(StrideB{}, {n, k, 1});
    auto stride_C = cutlass::make_cute_packed_stride(StrideC{}, {m, n, 1});
    auto stride_D = cutlass::make_cute_packed_stride(StrideD{}, {m, n, 1});
    auto layout_A = make_layout(make_shape(m, k, 1), stride_A);
    auto layout_B = make_layout(make_shape(n, k, 1), stride_B);
    auto layout_C = make_layout(make_shape(m, n, 1), stride_C);
    auto layout_D = make_layout(make_shape(m, n, 1), stride_D);
    auto layout_SFA =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFA(cute::make_shape(m, n, k, 1));
    auto layout_SFB =
        Sm1xxBlkScaledConfig::tile_atom_to_shape_SFB(cute::make_shape(m, n, k, 1));

    cutlass::HostTensor<typename ElementA::DataType,        cutlass::layout::PackedVectorLayout> A;
    cutlass::HostTensor<typename ElementA::ScaleFactorType, cutlass::layout::PackedVectorLayout> SFA;
    cutlass::HostTensor<typename ElementB::DataType,        cutlass::layout::PackedVectorLayout> B;
    cutlass::HostTensor<typename ElementB::ScaleFactorType, cutlass::layout::PackedVectorLayout> SFB;
    cutlass::HostTensor<ElementC, cutlass::layout::PackedVectorLayout> C;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> D;
    cutlass::HostTensor<ElementD, cutlass::layout::PackedVectorLayout> ref;

    A.reset  (cutlass::make_Coord(size(layout_A)));
    B.reset  (cutlass::make_Coord(size(layout_B)));
    C.reset  (cutlass::make_Coord(size(layout_C)));
    D.reset  (cutlass::make_Coord(size(layout_D)));
    ref.reset(cutlass::make_Coord(size(layout_D)));
    SFA.reset(cutlass::make_Coord(size(filter_zeros(layout_SFA))));
    SFB.reset(cutlass::make_Coord(size(filter_zeros(layout_SFB))));

    fill_random(A.host_view(),   2021);
    fill_random(B.host_view(),   2022);
    fill_random(C.host_view(),   2023);
    fill_random(SFA.host_view(), 2024);
    fill_random(SFB.host_view(), 2025);
    A.sync_device(); B.sync_device(); C.sync_device();
    SFA.sync_device(); SFB.sync_device();

    cutlass::device_memory::allocation<uint8_t> workspace(gemm.workspace_size_bytes());

    const float alpha = 1.0f, beta = 0.0f;
    if (!gemm.run(A.device_data(), SFA.device_data(),
                  B.device_data(), SFB.device_data(),
                  C.device_data(), D.device_data(),
                  workspace.get(), alpha, beta, nullptr)) {
        res.error_msg = gemm.last_error();
        return res;
    }
    if (cudaDeviceSynchronize() != cudaSuccess) {
        res.error_msg = "device sync failed";
        return res;
    }

    // -------- Verification against CUTLASS host reference --------
    {
        Tensor tA   = make_tensor(make_iter(A.host_data()),   layout_A);
        Tensor tSFA = make_tensor(SFA.host_data(),            layout_SFA);
        Tensor tB   = make_tensor(make_iter(B.host_data()),   layout_B);
        Tensor tSFB = make_tensor(SFB.host_data(),            layout_SFB);
        Tensor tC   = make_tensor(make_iter(C.host_data()),   layout_C);
        Tensor tRef = make_tensor(make_iter(ref.host_data()), layout_D);

        cutlass::reference::host::GettBlockScalingMainloopParams<
            ElementAccumulator, decltype(tA), decltype(tSFA),
                                 decltype(tB), decltype(tSFB)
        > mainloop{tA, tSFA, tB, tSFB};
        cutlass::reference::host::GettBlockScalingEpilogueParams<
            ElementAccumulator, ElementAccumulator, ElementAccumulator,
            decltype(tC), decltype(tRef)
        > epilogue{alpha, beta, tC, tRef};
        cutlass::reference::host::Gemm3x(mainloop, epilogue);

        D.sync_host();
        const ElementD* rp = ref.host_data();
        const ElementD* op = D.host_data();
        const size_t n_elem = static_cast<size_t>(m) * static_cast<size_t>(n);
        double max_e = 0.0, sum_e = 0.0;
        for (size_t i = 0; i < n_elem; ++i) {
            const double a = static_cast<float>(rp[i]);
            const double b = static_cast<float>(op[i]);
            const double e = std::fabs(a - b);
            if (e > max_e) max_e = e;
            sum_e += e;
        }
        res.max_abs_err  = max_e;
        res.mean_abs_err = sum_e / static_cast<double>(n_elem);

        const auto rv = ref.host_view();
        const auto ov = D.host_view();
        res.verify_passed =
            cutlass::reference::host::TensorEquals(rv, ov) &&
            cutlass::reference::host::TensorNorm(rv) > 0   &&
            cutlass::reference::host::TensorNorm(ov) > 0;
    }

    if (iterations > 0) {
        GpuTimer t;
        t.start();
        for (int i = 0; i < iterations; ++i) {
            if (!gemm.run(A.device_data(), SFA.device_data(),
                          B.device_data(), SFB.device_data(),
                          C.device_data(), D.device_data(),
                          workspace.get(), alpha, beta, nullptr)) {
                res.error_msg = gemm.last_error();
                return res;
            }
        }
        t.stop();
        res.avg_runtime_ms = static_cast<double>(t.ms()) / static_cast<double>(iterations);
        const double flop = 2.0 * m * n * k;
        res.tflops = (flop / 1e12) / (res.avg_runtime_ms / 1000.0);
    }
    res.ok = true;
    return res;
}

} // namespace f2k::cuda

#endif // CUTLASS_ARCH_MMA_SM120/121 supported
