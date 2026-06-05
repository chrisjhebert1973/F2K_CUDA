// safetensors → F2K1 converter with optional NVFP4 quantization.
//
// Usage:
//   f2k_convert <input.safetensors> <output.f2k1> [opts]
//     --quant nvfp4|bf16     default: bf16 pass-through
//     --include <substr>     keep only tensors whose name contains substr (repeatable)
//     --exclude <substr>     drop tensors whose name contains substr (repeatable)
//     --dry-run              list policy decisions, don't write
//     -v                     per-tensor verbose log
//
// NVFP4 policy (when --quant nvfp4):
//   - rank-2 tensors with N % 128 == 0 and K % 64 == 0 whose name ends in
//     ".weight" → quantized to NVFP4 (E2M1 packed + per-microblock E4M3 scales).
//   - Everything else → pass through in source dtype (typically BF16).
//
// Storage layout for NVFP4 tensors in F2K:
//   - data:   [N, K/2] row-major packed FP4 bytes (low nibble = even k, high = odd)
//   - scales: [N, K/16] row-major E4M3 bytes (one scale per (n, k_microblock))
//   - tensor_scale: 1.0 (per-microblock scale absorbs the scaling fully)
//   - microblock_size: 16
//
// This is a *natural* row-major layout — the runtime loader is expected to
// re-pack into the CUTLASS-specific tiled SFB layout at load time. We chose
// this over storing in CUTLASS layout directly to keep the on-disk format
// independent of the GEMM kernel's internals.

#include "common/f2k_format.h"
#include "common/safetensors.h"
#include "common/tensor.h"

#include "cutlass/float_subbyte.h"
#include "cutlass/float8.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <string_view>
#include <vector>

using namespace f2k;

namespace {

enum class Quant { BF16PassThrough, NVFP4, MXFP8 };

struct Args {
    std::string input;
    std::string output;
    Quant       quant = Quant::BF16PassThrough;
    std::vector<std::string> includes;
    std::vector<std::string> excludes;
    std::vector<std::string> keep_bf16; // names matching any of these stay BF16 even with --quant nvfp4
    bool        dry_run = false;
    bool        verbose = false;
};

void print_usage() {
    std::fprintf(stderr,
        "Usage: f2k_convert <in.safetensors> <out.f2k1> [opts]\n"
        "  --quant nvfp4|bf16     default: bf16 pass-through\n"
        "  --include <substr>     keep tensors whose name contains substr\n"
        "  --exclude <substr>     drop tensors whose name contains substr\n"
        "  --dry-run              print policy decisions, don't write\n"
        "  -v                     verbose per-tensor logging\n"
    );
}

bool parse_args(int argc, char** argv, Args& a) {
    if (argc < 3) return false;
    a.input = argv[1];
    a.output = argv[2];
    for (int i = 3; i < argc; ++i) {
        std::string_view s = argv[i];
        if (s == "--include" && i + 1 < argc) { a.includes.emplace_back(argv[++i]); }
        else if (s == "--exclude" && i + 1 < argc) { a.excludes.emplace_back(argv[++i]); }
        else if (s == "--keep-bf16" && i + 1 < argc) { a.keep_bf16.emplace_back(argv[++i]); }
        else if (s == "--quant" && i + 1 < argc) {
            std::string_view q = argv[++i];
            if      (q == "nvfp4") a.quant = Quant::NVFP4;
            else if (q == "mxfp8") a.quant = Quant::MXFP8;
            else if (q == "bf16")  a.quant = Quant::BF16PassThrough;
            else { std::fprintf(stderr, "unknown --quant: %s\n", argv[i]); return false; }
        }
        else if (s == "--dry-run")         { a.dry_run = true; }
        else if (s == "-v" || s == "--verbose") { a.verbose = true; }
        else { std::fprintf(stderr, "unknown arg: %s\n", argv[i]); return false; }
    }
    return true;
}

bool name_matches_filter(const std::string& name,
                         const std::vector<std::string>& includes,
                         const std::vector<std::string>& excludes) {
    if (!includes.empty()) {
        bool any = false;
        for (const auto& s : includes) if (name.find(s) != std::string::npos) { any = true; break; }
        if (!any) return false;
    }
    for (const auto& s : excludes) if (name.find(s) != std::string::npos) return false;
    return true;
}

std::string human_bytes(size_t b) {
    char buf[32];
    if      (b < 1024)              std::snprintf(buf, sizeof(buf), "%zu B",   b);
    else if (b < 1024ULL*1024)      std::snprintf(buf, sizeof(buf), "%.1f KiB", b / 1024.0);
    else if (b < 1024ULL*1024*1024) std::snprintf(buf, sizeof(buf), "%.1f MiB", b / (1024.0*1024));
    else                            std::snprintf(buf, sizeof(buf), "%.2f GiB", b / (1024.0*1024*1024));
    return buf;
}

bool ends_with(const std::string& s, std::string_view suf) {
    return s.size() >= suf.size() && std::memcmp(s.data() + s.size() - suf.size(), suf.data(), suf.size()) == 0;
}

bool should_quantize(const std::string& name, const TensorView& v) {
    if (v.shape.size() != 2) return false;
    if (!(v.dtype == DType::F16 || v.dtype == DType::BF16 || v.dtype == DType::F32)) return false;
    if (!ends_with(name, ".weight")) return false;
    const int64_t N = v.shape[0], K = v.shape[1];
    if (N % 128 != 0 || K % 64 != 0) return false;
    return true;
}

// ---- BF16/FP16/FP32 → FP32 element reader ------------------------------

float element_to_fp32(const uint8_t* base, size_t i, DType dt) {
    switch (dt) {
        case DType::F32: {
            float f; std::memcpy(&f, base + i * 4, 4); return f;
        }
        case DType::BF16: {
            uint16_t b; std::memcpy(&b, base + i * 2, 2);
            __nv_bfloat16 v = *reinterpret_cast<const __nv_bfloat16*>(&b);
            return __bfloat162float(v);
        }
        case DType::F16: {
            uint16_t b; std::memcpy(&b, base + i * 2, 2);
            __half h = *reinterpret_cast<const __half*>(&b);
            return __half2float(h);
        }
        default: return 0.0f;
    }
}

// ---- NVFP4 quantization (natural row-major layout) ----------------------

struct QuantOut {
    std::vector<uint8_t> packed;     // [N, K/2] row-major
    std::vector<uint8_t> scales;     // [N, K/16] row-major (E4M3 stored as bytes)
    float                tensor_scale = 1.0f;
    int                  microblock_size = 16;
};

QuantOut quantize_nvfp4(const uint8_t* src, DType src_dt, int N, int K) {
    QuantOut q;
    q.packed.assign(static_cast<size_t>(N) * (K / 2), 0);
    q.scales.assign(static_cast<size_t>(N) * (K / 16), 0);
    constexpr float E2M1_MAX = 6.0f;

    const int n_kblocks = K / 16;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            float absmax = 0.0f;
            for (int i = 0; i < 16; ++i) {
                const float v = element_to_fp32(src, static_cast<size_t>(n) * K + kb * 16 + i, src_dt);
                absmax = std::max(absmax, std::fabs(v));
            }
            const float scale = (absmax > 1e-30f) ? (absmax / E2M1_MAX) : 1.0f;
            const cutlass::float_ue4m3_t scale_e4m3(scale);
            q.scales[static_cast<size_t>(n) * n_kblocks + kb] =
                static_cast<uint8_t>(scale_e4m3.raw());

            for (int i = 0; i < 16; ++i) {
                const int k = kb * 16 + i;
                const float v = element_to_fp32(src, static_cast<size_t>(n) * K + k, src_dt);
                const cutlass::float_e2m1_t e2m1(v / scale);
                const uint8_t nibble = static_cast<uint8_t>(e2m1.raw()) & 0xFu;
                const size_t byte_idx = static_cast<size_t>(n) * (K / 2) + (k >> 1);
                if ((k & 1) == 0) {
                    q.packed[byte_idx] = (q.packed[byte_idx] & 0xF0u) | nibble;
                } else {
                    q.packed[byte_idx] = (q.packed[byte_idx] & 0x0Fu) | static_cast<uint8_t>(nibble << 4);
                }
            }
        }
    }
    return q;
}

// Dequantize for validation. Returns reconstructed FP32 [N*K].
std::vector<float> dequantize_nvfp4(const QuantOut& q, int N, int K) {
    std::vector<float> out(static_cast<size_t>(N) * K, 0.0f);
    const int n_kblocks = K / 16;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            cutlass::float_ue4m3_t sc;
            sc.raw() = q.scales[static_cast<size_t>(n) * n_kblocks + kb];
            const float scale = static_cast<float>(sc);
            for (int i = 0; i < 16; ++i) {
                const int k = kb * 16 + i;
                const size_t byte_idx = static_cast<size_t>(n) * (K / 2) + (k >> 1);
                const uint8_t byte = q.packed[byte_idx];
                const uint8_t nibble = (k & 1) ? (byte >> 4) : (byte & 0xFu);
                cutlass::float_e2m1_t e2m1;
                e2m1.raw() = nibble;
                out[static_cast<size_t>(n) * K + k] =
                    static_cast<float>(e2m1) * scale;
            }
        }
    }
    return out;
}

// ---- MXFP8 quantization: E4M3 data [N,K] + per-32 UE8M0 scales [N,K/32]. ----
// Mirrors the on-device scheme in linear.cu (scale X = 2^ceil(log2(absmax/448))).
QuantOut quantize_mxfp8(const uint8_t* src, DType src_dt, int N, int K) {
    QuantOut q;
    q.microblock_size = 32;
    q.packed.assign(static_cast<size_t>(N) * K, 0);          // E4M3, 1 byte/elem
    q.scales.assign(static_cast<size_t>(N) * (K / 32), 0);   // UE8M0, 1 byte each
    constexpr float E4M3_MAX = 448.0f;

    const int n_kblocks = K / 32;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            float absmax = 0.0f;
            for (int i = 0; i < 32; ++i) {
                const float v = element_to_fp32(src, static_cast<size_t>(n) * K + kb * 32 + i, src_dt);
                absmax = std::max(absmax, std::fabs(v));
            }
            const float X = (absmax > 0.0f)
                ? std::ldexp(1.0f, static_cast<int>(std::ceil(std::log2(absmax / E4M3_MAX))))
                : 1.0f;
            const cutlass::float_ue8m0_t scale_u(X);
            q.scales[static_cast<size_t>(n) * n_kblocks + kb] =
                static_cast<uint8_t>(scale_u.raw());
            for (int i = 0; i < 32; ++i) {
                const int k = kb * 32 + i;
                const float v = element_to_fp32(src, static_cast<size_t>(n) * K + k, src_dt);
                const cutlass::float_e4m3_t e(v / X);
                q.packed[static_cast<size_t>(n) * K + k] = static_cast<uint8_t>(e.raw());
            }
        }
    }
    return q;
}

std::vector<float> dequantize_mxfp8(const QuantOut& q, int N, int K) {
    std::vector<float> out(static_cast<size_t>(N) * K, 0.0f);
    const int n_kblocks = K / 32;
    for (int n = 0; n < N; ++n) {
        for (int kb = 0; kb < n_kblocks; ++kb) {
            cutlass::float_ue8m0_t sc;
            sc.raw() = q.scales[static_cast<size_t>(n) * n_kblocks + kb];
            const float scale = static_cast<float>(sc);
            for (int i = 0; i < 32; ++i) {
                const int k = kb * 32 + i;
                cutlass::float_e4m3_t e;
                e.raw() = q.packed[static_cast<size_t>(n) * K + k];
                out[static_cast<size_t>(n) * K + k] = static_cast<float>(e) * scale;
            }
        }
    }
    return out;
}

struct QuantStats {
    double max_abs_err = 0.0;
    double mean_abs_err = 0.0;
};

QuantStats compare_against_source(const uint8_t* src, DType src_dt, int N, int K,
                                  const std::vector<float>& reconstructed) {
    double max_e = 0, sum_e = 0;
    const size_t n = static_cast<size_t>(N) * K;
    for (size_t i = 0; i < n; ++i) {
        const double v = element_to_fp32(src, i, src_dt);
        const double e = std::fabs(reconstructed[i] - v);
        max_e = std::max(max_e, e);
        sum_e += e;
    }
    return {max_e, sum_e / static_cast<double>(n)};
}

} // anonymous namespace

int main(int argc, char** argv) {
    Args args;
    if (!parse_args(argc, argv, args)) { print_usage(); return 1; }

    Safetensors st;
    if (!st.open(args.input)) {
        std::fprintf(stderr, "open(%s): %s\n", args.input.c_str(), st.last_error().c_str());
        return 1;
    }
    std::printf("opened %s  (%s, %zu tensors)\n",
                args.input.c_str(), human_bytes(st.file_size()).c_str(), st.names().size());

    // First pass: classify per-tensor.
    size_t n_passthrough = 0, n_quant = 0;
    size_t bytes_passthrough = 0, bytes_quant_src = 0, bytes_quant_out = 0;
    for (const auto& name : st.names()) {
        if (!name_matches_filter(name, args.includes, args.excludes)) continue;
        const TensorView* v = st.find(name);
        if (!v) continue;
        bool keep_bf16 = false;
        for (const auto& kp : args.keep_bf16)
            if (name.find(kp) != std::string::npos) { keep_bf16 = true; break; }
        const bool want_q = (args.quant == Quant::NVFP4 || args.quant == Quant::MXFP8);
        const bool quant = want_q && !keep_bf16 && should_quantize(name, *v);
        if (quant) {
            ++n_quant;
            bytes_quant_src += v->data_size;
            const int64_t N = v->shape[0], K = v->shape[1];
            if (args.quant == Quant::MXFP8)
                bytes_quant_out += static_cast<size_t>(N) * K +        // E4M3 data
                                    static_cast<size_t>(N) * K / 32;   // UE8M0 scales
            else
                bytes_quant_out += static_cast<size_t>(N) * K / 2 +    // FP4 packed
                                    static_cast<size_t>(N) * K / 16;   // E4M3 scales
        } else {
            ++n_passthrough;
            bytes_passthrough += v->data_size;
        }
        if (args.verbose) {
            std::string shape_s = "(";
            for (size_t i = 0; i < v->shape.size(); ++i) {
                shape_s += std::to_string(v->shape[i]);
                if (i + 1 < v->shape.size()) shape_s += ",";
            }
            shape_s += ")";
            std::printf("  %-8s %-12s %-60s %s %s\n",
                        quant ? "NVFP4" : dtype_name(v->dtype),
                        shape_s.c_str(), name.c_str(),
                        human_bytes(v->data_size).c_str(),
                        quant ? "[quantized]" : "");
        }
    }
    std::printf("plan: %zu passthrough (%s) + %zu %s (%s src → %s out)\n",
                n_passthrough, human_bytes(bytes_passthrough).c_str(),
                n_quant, args.quant == Quant::MXFP8 ? "MXFP8" : "NVFP4",
                human_bytes(bytes_quant_src).c_str(),
                human_bytes(bytes_quant_out).c_str());
    if (args.dry_run) return 0;

    // Second pass: write.
    F2KWriter w(args.output);
    if (!w.ok()) {
        std::fprintf(stderr, "F2KWriter(%s): %s\n", args.output.c_str(), w.last_error().c_str());
        return 1;
    }
    const auto t0 = std::chrono::steady_clock::now();
    double worst_max_err = 0.0;
    double worst_mean_err = 0.0;
    std::string worst_name_max, worst_name_mean;

    for (const auto& name : st.names()) {
        if (!name_matches_filter(name, args.includes, args.excludes)) continue;
        const TensorView* v = st.find(name);
        if (!v) continue;
        bool keep_bf16 = false;
        for (const auto& kp : args.keep_bf16)
            if (name.find(kp) != std::string::npos) { keep_bf16 = true; break; }
        const bool want_q = (args.quant == Quant::NVFP4 || args.quant == Quant::MXFP8);
        const bool quant = want_q && !keep_bf16 && should_quantize(name, *v);
        if (!quant) {
            if (!w.add_tensor(name, v->dtype, v->shape, v->data, v->data_size)) {
                std::fprintf(stderr, "add_tensor(%s): %s\n", name.c_str(), w.last_error().c_str());
                return 1;
            }
            continue;
        }
        const int N = static_cast<int>(v->shape[0]);
        const int K = static_cast<int>(v->shape[1]);
        const bool mxfp8 = (args.quant == Quant::MXFP8);
        QuantOut q = mxfp8 ? quantize_mxfp8(v->data, v->dtype, N, K)
                           : quantize_nvfp4(v->data, v->dtype, N, K);

        // Cheap online validation: dequantize and compare to source.
        auto recon = mxfp8 ? dequantize_mxfp8(q, N, K) : dequantize_nvfp4(q, N, K);
        auto stats = compare_against_source(v->data, v->dtype, N, K, recon);
        if (stats.max_abs_err > worst_max_err)   { worst_max_err  = stats.max_abs_err;  worst_name_max  = name; }
        if (stats.mean_abs_err > worst_mean_err) { worst_mean_err = stats.mean_abs_err; worst_name_mean = name; }
        if (args.verbose) {
            std::printf("    [quant] %s  max_err=%.5f  mean_err=%.5f\n",
                        name.c_str(), stats.max_abs_err, stats.mean_abs_err);
        }

        if (!w.add_tensor(name, mxfp8 ? DType::F8_E4M3 : DType::NVFP4, v->shape,
                          q.packed.data(), q.packed.size(),
                          q.scales.data(), q.scales.size(),
                          q.tensor_scale, q.microblock_size)) {
            std::fprintf(stderr, "add_tensor(%s): %s\n", name.c_str(), w.last_error().c_str());
            return 1;
        }
    }
    if (!w.commit()) {
        std::fprintf(stderr, "commit: %s\n", w.last_error().c_str());
        return 1;
    }
    const auto t1 = std::chrono::steady_clock::now();
    const double secs = std::chrono::duration<double>(t1 - t0).count();
    std::printf("wrote → %s in %.2fs\n", args.output.c_str(), secs);
    if (n_quant > 0) {
        std::printf("%s quant quality: worst max_err=%.4f (%s)  worst mean_err=%.5f (%s)\n",
                    args.quant == Quant::MXFP8 ? "MXFP8" : "NVFP4",
                    worst_max_err, worst_name_max.c_str(),
                    worst_mean_err, worst_name_mean.c_str());
    }
    return 0;
}
