#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

namespace f2k {

// Tensor data type. Matches what the F2K on-disk format and runtime know
// about. Stored as u8 on disk; do not renumber once shipped.
enum class DType : uint8_t {
    F16     = 0,
    BF16    = 1,
    F32     = 2,
    F8_E4M3 = 3,
    F8_E5M2 = 4,
    NVFP4   = 5,   // 4-bit E2M1, packed two-per-byte, with separate scales
    U8      = 6,
    I8      = 7,
    I32     = 8,
    Unknown = 255,
};

inline const char* dtype_name(DType d) {
    switch (d) {
        case DType::F16:     return "F16";
        case DType::BF16:    return "BF16";
        case DType::F32:     return "F32";
        case DType::F8_E4M3: return "F8_E4M3";
        case DType::F8_E5M2: return "F8_E5M2";
        case DType::NVFP4:   return "NVFP4";
        case DType::U8:      return "U8";
        case DType::I8:      return "I8";
        case DType::I32:     return "I32";
        default:             return "?";
    }
}

// Bits per element. NVFP4 is sub-byte; everything else is byte-aligned.
inline size_t dtype_element_bits(DType d) {
    switch (d) {
        case DType::F16: case DType::BF16:                 return 16;
        case DType::F32: case DType::I32:                  return 32;
        case DType::F8_E4M3: case DType::F8_E5M2:          return 8;
        case DType::U8: case DType::I8:                    return 8;
        case DType::NVFP4:                                 return 4;
        default:                                           return 0;
    }
}

inline size_t num_elements(const std::vector<int64_t>& shape) {
    if (shape.empty()) return 0;
    size_t n = 1;
    for (auto d : shape) {
        if (d < 0) return 0;
        n *= static_cast<size_t>(d);
    }
    return n;
}

// Packed weight blob byte size (rounded up for sub-byte dtypes).
inline size_t weight_bytes(DType d, const std::vector<int64_t>& shape) {
    const size_t bits = dtype_element_bits(d) * num_elements(shape);
    return (bits + 7) / 8;
}

// Non-owning view over a tensor's raw bytes, returned by both Safetensors and
// F2KReader so callers don't have to know which container they came from.
struct TensorView {
    std::string          name;
    DType                dtype = DType::Unknown;
    std::vector<int64_t> shape;

    const uint8_t* data        = nullptr;
    size_t         data_size   = 0;

    // Quantization metadata (only relevant for sub-byte dtypes).
    const uint8_t* scales         = nullptr;   // per-microblock E4M3 scales
    size_t         scales_size    = 0;
    float          tensor_scale   = 1.0f;      // global FP32 scale
    int            microblock_size = 0;        // 0 if not quantized; 16 for NVFP4
};

// Tensor data is aligned to this boundary inside F2K files so that callers
// can ingest directly into a 128-bit-aligned device pointer.
constexpr size_t F2K_DATA_ALIGN = 64;

} // namespace f2k
