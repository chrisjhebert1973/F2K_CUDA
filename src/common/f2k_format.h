#pragma once

#include "common/tensor.h"
#include "common/platform.h"

#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

namespace f2k {

// On-disk magic for F2K v1: ASCII "F2K1" stored little-endian.
constexpr uint32_t F2K_MAGIC   = 0x314B3246u;
constexpr uint32_t F2K_VERSION = 1u;

// -----------------------------------------------------------------------------
// File layout
// -----------------------------------------------------------------------------
// All offsets are file-relative.
//
//   +0   FileHeader (64 bytes, fixed)
//
//   data region (variable, all blobs aligned to F2K_DATA_ALIGN):
//        for each tensor:
//            [tensor blob]
//            [microblock scales blob]   // present only for quantized dtypes
//
//   manifest region (variable):
//        for each tensor, in original add order:
//            TensorEntryFixed (48 bytes)
//            int64 shape[rank]
//            char  name[name_length]    // no null terminator
//
// We write data before manifest so the writer can stream tensors without
// knowing the final tensor count up front. The header is patched on commit().

#pragma pack(push, 1)
struct FileHeader {
    uint32_t magic;
    uint32_t version;
    uint64_t manifest_offset;
    uint64_t manifest_size_bytes;
    uint64_t data_offset;
    uint64_t data_size_bytes;
    uint64_t total_tensors;
    uint8_t  reserved[16];
};
static_assert(sizeof(FileHeader) == 64, "FileHeader must be 64 bytes");

struct TensorEntryFixed {
    uint8_t  dtype;
    uint8_t  rank;
    uint16_t name_length;
    uint32_t microblock_size;
    uint64_t data_offset;
    uint64_t data_size;
    uint64_t scales_offset;
    uint64_t scales_size;
    float    tensor_scale;
    uint32_t reserved;
};
static_assert(sizeof(TensorEntryFixed) == 48, "TensorEntryFixed must be 48 bytes");
#pragma pack(pop)

// -----------------------------------------------------------------------------
// Writer
// -----------------------------------------------------------------------------

class F2KWriter {
public:
    explicit F2KWriter(const std::string& path);
    ~F2KWriter();

    F2KWriter(const F2KWriter&)            = delete;
    F2KWriter& operator=(const F2KWriter&) = delete;

    bool ok() const { return ok_; }
    const std::string& last_error() const { return last_error_; }

    // Append one tensor. Data is written to disk immediately; the manifest
    // entry is buffered and emitted by commit().
    bool add_tensor(const std::string& name,
                    DType dtype,
                    const std::vector<int64_t>& shape,
                    const void* blob, size_t blob_size,
                    const void* scales = nullptr, size_t scales_size = 0,
                    float tensor_scale = 1.0f,
                    int microblock_size = 0);

    // Finalize: writes manifest + patches header. After commit() the writer
    // is sealed; further add_tensor calls will fail.
    bool commit();

private:
    struct Pending {
        std::string          name;
        DType                dtype;
        std::vector<int64_t> shape;
        uint64_t             data_offset;
        uint64_t             data_size;
        uint64_t             scales_offset;
        uint64_t             scales_size;
        float                tensor_scale;
        uint32_t             microblock_size;
    };

    bool pad_to_align(size_t align);

    bool                 ok_ = true;
    bool                 sealed_ = false;
    int                  fd_ = -1;
    std::string          path_;
    std::string          last_error_;
    uint64_t             cursor_ = 0;
    std::vector<Pending> pending_;
};

// -----------------------------------------------------------------------------
// Reader
// -----------------------------------------------------------------------------

class F2KReader {
public:
    F2KReader();
    ~F2KReader();

    F2KReader(const F2KReader&)            = delete;
    F2KReader& operator=(const F2KReader&) = delete;

    bool open(const std::string& path);
    bool ok() const { return ok_; }
    const std::string& last_error() const { return last_error_; }

    const std::vector<std::string>& names()  const { return order_; }
    const TensorView*               find(const std::string& name) const;
    const FileHeader&               header() const { return header_; }

private:
    void close();

    f2k::platform::FileMapping                   map_{};        // owns the OS mapping
    void*                                        mapped_ = nullptr;   // = map_.data
    size_t                                       file_size_ = 0;      // = map_.size
    FileHeader                                   header_{};
    std::vector<std::string>                     order_;
    std::unordered_map<std::string, TensorView>  tensors_;
    bool                                         ok_ = false;
    std::string                                  last_error_;
};

} // namespace f2k
