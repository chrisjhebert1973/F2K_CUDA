#include "common/f2k_format.h"

#include <cerrno>
#include <cstring>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace f2k {

// ---------- helpers ----------

static bool write_all(int fd, const void* p, size_t n) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    while (n) {
        ssize_t w = ::write(fd, b, n);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        b += w;
        n -= static_cast<size_t>(w);
    }
    return true;
}

static bool pwrite_all(int fd, const void* p, size_t n, off_t off) {
    const uint8_t* b = static_cast<const uint8_t*>(p);
    while (n) {
        ssize_t w = ::pwrite(fd, b, n, off);
        if (w < 0) {
            if (errno == EINTR) continue;
            return false;
        }
        b += w;
        n -= static_cast<size_t>(w);
        off += w;
    }
    return true;
}

// ---------- writer ----------

F2KWriter::F2KWriter(const std::string& path) : path_(path) {
    fd_ = ::open(path.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
    if (fd_ < 0) {
        ok_ = false;
        last_error_ = "open(): " + std::string(std::strerror(errno));
        return;
    }
    // Reserve the header; we'll patch it in commit().
    FileHeader hdr{};
    if (!write_all(fd_, &hdr, sizeof(hdr))) {
        ok_ = false;
        last_error_ = "write(header reservation)";
        return;
    }
    cursor_ = sizeof(FileHeader);
}

F2KWriter::~F2KWriter() {
    if (fd_ >= 0) ::close(fd_);
}

bool F2KWriter::pad_to_align(size_t align) {
    const uint64_t r = cursor_ % align;
    if (r == 0) return true;
    const uint64_t pad = align - r;
    static const uint8_t z[F2K_DATA_ALIGN] = {};
    uint64_t left = pad;
    while (left) {
        const size_t chunk = (left > sizeof(z)) ? sizeof(z) : static_cast<size_t>(left);
        if (!write_all(fd_, z, chunk)) return false;
        left -= chunk;
    }
    cursor_ += pad;
    return true;
}

bool F2KWriter::add_tensor(const std::string& name,
                           DType dtype,
                           const std::vector<int64_t>& shape,
                           const void* blob, size_t blob_size,
                           const void* scales, size_t scales_size,
                           float tensor_scale,
                           int microblock_size) {
    if (!ok_ || sealed_) {
        last_error_ = sealed_ ? "writer already sealed" : last_error_;
        return false;
    }
    if (!pad_to_align(F2K_DATA_ALIGN)) {
        ok_ = false;
        last_error_ = "pad before blob";
        return false;
    }

    Pending p{};
    p.name            = name;
    p.dtype           = dtype;
    p.shape           = shape;
    p.tensor_scale    = tensor_scale;
    p.microblock_size = static_cast<uint32_t>(microblock_size);

    p.data_offset = cursor_;
    p.data_size   = blob_size;
    if (blob_size > 0 && !write_all(fd_, blob, blob_size)) {
        ok_ = false;
        last_error_ = "write(blob)";
        return false;
    }
    cursor_ += blob_size;

    if (scales && scales_size > 0) {
        if (!pad_to_align(F2K_DATA_ALIGN)) {
            ok_ = false;
            last_error_ = "pad before scales";
            return false;
        }
        p.scales_offset = cursor_;
        p.scales_size   = scales_size;
        if (!write_all(fd_, scales, scales_size)) {
            ok_ = false;
            last_error_ = "write(scales)";
            return false;
        }
        cursor_ += scales_size;
    }

    pending_.push_back(std::move(p));
    return true;
}

bool F2KWriter::commit() {
    if (!ok_) return false;
    if (sealed_) {
        last_error_ = "already committed";
        return false;
    }
    if (!pad_to_align(F2K_DATA_ALIGN)) {
        ok_ = false;
        last_error_ = "pad before manifest";
        return false;
    }
    const uint64_t data_end = cursor_;
    const uint64_t data_size = data_end - sizeof(FileHeader);
    const uint64_t manifest_offset = data_end;

    std::vector<uint8_t> manifest;
    auto put = [&](const void* p, size_t n) {
        const uint8_t* b = static_cast<const uint8_t*>(p);
        manifest.insert(manifest.end(), b, b + n);
    };
    for (const auto& p : pending_) {
        TensorEntryFixed te{};
        te.dtype           = static_cast<uint8_t>(p.dtype);
        te.rank            = static_cast<uint8_t>(p.shape.size());
        te.name_length     = static_cast<uint16_t>(p.name.size());
        te.microblock_size = p.microblock_size;
        te.data_offset     = p.data_offset;
        te.data_size       = p.data_size;
        te.scales_offset   = p.scales_offset;
        te.scales_size     = p.scales_size;
        te.tensor_scale    = p.tensor_scale;
        te.reserved        = 0;
        put(&te, sizeof(te));
        for (int64_t d : p.shape) put(&d, sizeof(d));
        put(p.name.data(), p.name.size());
    }
    if (!write_all(fd_, manifest.data(), manifest.size())) {
        ok_ = false;
        last_error_ = "write(manifest)";
        return false;
    }

    FileHeader hdr{};
    hdr.magic               = F2K_MAGIC;
    hdr.version             = F2K_VERSION;
    hdr.manifest_offset     = manifest_offset;
    hdr.manifest_size_bytes = manifest.size();
    hdr.data_offset         = sizeof(FileHeader);
    hdr.data_size_bytes     = data_size;
    hdr.total_tensors       = pending_.size();
    if (!pwrite_all(fd_, &hdr, sizeof(hdr), 0)) {
        ok_ = false;
        last_error_ = "pwrite(header)";
        return false;
    }
    if (::fsync(fd_) != 0) {
        ok_ = false;
        last_error_ = "fsync()";
        return false;
    }
    sealed_ = true;
    return true;
}

// ---------- reader ----------

F2KReader::F2KReader()  = default;
F2KReader::~F2KReader() { close(); }

void F2KReader::close() {
    if (mapped_) {
        ::munmap(mapped_, file_size_);
        mapped_ = nullptr;
    }
    if (fd_ >= 0) {
        ::close(fd_);
        fd_ = -1;
    }
    file_size_ = 0;
    order_.clear();
    tensors_.clear();
    ok_ = false;
}

bool F2KReader::open(const std::string& path) {
    close();
    fd_ = ::open(path.c_str(), O_RDONLY);
    if (fd_ < 0) {
        last_error_ = "open(): " + std::string(std::strerror(errno));
        return false;
    }
    struct stat st;
    if (::fstat(fd_, &st) != 0) {
        last_error_ = "fstat()";
        return false;
    }
    file_size_ = static_cast<size_t>(st.st_size);
    if (file_size_ < sizeof(FileHeader)) {
        last_error_ = "file shorter than header";
        return false;
    }
    mapped_ = ::mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mapped_ == MAP_FAILED) {
        last_error_ = "mmap(): " + std::string(std::strerror(errno));
        mapped_ = nullptr;
        return false;
    }
    const uint8_t* base = static_cast<const uint8_t*>(mapped_);
    std::memcpy(&header_, base, sizeof(header_));
    if (header_.magic != F2K_MAGIC) {
        last_error_ = "bad magic";
        return false;
    }
    if (header_.version != F2K_VERSION) {
        last_error_ = "unsupported version";
        return false;
    }
    if (header_.manifest_offset + header_.manifest_size_bytes > file_size_) {
        last_error_ = "manifest out of bounds";
        return false;
    }

    const uint8_t* m     = base + header_.manifest_offset;
    const uint8_t* m_end = m + header_.manifest_size_bytes;
    for (uint64_t i = 0; i < header_.total_tensors; ++i) {
        if (m + sizeof(TensorEntryFixed) > m_end) {
            last_error_ = "manifest truncated at entry " + std::to_string(i);
            return false;
        }
        TensorEntryFixed te;
        std::memcpy(&te, m, sizeof(te));
        m += sizeof(te);

        TensorView v;
        v.dtype           = static_cast<DType>(te.dtype);
        v.tensor_scale    = te.tensor_scale;
        v.microblock_size = static_cast<int>(te.microblock_size);

        v.shape.resize(te.rank);
        for (int r = 0; r < te.rank; ++r) {
            if (m + 8 > m_end) { last_error_ = "shape truncated"; return false; }
            std::memcpy(&v.shape[r], m, 8);
            m += 8;
        }
        if (m + te.name_length > m_end) {
            last_error_ = "name truncated";
            return false;
        }
        v.name = std::string(reinterpret_cast<const char*>(m), te.name_length);
        m += te.name_length;

        if (te.data_offset + te.data_size > file_size_) {
            last_error_ = "tensor blob out of bounds: " + v.name;
            return false;
        }
        v.data      = base + te.data_offset;
        v.data_size = te.data_size;
        if (te.scales_offset) {
            if (te.scales_offset + te.scales_size > file_size_) {
                last_error_ = "scales out of bounds: " + v.name;
                return false;
            }
            v.scales      = base + te.scales_offset;
            v.scales_size = te.scales_size;
        }
        order_.push_back(v.name);
        tensors_.emplace(v.name, std::move(v));
    }
    ok_ = true;
    return true;
}

const TensorView* F2KReader::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

} // namespace f2k
