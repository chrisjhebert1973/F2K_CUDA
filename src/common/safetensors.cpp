#include "common/safetensors.h"

#include <nlohmann/json.hpp>

#include <cstring>

namespace f2k {

using json = nlohmann::ordered_json;

DType parse_safetensors_dtype(std::string_view s) {
    if (s == "F16")     return DType::F16;
    if (s == "BF16")    return DType::BF16;
    if (s == "F32")     return DType::F32;
    if (s == "F8_E4M3") return DType::F8_E4M3;
    if (s == "F8_E5M2") return DType::F8_E5M2;
    if (s == "U8")      return DType::U8;
    if (s == "I8")      return DType::I8;
    if (s == "I32")     return DType::I32;
    return DType::Unknown;
}

Safetensors::Safetensors()  = default;
Safetensors::~Safetensors() { close(); }

void Safetensors::close() {
    f2k::platform::unmap(map_);
    mapped_ = nullptr;
    file_size_ = 0;
    tensors_.clear();
    order_.clear();
    metadata_.clear();
}

bool Safetensors::open(const std::string& path) {
    close();
    if (!f2k::platform::map_readonly(path, map_, last_error_)) {
        return false;
    }
    mapped_ = map_.data;
    file_size_ = map_.size;
    if (file_size_ < 8) {
        last_error_ = "file too small (< 8 bytes)";
        return false;
    }
    const uint8_t* base = static_cast<const uint8_t*>(mapped_);

    // Header layout: u64 LE length, then UTF-8 JSON, then the data blob.
    uint64_t header_len = 0;
    std::memcpy(&header_len, base, 8);
    if (header_len > file_size_ - 8) {
        last_error_ = "header length exceeds file size";
        return false;
    }
    const char*     json_start = reinterpret_cast<const char*>(base + 8);
    const uint8_t*  data_base  = base + 8 + header_len;
    const size_t    data_max   = file_size_ - 8 - header_len;

    json h;
    try {
        h = json::parse(std::string_view(json_start, header_len));
    } catch (const std::exception& e) {
        last_error_ = std::string("json parse: ") + e.what();
        return false;
    }
    if (!h.is_object()) {
        last_error_ = "header is not a JSON object";
        return false;
    }

    for (auto it = h.begin(); it != h.end(); ++it) {
        const std::string& name = it.key();
        const auto& entry = it.value();

        if (name == "__metadata__") {
            if (entry.is_object()) {
                for (auto m = entry.begin(); m != entry.end(); ++m) {
                    if (m.value().is_string()) {
                        metadata_[m.key()] = m.value().get<std::string>();
                    }
                }
            }
            continue;
        }
        if (!entry.is_object()) {
            last_error_ = "tensor entry not an object: " + name;
            return false;
        }

        TensorView v;
        v.name = name;
        v.dtype = parse_safetensors_dtype(entry.value("dtype", std::string()));

        if (entry.contains("shape") && entry["shape"].is_array()) {
            for (const auto& d : entry["shape"]) {
                v.shape.push_back(d.get<int64_t>());
            }
        }

        if (!entry.contains("data_offsets") || !entry["data_offsets"].is_array()
            || entry["data_offsets"].size() != 2) {
            last_error_ = "bad data_offsets for: " + name;
            return false;
        }
        const uint64_t a = entry["data_offsets"][0].get<uint64_t>();
        const uint64_t b = entry["data_offsets"][1].get<uint64_t>();
        if (b < a || b > data_max) {
            last_error_ = "data_offsets out of range for: " + name;
            return false;
        }
        v.data      = data_base + a;
        v.data_size = static_cast<size_t>(b - a);

        order_.push_back(name);
        tensors_.emplace(name, std::move(v));
    }
    return true;
}

const TensorView* Safetensors::find(const std::string& name) const {
    auto it = tensors_.find(name);
    return it == tensors_.end() ? nullptr : &it->second;
}

std::optional<std::string> Safetensors::metadata_value(std::string_view key) const {
    auto it = metadata_.find(std::string(key));
    if (it == metadata_.end()) return std::nullopt;
    return it->second;
}

} // namespace f2k
