#pragma once

#include "common/tensor.h"

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>

namespace f2k {

// Mmaps a HuggingFace safetensors file and exposes non-owning views over
// each tensor's raw byte range. Header JSON parsing is done eagerly; tensor
// data is faulted in lazily by the OS as you touch it.
class Safetensors {
public:
    Safetensors();
    ~Safetensors();

    Safetensors(const Safetensors&)            = delete;
    Safetensors& operator=(const Safetensors&) = delete;

    bool open(const std::string& path);

    const std::vector<std::string>& names() const { return order_; }
    const TensorView* find(const std::string& name) const;

    size_t file_size() const { return file_size_; }

    std::optional<std::string> metadata_value(std::string_view key) const;

    const std::string& last_error() const { return last_error_; }

private:
    void close();

    int                 fd_        = -1;
    void*               mapped_    = nullptr;
    size_t              file_size_ = 0;
    std::vector<std::string>                       order_;
    std::unordered_map<std::string, TensorView>    tensors_;
    std::unordered_map<std::string, std::string>   metadata_;
    std::string         last_error_;
};

DType parse_safetensors_dtype(std::string_view s);

} // namespace f2k
