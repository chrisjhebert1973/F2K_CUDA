// F2KModelLoader — multi-shard reader for a full model split across N F2K1
// files. Each shard is mmapped via F2KReader; the loader unifies the per-
// shard tensor maps into one name → TensorView lookup.
//
// Lifecycle: add_shard() is called once per file (typically as soon as the
// loader is constructed). Once all shards are added, find() works across the
// union and last_shard_added_to() reports per-tensor provenance for logging.

#pragma once

#include "common/f2k_format.h"
#include "common/tensor.h"

#include <cstdint>
#include <memory>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace f2k {

class F2KModelLoader {
public:
    F2KModelLoader();
    ~F2KModelLoader();

    F2KModelLoader(const F2KModelLoader&)            = delete;
    F2KModelLoader& operator=(const F2KModelLoader&) = delete;

    // Opens a F2K1 file and merges its tensor map into the unified index.
    // Returns false on open failure or if a tensor name in this shard
    // collides with one already loaded from an earlier shard.
    bool add_shard(const std::string& path);

    size_t num_shards()       const { return shards_.size(); }
    size_t total_tensors()    const { return order_.size(); }
    size_t total_data_bytes() const;

    const std::vector<std::string>& names() const { return order_; }

    // Returns nullptr if the tensor is not present in any shard.
    const TensorView* find(const std::string& name) const;

    // For diagnostics: shard index (0-based) that owns a given tensor.
    std::optional<int> locate(const std::string& name) const;

    // Shard's source path (for logging).
    const std::string& shard_path(int shard_idx) const { return shard_paths_.at(shard_idx); }

    const std::string& last_error() const { return last_error_; }

private:
    std::vector<std::unique_ptr<F2KReader>> shards_;
    std::vector<std::string>                shard_paths_;
    std::unordered_map<std::string, int>    name_to_shard_;
    std::vector<std::string>                order_;
    std::string                             last_error_;
};

} // namespace f2k
