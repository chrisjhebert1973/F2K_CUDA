#include "common/f2k_model_loader.h"

namespace f2k {

F2KModelLoader::F2KModelLoader()  = default;
F2KModelLoader::~F2KModelLoader() = default;

bool F2KModelLoader::add_shard(const std::string& path) {
    auto r = std::make_unique<F2KReader>();
    if (!r->open(path)) {
        last_error_ = "open(" + path + "): " + r->last_error();
        return false;
    }
    const int shard_idx = static_cast<int>(shards_.size());
    for (const auto& name : r->names()) {
        auto [it, inserted] = name_to_shard_.try_emplace(name, shard_idx);
        if (!inserted) {
            last_error_ = "duplicate tensor name across shards: " + name +
                          " (already in shard " + std::to_string(it->second) +
                          ", also in " + path + ")";
            return false;
        }
        order_.push_back(name);
    }
    shards_.push_back(std::move(r));
    shard_paths_.push_back(path);
    return true;
}

size_t F2KModelLoader::total_data_bytes() const {
    size_t n = 0;
    for (const auto& s : shards_) n += s->header().data_size_bytes;
    return n;
}

const TensorView* F2KModelLoader::find(const std::string& name) const {
    auto it = name_to_shard_.find(name);
    if (it == name_to_shard_.end()) return nullptr;
    return shards_[it->second]->find(name);
}

std::optional<int> F2KModelLoader::locate(const std::string& name) const {
    auto it = name_to_shard_.find(name);
    if (it == name_to_shard_.end()) return std::nullopt;
    return it->second;
}

} // namespace f2k
