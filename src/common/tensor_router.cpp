#include "common/tensor_router.h"

#include <algorithm>
#include <cctype>
#include <cstring>
#include <string_view>

namespace f2k {

namespace {

// Parse an integer between a prefix and a delimiter. Returns -1 if absent.
// `name` is the full tensor name; `after_prefix_idx` is the index right after
// the prefix string (e.g. "transformer_blocks.").
int parse_block_idx(const std::string& name, size_t after_prefix_idx) {
    if (after_prefix_idx >= name.size()) return -1;
    size_t i = after_prefix_idx;
    int v = 0;
    bool any = false;
    while (i < name.size() && std::isdigit(static_cast<unsigned char>(name[i]))) {
        v = v * 10 + (name[i] - '0');
        ++i;
        any = true;
    }
    if (!any) return -1;
    if (i >= name.size() || name[i] != '.') return -1;
    return v;
}

bool starts_with(const std::string& s, std::string_view p) {
    return s.size() >= p.size() && std::memcmp(s.data(), p.data(), p.size()) == 0;
}

bool slot_is_empty(const TensorView* p) { return p == nullptr; }

// Tries to populate one role slot. Returns false on duplicate assignment.
template <typename Block>
bool assign(Block& b, const TensorView*& slot, const TensorView* v,
            int block_idx, const char* role, std::string& err) {
    if (!slot_is_empty(slot)) {
        err = std::string("duplicate role '") + role + "' in block " +
              std::to_string(block_idx);
        return false;
    }
    slot = v;
    (void)b;
    return true;
}

} // anonymous namespace

TensorRouter::TensorRouter(const F2KModelLoader& loader) : loader_(loader) {}

bool TensorRouter::build() {
    constexpr std::string_view DBL = "transformer_blocks.";
    constexpr std::string_view SGL = "single_transformer_blocks.";

    // First pass: discover block count by scanning name patterns.
    int max_double = -1, max_single = -1;
    for (const auto& name : loader_.names()) {
        if (starts_with(name, SGL)) {
            int idx = parse_block_idx(name, SGL.size());
            if (idx > max_single) max_single = idx;
        } else if (starts_with(name, DBL)) {
            int idx = parse_block_idx(name, DBL.size());
            if (idx > max_double) max_double = idx;
        }
    }
    if (max_double < 0 || max_single < 0) {
        last_error_ = "no transformer block tensors found";
        return false;
    }
    double_blocks_.assign(max_double + 1, DoubleBlockTensors{});
    single_blocks_.assign(max_single + 1, SingleBlockTensors{});

    // Second pass: route each tensor into its slot.
    for (const auto& name : loader_.names()) {
        const TensorView* v = loader_.find(name);
        if (!v) continue;

        // --- Single-stream block (check first since prefix subsumes DBL) ---
        if (starts_with(name, SGL)) {
            const int idx = parse_block_idx(name, SGL.size());
            if (idx < 0 || idx >= static_cast<int>(single_blocks_.size())) {
                unrouted_.push_back(name); continue;
            }
            auto& sb = single_blocks_[idx];
            const std::string suffix = name.substr(SGL.size() + std::to_string(idx).size() + 1);
            const TensorView** slot = nullptr;
            if      (suffix == "attn.norm_q.weight")          slot = &sb.norm_q;
            else if (suffix == "attn.norm_k.weight")          slot = &sb.norm_k;
            else if (suffix == "attn.to_qkv_mlp_proj.weight") slot = &sb.to_qkv_mlp_proj;
            else if (suffix == "attn.to_out.weight")          slot = &sb.to_out;
            else { unrouted_.push_back(name); continue; }
            if (!assign(sb, *slot, v, idx, suffix.c_str(), last_error_)) return false;
            continue;
        }

        // --- Double-stream block ---
        if (starts_with(name, DBL)) {
            const int idx = parse_block_idx(name, DBL.size());
            if (idx < 0 || idx >= static_cast<int>(double_blocks_.size())) {
                unrouted_.push_back(name); continue;
            }
            auto& db = double_blocks_[idx];
            const std::string suffix = name.substr(DBL.size() + std::to_string(idx).size() + 1);
            const TensorView** slot = nullptr;
            if      (suffix == "attn.norm_q.weight")          slot = &db.norm_q;
            else if (suffix == "attn.norm_k.weight")          slot = &db.norm_k;
            else if (suffix == "attn.norm_added_q.weight")    slot = &db.norm_added_q;
            else if (suffix == "attn.norm_added_k.weight")    slot = &db.norm_added_k;
            else if (suffix == "attn.to_q.weight")            slot = &db.to_q;
            else if (suffix == "attn.to_k.weight")            slot = &db.to_k;
            else if (suffix == "attn.to_v.weight")            slot = &db.to_v;
            else if (suffix == "attn.to_out.0.weight")        slot = &db.to_out;
            else if (suffix == "attn.add_q_proj.weight")      slot = &db.add_q_proj;
            else if (suffix == "attn.add_k_proj.weight")      slot = &db.add_k_proj;
            else if (suffix == "attn.add_v_proj.weight")      slot = &db.add_v_proj;
            else if (suffix == "attn.to_add_out.weight")      slot = &db.to_add_out;
            else if (suffix == "ff.linear_in.weight")         slot = &db.ff_in;
            else if (suffix == "ff.linear_out.weight")        slot = &db.ff_out;
            else if (suffix == "ff_context.linear_in.weight")  slot = &db.ff_ctx_in;
            else if (suffix == "ff_context.linear_out.weight") slot = &db.ff_ctx_out;
            else { unrouted_.push_back(name); continue; }
            if (!assign(db, *slot, v, idx, suffix.c_str(), last_error_)) return false;
            continue;
        }

        // --- Globals ---
        if      (name == "x_embedder.weight")                                       globals_.x_embedder       = v;
        else if (name == "context_embedder.weight")                                  globals_.context_embedder = v;
        else if (name == "time_guidance_embed.timestep_embedder.linear_1.weight")    globals_.time_embed_1     = v;
        else if (name == "time_guidance_embed.timestep_embedder.linear_2.weight")    globals_.time_embed_2     = v;
        else if (name == "double_stream_modulation_img.linear.weight")               globals_.double_mod_img   = v;
        else if (name == "double_stream_modulation_txt.linear.weight")               globals_.double_mod_txt   = v;
        else if (name == "single_stream_modulation.linear.weight")                   globals_.single_mod       = v;
        else if (name == "norm_out.linear.weight")                                   globals_.norm_out         = v;
        else if (name == "proj_out.weight")                                          globals_.proj_out         = v;
        else {
            unrouted_.push_back(name);
        }
    }

    // Required-slot completeness checks.
    auto require = [&](const TensorView* p, const char* what) {
        if (!p) { last_error_ = std::string("missing required global tensor: ") + what; return false; }
        return true;
    };
    if (!require(globals_.x_embedder,       "x_embedder")) return false;
    if (!require(globals_.context_embedder, "context_embedder")) return false;
    if (!require(globals_.time_embed_1,     "time_embed_1")) return false;
    if (!require(globals_.time_embed_2,     "time_embed_2")) return false;
    if (!require(globals_.double_mod_img,   "double_stream_modulation_img")) return false;
    if (!require(globals_.double_mod_txt,   "double_stream_modulation_txt")) return false;
    if (!require(globals_.single_mod,       "single_stream_modulation")) return false;
    if (!require(globals_.norm_out,         "norm_out")) return false;
    if (!require(globals_.proj_out,         "proj_out")) return false;

    for (int i = 0; i < num_double_blocks(); ++i) {
        const auto& db = double_blocks_[i];
        auto need = [&](const TensorView* p, const char* what) {
            if (!p) { last_error_ = std::string("double block ") + std::to_string(i) + " missing " + what; return false; }
            return true;
        };
        if (!need(db.norm_q,        "norm_q"))        return false;
        if (!need(db.norm_k,        "norm_k"))        return false;
        if (!need(db.norm_added_q,  "norm_added_q"))  return false;
        if (!need(db.norm_added_k,  "norm_added_k"))  return false;
        if (!need(db.to_q,          "to_q"))          return false;
        if (!need(db.to_k,          "to_k"))          return false;
        if (!need(db.to_v,          "to_v"))          return false;
        if (!need(db.to_out,        "to_out"))        return false;
        if (!need(db.add_q_proj,    "add_q_proj"))    return false;
        if (!need(db.add_k_proj,    "add_k_proj"))    return false;
        if (!need(db.add_v_proj,    "add_v_proj"))    return false;
        if (!need(db.to_add_out,    "to_add_out"))    return false;
        if (!need(db.ff_in,         "ff_in"))         return false;
        if (!need(db.ff_out,        "ff_out"))        return false;
        if (!need(db.ff_ctx_in,     "ff_ctx_in"))     return false;
        if (!need(db.ff_ctx_out,    "ff_ctx_out"))    return false;
    }
    for (int i = 0; i < num_single_blocks(); ++i) {
        const auto& sb = single_blocks_[i];
        auto need = [&](const TensorView* p, const char* what) {
            if (!p) { last_error_ = std::string("single block ") + std::to_string(i) + " missing " + what; return false; }
            return true;
        };
        if (!need(sb.norm_q,          "norm_q"))          return false;
        if (!need(sb.norm_k,          "norm_k"))          return false;
        if (!need(sb.to_qkv_mlp_proj, "to_qkv_mlp_proj")) return false;
        if (!need(sb.to_out,          "to_out"))          return false;
    }
    return true;
}

} // namespace f2k
