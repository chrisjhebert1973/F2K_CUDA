// TensorRouter test — synthetic (cheap, deterministic) + real model (opportunistic).

#include "common/f2k_format.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <string>
#include <vector>

using namespace f2k;

namespace {

bool write_synth_model(const std::string& path, int n_double, int n_single) {
    F2KWriter w(path);
    if (!w.ok()) return false;
    std::vector<uint8_t> tiny{0xAB};
    std::vector<int64_t> tiny_shape{1};
    auto add = [&](const std::string& name) {
        return w.add_tensor(name, DType::U8, tiny_shape, tiny.data(), tiny.size());
    };

    // Globals.
    if (!add("x_embedder.weight"))                                                       return false;
    if (!add("context_embedder.weight"))                                                  return false;
    if (!add("time_guidance_embed.timestep_embedder.linear_1.weight"))                    return false;
    if (!add("time_guidance_embed.timestep_embedder.linear_2.weight"))                    return false;
    if (!add("double_stream_modulation_img.linear.weight"))                               return false;
    if (!add("double_stream_modulation_txt.linear.weight"))                               return false;
    if (!add("single_stream_modulation.linear.weight"))                                   return false;
    if (!add("norm_out.linear.weight"))                                                   return false;
    if (!add("proj_out.weight"))                                                          return false;

    // Per double-stream block.
    for (int i = 0; i < n_double; ++i) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        if (!add(p + "attn.norm_q.weight"))        return false;
        if (!add(p + "attn.norm_k.weight"))        return false;
        if (!add(p + "attn.norm_added_q.weight"))  return false;
        if (!add(p + "attn.norm_added_k.weight"))  return false;
        if (!add(p + "attn.to_q.weight"))          return false;
        if (!add(p + "attn.to_k.weight"))          return false;
        if (!add(p + "attn.to_v.weight"))          return false;
        if (!add(p + "attn.to_out.0.weight"))      return false;
        if (!add(p + "attn.add_q_proj.weight"))    return false;
        if (!add(p + "attn.add_k_proj.weight"))    return false;
        if (!add(p + "attn.add_v_proj.weight"))    return false;
        if (!add(p + "attn.to_add_out.weight"))    return false;
        if (!add(p + "ff.linear_in.weight"))       return false;
        if (!add(p + "ff.linear_out.weight"))      return false;
        if (!add(p + "ff_context.linear_in.weight"))  return false;
        if (!add(p + "ff_context.linear_out.weight")) return false;
    }
    for (int i = 0; i < n_single; ++i) {
        const std::string p = "single_transformer_blocks." + std::to_string(i) + ".";
        if (!add(p + "attn.norm_q.weight"))          return false;
        if (!add(p + "attn.norm_k.weight"))          return false;
        if (!add(p + "attn.to_qkv_mlp_proj.weight")) return false;
        if (!add(p + "attn.to_out.weight"))          return false;
    }
    return w.commit();
}

bool test_synthetic() {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "f2k_router_test";
    fs::create_directories(dir);
    const auto p = (dir / "synth.f2k1").string();
    if (!write_synth_model(p, /*double*/ 8, /*single*/ 24)) {
        std::fprintf(stderr, "synth write failed\n"); return false;
    }
    F2KModelLoader ld;
    if (!ld.add_shard(p)) { std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return false; }
    TensorRouter r(ld);
    if (!r.build()) { std::fprintf(stderr, "router: %s\n", r.last_error().c_str()); return false; }

    if (r.num_double_blocks() != 8 || r.num_single_blocks() != 24) {
        std::fprintf(stderr, "block counts wrong: %d / %d\n",
                     r.num_double_blocks(), r.num_single_blocks());
        return false;
    }
    if (!r.unrouted().empty()) {
        std::fprintf(stderr, "unexpected unrouted: %zu names\n", r.unrouted().size());
        for (const auto& n : r.unrouted()) std::fprintf(stderr, "  %s\n", n.c_str());
        return false;
    }
    std::printf("[1/2] synthetic: PASS  (8 double + 24 single, 0 unrouted)\n");
    return true;
}

bool test_real_if_present() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("[2/2] real-model: SKIPPED  (converted shards not present)\n");
        return true;
    }
    F2KModelLoader ld;
    if (!ld.add_shard(s1.string())) { std::fprintf(stderr, "real shard1: %s\n", ld.last_error().c_str()); return false; }
    if (!ld.add_shard(s2.string())) { std::fprintf(stderr, "real shard2: %s\n", ld.last_error().c_str()); return false; }

    TensorRouter r(ld);
    if (!r.build()) { std::fprintf(stderr, "router: %s\n", r.last_error().c_str()); return false; }

    if (r.num_double_blocks() != 8) {
        std::fprintf(stderr, "expected 8 double-stream blocks, got %d\n", r.num_double_blocks());
        return false;
    }
    if (r.num_single_blocks() != 24) {
        std::fprintf(stderr, "expected 24 single-stream blocks, got %d\n", r.num_single_blocks());
        return false;
    }
    if (!r.unrouted().empty()) {
        std::fprintf(stderr, "unrouted tensors in real model: %zu\n", r.unrouted().size());
        for (const auto& n : r.unrouted()) std::fprintf(stderr, "  %s\n", n.c_str());
        return false;
    }

    // Spot-check that a few block weights have the shapes we expect.
    const auto& sb0 = r.single_blocks()[0];
    if (sb0.to_qkv_mlp_proj->shape != std::vector<int64_t>{36864, 4096}) {
        std::fprintf(stderr, "single[0].to_qkv_mlp_proj shape unexpected\n"); return false;
    }
    if (sb0.to_out->shape != std::vector<int64_t>{4096, 16384}) {
        std::fprintf(stderr, "single[0].to_out shape unexpected\n"); return false;
    }
    const auto& db0 = r.double_blocks()[0];
    if (db0.to_q->shape != std::vector<int64_t>{4096, 4096}) {
        std::fprintf(stderr, "double[0].to_q shape unexpected\n"); return false;
    }
    if (db0.ff_in->shape != std::vector<int64_t>{24576, 4096}) {
        std::fprintf(stderr, "double[0].ff_in shape unexpected\n"); return false;
    }

    std::printf("[2/2] real-model: PASS  (8 double + 24 single + 9 globals, every slot filled, 0 unrouted)\n");
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_synthetic();
    ok &= test_real_if_present();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
