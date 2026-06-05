// F2KModelLoader test:
//   1. Synthetic two-shard round-trip — deterministic, runs in CI without
//      the model files.
//   2. Real-model smoke check — runs only if the converted FLUX.2-klein
//      shards exist; verifies cross-shard totals + spot-checks known tensors.

#include "common/f2k_format.h"
#include "common/f2k_model_loader.h"
#include "common/tensor.h"

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <string>
#include <vector>

using namespace f2k;

namespace {

bool write_synth_shard(const std::string& path,
                       const std::vector<std::pair<std::string, std::vector<uint8_t>>>& tensors) {
    F2KWriter w(path);
    if (!w.ok()) return false;
    for (const auto& [name, bytes] : tensors) {
        std::vector<int64_t> shape{static_cast<int64_t>(bytes.size())};
        if (!w.add_tensor(name, DType::U8, shape, bytes.data(), bytes.size())) return false;
    }
    return w.commit();
}

bool test_synthetic_two_shard() {
    namespace fs = std::filesystem;
    fs::path dir = fs::temp_directory_path() / "f2k_loader_test";
    fs::create_directories(dir);
    const auto p1 = (dir / "shard1.f2k1").string();
    const auto p2 = (dir / "shard2.f2k1").string();

    std::vector<std::pair<std::string, std::vector<uint8_t>>> t1 = {
        {"alpha.weight", {0x01, 0x02, 0x03, 0x04}},
        {"alpha.bias",   {0xAA, 0xBB}},
        {"beta.weight",  {0x10, 0x20, 0x30}},
    };
    std::vector<std::pair<std::string, std::vector<uint8_t>>> t2 = {
        {"gamma.weight", {0xF0, 0xF1, 0xF2, 0xF3, 0xF4}},
        {"delta.bias",   {0x55}},
    };
    if (!write_synth_shard(p1, t1)) { std::fprintf(stderr, "write shard1\n"); return false; }
    if (!write_synth_shard(p2, t2)) { std::fprintf(stderr, "write shard2\n"); return false; }

    F2KModelLoader ld;
    if (!ld.add_shard(p1)) { std::fprintf(stderr, "add shard1: %s\n", ld.last_error().c_str()); return false; }
    if (!ld.add_shard(p2)) { std::fprintf(stderr, "add shard2: %s\n", ld.last_error().c_str()); return false; }

    if (ld.num_shards()    != 2) { std::fprintf(stderr, "num_shards != 2\n"); return false; }
    if (ld.total_tensors() != 5) { std::fprintf(stderr, "total_tensors != 5\n"); return false; }

    // Spot-check lookups.
    const auto* a = ld.find("alpha.weight");
    if (!a || a->data_size != 4 || std::memcmp(a->data, "\x01\x02\x03\x04", 4) != 0) {
        std::fprintf(stderr, "alpha.weight mismatch\n"); return false;
    }
    const auto* d = ld.find("delta.bias");
    if (!d || d->data_size != 1 || d->data[0] != 0x55) {
        std::fprintf(stderr, "delta.bias mismatch\n"); return false;
    }
    if (ld.find("nonexistent")) { std::fprintf(stderr, "nonexistent should be null\n"); return false; }

    auto loc_a = ld.locate("alpha.weight");
    auto loc_d = ld.locate("delta.bias");
    if (!loc_a || *loc_a != 0) { std::fprintf(stderr, "alpha not in shard 0\n"); return false; }
    if (!loc_d || *loc_d != 1) { std::fprintf(stderr, "delta not in shard 1\n"); return false; }

    // Duplicate name should fail.
    F2KModelLoader dup;
    if (!dup.add_shard(p1)) return false;
    if (dup.add_shard(p1)) {
        std::fprintf(stderr, "duplicate add should have failed\n"); return false;
    }
    std::printf("[1/2] synthetic two-shard: PASS  (5 tensors, lookups + locate + dup-detect)\n");
    return true;
}

bool test_real_model_if_present() {
    namespace fs = std::filesystem;
    const fs::path s1 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00001.f2k1";
    const fs::path s2 = fs::path(std::getenv("HOME") ? std::getenv("HOME") : "")
                        / "models" / "flux2-klein-9B" / "transformer_f2k" / "shard-00002.f2k1";
    if (!fs::exists(s1) || !fs::exists(s2)) {
        std::printf("[2/2] real-model: SKIPPED  (converted shards not at ~/models/flux2-klein-9B/transformer_f2k/)\n");
        return true;
    }

    F2KModelLoader ld;
    if (!ld.add_shard(s1.string())) {
        std::fprintf(stderr, "real shard1: %s\n", ld.last_error().c_str()); return false;
    }
    if (!ld.add_shard(s2.string())) {
        std::fprintf(stderr, "real shard2: %s\n", ld.last_error().c_str()); return false;
    }

    // Expected totals from converter output we recorded earlier.
    const size_t want = 155 + 78;
    if (ld.total_tensors() != want) {
        std::fprintf(stderr, "real model: expected %zu tensors, got %zu\n",
                     want, ld.total_tensors());
        return false;
    }

    // Spot-check a few canonical tensor names.
    const char* known[] = {
        "single_stream_modulation.linear.weight",            // shard 1
        "context_embedder.weight",                           // shard 1
        "single_transformer_blocks.0.attn.to_qkv_mlp_proj.weight",  // shard 1
        "transformer_blocks.0.attn.add_q_proj.weight",       // shard 1
        // Anything that lives in shard 2 (e.g., later blocks). The split is
        // by file position; let's spot-check whichever shard the late blocks
        // landed in.
        "single_transformer_blocks.23.attn.to_qkv_mlp_proj.weight",  // probably shard 2
    };
    int missing = 0;
    for (const char* n : known) {
        const TensorView* v = ld.find(n);
        if (!v) { std::fprintf(stderr, "  MISSING: %s\n", n); ++missing; continue; }
        auto loc = ld.locate(n);
        std::printf("  %-66s  shard=%d  dtype=%-6s  shape=(",
                    n, loc ? *loc : -1, dtype_name(v->dtype));
        for (size_t i = 0; i < v->shape.size(); ++i)
            std::printf("%lld%s", (long long)v->shape[i],
                        i + 1 < v->shape.size() ? "," : "");
        std::printf(")  scaled=%s\n", v->dtype == DType::NVFP4 ? "yes" : "no");
    }
    if (missing) return false;

    const double total_gib = ld.total_data_bytes() / (1024.0 * 1024.0 * 1024.0);
    std::printf("[2/2] real-model: PASS  (%zu tensors, %.2f GiB on disk)\n",
                ld.total_tensors(), total_gib);
    return true;
}

} // namespace

int main() {
    bool ok = true;
    ok &= test_synthetic_two_shard();
    ok &= test_real_model_if_present();
    std::printf("%s\n", ok ? "ALL OK" : "FAILURES");
    return ok ? 0 : 1;
}
