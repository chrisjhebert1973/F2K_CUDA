// Per-model-root architecture manifest: <model root>/f2k_model.json.
//
// A "model root" (e.g. ~/models/flux2-klein-4B) holds the converted F2K
// components — transformer_{f2k,mxfp8}/, qwen3_f2k/, vae_f2k/vae.f2k1,
// tokenizer/ — plus this manifest describing the dims that differ between
// FLUX.2-klein variants. A missing file or missing key means klein-9B
// (the defaults below), so the stock root and finetune overlay dirs need
// no manifest at all.
//
// {
//   "transformer": { "n_heads":24, "ffn_dim":9216, "num_double_blocks":5,
//                    "num_single_blocks":20, "t5_dim":7680 },
//   "qwen":        { "hidden":2560, "ffn_dim":9728 }
// }

#pragma once

#include <nlohmann/json.hpp>

#include <array>
#include <filesystem>
#include <fstream>
#include <string>

namespace f2k {

struct ModelManifest {
    // FLUX.2 MMDiT (defaults: klein-9B)
    int   t5_dim     = 12288;
    int   time_dim   = 256;
    int   n_heads    = 32;
    int   head_dim   = 128;
    int   ffn_dim    = 12288;
    int   n_double   = 8;
    int   n_single   = 24;
    float rope_theta = 2000.0f;

    // Qwen3 text encoder (defaults: Qwen3-8B)
    int   q_hidden     = 4096;
    int   q_heads      = 32;
    int   q_kv_heads   = 8;
    int   q_head_dim   = 128;
    int   q_ffn        = 12288;
    int   q_layers     = 36;
    int   q_vocab      = 151936;
    float q_rope_theta = 1e6f;
    std::array<int, 3> capture_layers = { 8, 17, 26 };

    // Reads root/f2k_model.json; absent file/keys keep the 9B defaults.
    // Returns false only on a malformed file (error in *err).
    bool load(const std::filesystem::path& root, std::string* err = nullptr) {
        const auto p = root / "f2k_model.json";
        std::ifstream f(p);
        if (!f) return true;   // no manifest → stock 9B dims
        try {
            const auto j = nlohmann::json::parse(f);
            if (j.contains("transformer")) {
                const auto& t = j["transformer"];
                t5_dim     = t.value("t5_dim", t5_dim);
                time_dim   = t.value("time_dim", time_dim);
                n_heads    = t.value("n_heads", n_heads);
                head_dim   = t.value("head_dim", head_dim);
                ffn_dim    = t.value("ffn_dim", ffn_dim);
                n_double   = t.value("num_double_blocks", n_double);
                n_single   = t.value("num_single_blocks", n_single);
                rope_theta = t.value("rope_theta", rope_theta);
            }
            if (j.contains("qwen")) {
                const auto& q = j["qwen"];
                q_hidden     = q.value("hidden", q_hidden);
                q_heads      = q.value("n_heads", q_heads);
                q_kv_heads   = q.value("n_kv_heads", q_kv_heads);
                q_head_dim   = q.value("head_dim", q_head_dim);
                q_ffn        = q.value("ffn_dim", q_ffn);
                q_layers     = q.value("n_layers", q_layers);
                q_vocab      = q.value("vocab_size", q_vocab);
                q_rope_theta = q.value("rope_theta", q_rope_theta);
                if (q.contains("capture_layers")) {
                    const auto& c = q["capture_layers"];
                    for (int i = 0; i < 3 && i < (int)c.size(); ++i)
                        capture_layers[i] = c[i].get<int>();
                }
            }
            if (t5_dim != 3 * q_hidden) {
                if (err) *err = "manifest inconsistent: t5_dim (" + std::to_string(t5_dim) +
                                ") != 3*qwen.hidden (" + std::to_string(3 * q_hidden) + ")";
                return false;
            }
        } catch (const std::exception& e) {
            if (err) *err = std::string("parse ") + p.string() + ": " + e.what();
            return false;
        }
        return true;
    }
};

} // namespace f2k
