// Compare QwenEncoder layer outputs against a HF golden file produced by
// tools/qwen3_golden.py.  Localizes bugs by comparing 3 spread layer indices.
//
// Golden file layout:
//   int32  seq, n_layers, hidden, pad
//   int32[seq]                         token IDs (already padded to `seq`)
//   bfloat16[n_layers+1][seq][hidden]  layer hidden states
//      index 0   : embed_tokens output
//      index 1..N: post-layer i hidden state

#include "backend/cuda/qwen_encoder.h"
#include "common/f2k_model_loader.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <string>
#include <vector>

namespace fs = std::filesystem;

struct Golden {
    int seq = 0, n_layers = 0, hidden = 0;
    std::vector<int32_t> ids;
    std::vector<uint16_t> hs;   // [n_layers+1, seq, hidden] BF16 raw

    // Pointer to a specific layer index in `hs`.
    const __nv_bfloat16* layer(int idx) const {
        return reinterpret_cast<const __nv_bfloat16*>(
            hs.data() + (size_t)idx * seq * hidden);
    }
};

bool load_golden(const std::string& path, Golden& g) {
    std::ifstream f(path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open %s failed\n", path.c_str()); return false; }
    int32_t hdr[4];
    f.read(reinterpret_cast<char*>(hdr), sizeof(hdr));
    g.seq = hdr[0]; g.n_layers = hdr[1]; g.hidden = hdr[2];
    g.ids.resize(g.seq);
    f.read(reinterpret_cast<char*>(g.ids.data()), g.seq * sizeof(int32_t));
    const size_t hs_n = (size_t)(g.n_layers + 1) * g.seq * g.hidden;
    g.hs.resize(hs_n);
    f.read(reinterpret_cast<char*>(g.hs.data()), hs_n * sizeof(uint16_t));
    if (!f) { std::fprintf(stderr, "read truncated\n"); return false; }
    return true;
}

struct LayerStats {
    int layer_idx;
    double max_err, mean_err, cos_sim, magnitude_ours, magnitude_ref;
};

LayerStats compare(const __nv_bfloat16* ours, const __nv_bfloat16* ref,
                   int seq, int hidden, int layer_idx) {
    const size_t n = (size_t)seq * hidden;
    double max_err = 0, sum_err = 0;
    double dot = 0, mag_o = 0, mag_r = 0;
    for (size_t i = 0; i < n; ++i) {
        const double a = __bfloat162float(ours[i]);
        const double b = __bfloat162float(ref[i]);
        const double e = std::fabs(a - b);
        max_err = std::max(max_err, e);
        sum_err += e;
        dot   += a * b;
        mag_o += a * a;
        mag_r += b * b;
    }
    LayerStats s{};
    s.layer_idx = layer_idx;
    s.max_err = max_err;
    s.mean_err = sum_err / n;
    s.cos_sim = (mag_o > 0 && mag_r > 0) ? dot / std::sqrt(mag_o * mag_r) : 0;
    s.magnitude_ours = std::sqrt(mag_o / n);
    s.magnitude_ref  = std::sqrt(mag_r / n);
    return s;
}

int main(int argc, char** argv) {
    const std::string golden_path = (argc > 1) ? argv[1] : "/tmp/qwen3_golden.bin";
    if (!fs::exists(golden_path)) {
        std::printf("SKIP: %s missing (run tools/qwen3_golden.py first)\n", golden_path.c_str());
        return 0;
    }
    Golden g;
    if (!load_golden(golden_path, g)) return 1;
    std::printf("Golden: seq=%d n_layers=%d hidden=%d\n", g.seq, g.n_layers, g.hidden);
    std::printf("First 8 token IDs: ");
    for (int i = 0; i < 8 && i < g.seq; ++i) std::printf("%d ", g.ids[i]);
    std::printf("\n");

    const char* home = std::getenv("HOME");
    f2k::F2KModelLoader ld;
    for (int i = 1; i <= 4; ++i) {
        char p[256];
        std::snprintf(p, sizeof(p), "%s/models/flux2-klein-9B/qwen3_f2k/shard-%05d.f2k1", home, i);
        if (!ld.add_shard(p)) { std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return 1; }
    }

    // First: directly compare the embedding output to golden[0] (no transformer).
    // We do this with a manual embed_lookup via the encoder's table.
    // Trick: temporarily run encoder with capture_layers={0,0,0} (any),
    // but pull the embedding from a separate path. For now we just trust the
    // overall stats from layer 0 ≈ shifted by 1 layer.
    // -- skip direct embed compare for now; the first run gives enough signal.

    f2k::cuda::QwenEncoder::Config cfg{};
    cfg.seq = g.seq; cfg.loader = &ld;

    f2k::cuda::QwenEncoder enc(cfg);
    if (!enc.ok()) { std::fprintf(stderr, "ctor: %s\n", enc.last_error()); return 1; }

    int32_t* d_ids; cudaMalloc(&d_ids, g.seq * sizeof(int32_t));
    cudaMemcpy(d_ids, g.ids.data(), g.seq * sizeof(int32_t), cudaMemcpyHostToDevice);

    const size_t out_elems = (size_t)g.seq * enc.output_hidden();
    void *d_out, *d_ws;
    cudaMalloc(&d_out, out_elems * sizeof(__nv_bfloat16));
    cudaMalloc(&d_ws, enc.workspace_size_bytes());

    auto t0 = std::chrono::steady_clock::now();
    if (!enc.forward(d_ids, d_out, d_ws, enc.workspace_size_bytes())) {
        std::fprintf(stderr, "fwd: %s\n", enc.last_error()); return 1;
    }
    cudaDeviceSynchronize();
    auto t1 = std::chrono::steady_clock::now();
    std::printf("Forward: %.2f ms\n",
                std::chrono::duration<double, std::milli>(t1 - t0).count());

    std::vector<__nv_bfloat16> h_out(out_elems);
    cudaMemcpy(h_out.data(), d_out, out_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
    cudaFree(d_ids); cudaFree(d_out); cudaFree(d_ws);

    // h_out is [seq, 3*hidden] with 3 captures concatenated along feature dim:
    //   col [0,           hidden)        = capture_layers[0]
    //   col [hidden,    2*hidden)        = capture_layers[1]
    //   col [2*hidden,  3*hidden)        = capture_layers[2]
    auto extract = [&](int slot, std::vector<__nv_bfloat16>& dst) {
        const int hidden = g.hidden;
        const int total  = enc.output_hidden();
        const int col_off = slot * hidden;
        dst.assign((size_t)g.seq * hidden, __nv_bfloat16());
        for (int s = 0; s < g.seq; ++s) {
            std::memcpy(dst.data() + (size_t)s * hidden,
                        h_out.data() + (size_t)s * total + col_off,
                        hidden * sizeof(__nv_bfloat16));
        }
    };

    std::printf("\n%-6s %-12s %-12s %-10s %-12s %-12s\n",
                "Layer", "max_err", "mean_err", "cos_sim", "rms_ours", "rms_ref");
    std::printf("%-6s %-12s %-12s %-10s %-12s %-12s\n",
                "-----", "-------", "--------", "-------", "--------", "-------");

    // Sweep a bunch of layers in 3-at-a-time batches.
    const int probes[] = { 0, 5, 11, 17, 23, 29, 32, 33, 34 };
    constexpr int N_PROBES = sizeof(probes) / sizeof(int);
    bool pass = true;
    std::vector<__nv_bfloat16> slot;
    for (int batch = 0; batch < N_PROBES; batch += 3) {
        f2k::cuda::QwenEncoder::Config cb = cfg;
        cb.capture_layers = { probes[std::min(batch + 0, N_PROBES - 1)],
                              probes[std::min(batch + 1, N_PROBES - 1)],
                              probes[std::min(batch + 2, N_PROBES - 1)] };
        f2k::cuda::QwenEncoder e2(cb);
        if (!e2.ok()) { std::fprintf(stderr, "ctor: %s\n", e2.last_error()); return 1; }
        int32_t* d_ids2; cudaMalloc(&d_ids2, g.seq * sizeof(int32_t));
        cudaMemcpy(d_ids2, g.ids.data(), g.seq * sizeof(int32_t), cudaMemcpyHostToDevice);
        void *d_out2, *d_ws2;
        cudaMalloc(&d_out2, out_elems * sizeof(__nv_bfloat16));
        cudaMalloc(&d_ws2, e2.workspace_size_bytes());
        if (!e2.forward(d_ids2, d_out2, d_ws2, e2.workspace_size_bytes())) {
            std::fprintf(stderr, "fwd: %s\n", e2.last_error()); return 1;
        }
        cudaDeviceSynchronize();
        cudaMemcpy(h_out.data(), d_out2, out_elems * sizeof(__nv_bfloat16), cudaMemcpyDeviceToHost);
        cudaFree(d_ids2); cudaFree(d_out2); cudaFree(d_ws2);

        for (int i = 0; i < 3 && batch + i < N_PROBES; ++i) {
            const int layer = probes[batch + i];
            extract(i, slot);
            const LayerStats s = compare(slot.data(), g.layer(layer + 1),
                                          g.seq, g.hidden, layer);
            std::printf("%-6d %-12.4f %-12.5f %-10.5f %-12.3f %-12.3f\n",
                        s.layer_idx, s.max_err, s.mean_err, s.cos_sim,
                        s.magnitude_ours, s.magnitude_ref);
            if (!(s.cos_sim > 0.90)) pass = false;
        }
    }
    std::printf("\n%s\n", pass ? "ALL OK" : "FAIL");
    return pass ? 0 : 1;
}
