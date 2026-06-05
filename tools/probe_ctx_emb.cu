// Targeted probe: run ContextEmbedder (our NVFP4 text Linear) on two different
// diffusers prompt_embeds and check whether prompt distinction survives.
//
// Compares input cosine-similarity-per-row to output cosine-similarity-per-row.
// If input rows differ but output rows do NOT, the NVFP4 path is the bug.

#include "backend/cuda/embeddings.h"
#include "backend/cuda/linear.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"
#include "common/tensor.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <vector>

namespace fs = std::filesystem;

static f2k::cuda::PreQuantNVFP4 view_pq(const f2k::TensorView* t) {
    f2k::cuda::PreQuantNVFP4 r{};
    r.packed = t->data;
    r.scales = t->scales;
    r.N = (int)t->shape[0]; r.K = (int)t->shape[1];
    r.microblock_size = t->microblock_size ? t->microblock_size : 16;
    r.tensor_scale = t->tensor_scale ? t->tensor_scale : 1.0f;
    return r;
}

struct Embeds {
    int seq, dim;
    std::vector<__nv_bfloat16> data;
};

bool load_embeds(const std::string& path, Embeds& e) {
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    int32_t hdr[2]; f.read((char*)hdr, 8);
    e.seq = hdr[0]; e.dim = hdr[1];
    e.data.resize((size_t)e.seq * e.dim);
    f.read((char*)e.data.data(), e.data.size() * sizeof(__nv_bfloat16));
    return (bool)f;
}

// Per-row stats comparing two [seq, dim] BF16 tensors.
struct RowStats { double cos_sim, max_diff, rms_a, rms_b; };
RowStats row_compare(const __nv_bfloat16* a, const __nv_bfloat16* b, int seq, int dim, int row) {
    const __nv_bfloat16* ra = a + (size_t)row * dim;
    const __nv_bfloat16* rb = b + (size_t)row * dim;
    double dot = 0, na = 0, nb = 0, mx = 0;
    for (int i = 0; i < dim; ++i) {
        const double va = __bfloat162float(ra[i]);
        const double vb = __bfloat162float(rb[i]);
        dot += va * vb; na += va * va; nb += vb * vb;
        mx = std::max(mx, std::fabs(va - vb));
    }
    RowStats s{};
    s.cos_sim = (na > 0 && nb > 0) ? dot / std::sqrt(na * nb) : 0.0;
    s.max_diff = mx;
    s.rms_a = std::sqrt(na / dim);
    s.rms_b = std::sqrt(nb / dim);
    return s;
}

int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: probe_ctx_emb <cat_embeds.bin> <truck_embeds.bin>\n");
        return 1;
    }
    Embeds A, B;
    if (!load_embeds(argv[1], A)) { std::fprintf(stderr, "load %s failed\n", argv[1]); return 1; }
    if (!load_embeds(argv[2], B)) { std::fprintf(stderr, "load %s failed\n", argv[2]); return 1; }
    if (A.seq != B.seq || A.dim != B.dim) { std::fprintf(stderr, "shape mismatch\n"); return 1; }
    const int seq = A.seq, dim = A.dim;
    std::printf("Loaded embeds: [%d, %d]\n", seq, dim);

    // Input row comparisons.
    std::printf("\nINPUT (diffusers prompt_embeds, before ctx_emb):\n");
    std::printf("%-6s %-10s %-12s %-10s %-10s\n",
                "Row", "cos_sim", "max_diff", "rms_a", "rms_b");
    for (int r : {0, 1, 4, 8, 12, 16, 20, 100, 200, 400}) {
        const RowStats s = row_compare(A.data.data(), B.data.data(), seq, dim, r);
        std::printf("%-6d %-10.4f %-12.3f %-10.3f %-10.3f\n",
                    r, s.cos_sim, s.max_diff, s.rms_a, s.rms_b);
    }

    // Load FluxTransformer weights + build ContextEmbedder.
    const char* home = std::getenv("HOME");
    f2k::F2KModelLoader ld;
    for (int i = 1; i <= 2; ++i) {
        char p[256];
        std::snprintf(p, sizeof(p), "%s/models/flux2-klein-9B/transformer_f2k/shard-%05d.f2k1", home, i);
        if (!ld.add_shard(p)) { std::fprintf(stderr, "shard: %s\n", ld.last_error().c_str()); return 1; }
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return 1; }
    const auto* W = ld.find("context_embedder.weight");
    if (!W) { std::fprintf(stderr, "no context_embedder.weight\n"); return 1; }
    auto W_pq = view_pq(W);

    f2k::cuda::ContextEmbedder::Config ec{};
    ec.batch_rows = seq;
    ec.t5_dim = dim;
    ec.hidden_dim = 4096;
    ec.W = &W_pq;
    f2k::cuda::ContextEmbedder ctx(ec);
    if (!ctx.ok()) { std::fprintf(stderr, "ctx_emb ctor: %s\n", ctx.last_error()); return 1; }

    void *d_in_a, *d_in_b, *d_out_a, *d_out_b, *d_ws;
    cudaMalloc(&d_in_a, A.data.size() * 2);
    cudaMalloc(&d_in_b, B.data.size() * 2);
    cudaMalloc(&d_out_a, (size_t)seq * 4096 * 2);
    cudaMalloc(&d_out_b, (size_t)seq * 4096 * 2);
    cudaMalloc(&d_ws, ctx.workspace_size_bytes());
    cudaMemcpy(d_in_a, A.data.data(), A.data.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(d_in_b, B.data.data(), B.data.size() * 2, cudaMemcpyHostToDevice);

    if (!ctx.forward(d_in_a, d_out_a, d_ws, ctx.workspace_size_bytes())) {
        std::fprintf(stderr, "ctx fwd A: %s\n", ctx.last_error()); return 1;
    }
    if (!ctx.forward(d_in_b, d_out_b, d_ws, ctx.workspace_size_bytes())) {
        std::fprintf(stderr, "ctx fwd B: %s\n", ctx.last_error()); return 1;
    }
    cudaDeviceSynchronize();

    std::vector<__nv_bfloat16> out_a((size_t)seq * 4096), out_b((size_t)seq * 4096);
    cudaMemcpy(out_a.data(), d_out_a, out_a.size() * 2, cudaMemcpyDeviceToHost);
    cudaMemcpy(out_b.data(), d_out_b, out_b.size() * 2, cudaMemcpyDeviceToHost);
    cudaFree(d_in_a); cudaFree(d_in_b); cudaFree(d_out_a); cudaFree(d_out_b); cudaFree(d_ws);

    // Output row comparisons.
    std::printf("\nOUTPUT (after ctx_emb NVFP4 Linear, hidden=4096):\n");
    std::printf("%-6s %-10s %-12s %-10s %-10s\n",
                "Row", "cos_sim", "max_diff", "rms_a", "rms_b");
    for (int r : {0, 1, 4, 8, 12, 16, 20, 100, 200, 400}) {
        const RowStats s = row_compare(out_a.data(), out_b.data(), seq, 4096, r);
        std::printf("%-6d %-10.4f %-12.3f %-10.3f %-10.3f\n",
                    r, s.cos_sim, s.max_diff, s.rms_a, s.rms_b);
    }
    return 0;
}
