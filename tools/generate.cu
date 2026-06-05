// End-to-end image generation:
//   [optional] prompt token IDs (from tools/encode_prompt.py) → Qwen3-8B
//   text encoder → 12288-dim conditioning. If no --tokens file is supplied,
//   falls back to random text (noise images).
//   + random initial latent → 4-step flow-matching denoise → unpatchify
//   → VAE decode → write PPM.

#include "backend/cuda/flux_transformer.h"
#include "backend/cuda/qwen_encoder.h"
#include "backend/cuda/sampler.h"
#include "backend/cuda/vae_decoder.h"
#include "backend/cuda/kernels/patchify.h"

#include "common/f2k_format.h"
#include "common/f2k_model_loader.h"
#include "common/tensor_router.h"

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <random>
#include <string>
#include <vector>

namespace fs = std::filesystem;

static inline __nv_bfloat16 fp32_to_bf16(float v) { return __float2bfloat16(v); }
static inline float         bf16_to_fp32(__nv_bfloat16 v) { return __bfloat162float(v); }

// Maps pixels in [-1, 1] (per-channel) to uint8 RGB and writes a PPM P6.
static bool write_ppm(const std::string& path,
                      const float* rgb_chw, int C, int H, int W) {
    if (C != 3) return false;
    std::vector<uint8_t> bytes(static_cast<size_t>(H) * W * 3);
    for (int h = 0; h < H; ++h) {
        for (int w = 0; w < W; ++w) {
            for (int c = 0; c < 3; ++c) {
                const float v = rgb_chw[(c * H + h) * W + w];
                const float u = std::clamp((v + 1.0f) * 0.5f, 0.0f, 1.0f);
                bytes[(h * W + w) * 3 + c] = static_cast<uint8_t>(u * 255.0f + 0.5f);
            }
        }
    }
    std::ofstream f(path, std::ios::binary);
    if (!f) return false;
    f << "P6\n" << W << " " << H << "\n255\n";
    f.write(reinterpret_cast<const char*>(bytes.data()), bytes.size());
    return true;
}

int main(int argc, char** argv) {
    // CLI: generate [--tokens <file>] [--embeds <file>] [--out <path>]
    //   --tokens : feed token IDs to our Qwen3 encoder
    //   --embeds : skip our encoder; load a precomputed [seq, 12288] BF16 tensor
    //              (produced by tools/diffusers_prompt_embeds.py) and use it
    //              directly as text conditioning
    std::string out_path = "out.ppm";
    std::string tokens_path;
    std::string embeds_path;
    std::string decode_latent_path;   // isolation: decode a ground-truth latent
    uint32_t seed = 0xCAFEBABEu;
    int n_steps = 4;
    f2k::cuda::Precision precision = f2k::cuda::Precision::NVFP4;
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i];
        if      (a == "--tokens" && i + 1 < argc) tokens_path = argv[++i];
        else if (a == "--embeds" && i + 1 < argc) embeds_path = argv[++i];
        else if (a == "--out"    && i + 1 < argc) out_path    = argv[++i];
        else if (a == "--seed"   && i + 1 < argc) seed = static_cast<uint32_t>(std::stoul(argv[++i]));
        else if (a == "--steps"  && i + 1 < argc) n_steps = std::stoi(argv[++i]);
        else if (a == "--decode_latent" && i + 1 < argc) decode_latent_path = argv[++i];
        else if (a == "--precision" && i + 1 < argc) {
            std::string p = argv[++i];
            precision = (p == "fp8" || p == "mxfp8")
                        ? f2k::cuda::Precision::MXFP8 : f2k::cuda::Precision::NVFP4;
        }
        else if (a.size() && a[0] != '-')         out_path    = a;
    }

    const char* home = std::getenv("HOME");
    // FP8 loads the pre-quantized MXFP8 transformer (fast disk-copy at ctor);
    // NVFP4 loads the pre-quantized NVFP4 transformer.
    const bool use_fp8 = (precision == f2k::cuda::Precision::MXFP8);
    const char* tf_dir = use_fp8 ? "transformer_mxfp8" : "transformer_f2k";
    const fs::path s1 = fs::path(home) / "models/flux2-klein-9B" / tf_dir / "shard-00001.f2k1";
    const fs::path s2 = fs::path(home) / "models/flux2-klein-9B" / tf_dir / "shard-00002.f2k1";
    const fs::path vae_path = fs::path(home) / "models/flux2-klein-9B/vae_f2k/vae.f2k1";
    for (const auto& p : {s1, s2, vae_path}) {
        if (!fs::exists(p)) { std::fprintf(stderr, "missing: %s\n", p.c_str()); return 1; }
    }

    // ============================================================
    // 1. Load transformer + VAE
    // ============================================================
    f2k::F2KModelLoader ld;
    if (!ld.add_shard(s1.string()) || !ld.add_shard(s2.string())) {
        std::fprintf(stderr, "loader: %s\n", ld.last_error().c_str()); return 1;
    }
    f2k::TensorRouter router(ld);
    if (!router.build()) { std::fprintf(stderr, "router: %s\n", router.last_error().c_str()); return 1; }
    std::printf("Loaded transformer (%zu tensors)\n", ld.total_tensors());

    f2k::F2KReader vae_r;
    if (!vae_r.open(vae_path.string())) { std::fprintf(stderr, "open vae failed\n"); return 1; }
    std::printf("Loaded VAE       (%zu tensors)\n", vae_r.names().size());

    // Qwen3 encoder — optional; only built if --tokens is supplied.
    std::unique_ptr<f2k::F2KModelLoader> qwen_ld;
    std::unique_ptr<f2k::cuda::QwenEncoder> qwen;
    bool use_qwen = !tokens_path.empty();
    if (use_qwen) {
        qwen_ld = std::make_unique<f2k::F2KModelLoader>();
        for (int i = 1; i <= 4; ++i) {
            char p[256];
            std::snprintf(p, sizeof(p), "%s/models/flux2-klein-9B/qwen3_f2k/shard-%05d.f2k1", home, i);
            if (!qwen_ld->add_shard(p)) {
                std::fprintf(stderr, "qwen loader: %s\n", qwen_ld->last_error().c_str()); return 1;
            }
        }
        std::printf("Loaded Qwen3      (%zu tensors)\n", qwen_ld->total_tensors());
    }

    // ============================================================
    // 2. Configure shapes
    // ============================================================
    // 32×32 latent → 16×16 patch grid (p=2) → seq_img=256 → 256×256 image.
    constexpr int H_LAT = 32, W_LAT = 32, PATCH = 2;
    constexpr int SEQ_IMG = (H_LAT / PATCH) * (W_LAT / PATCH);    // 256
    constexpr int SEQ_TXT = 512;  // diffusers Flux2 max_sequence_length default
    constexpr int IN_CH   = PATCH * PATCH * 32;                   // 128
    constexpr int T5_DIM  = 12288;
    constexpr int TIME_DIM = 256;
    constexpr int N_HEADS = 32, HEAD_DIM = 128;
    constexpr int FFN_DIM = 12288;
    constexpr int N_DOUBLE = 8, N_SINGLE = 24;
    const int N_STEPS = n_steps;

    f2k::cuda::FluxTransformer::Config tcfg{};
    tcfg.batch = 1; tcfg.seq_img = SEQ_IMG; tcfg.seq_txt = SEQ_TXT;
    tcfg.in_channels = IN_CH; tcfg.t5_dim = T5_DIM; tcfg.time_dim = TIME_DIM;
    tcfg.n_heads = N_HEADS; tcfg.head_dim = HEAD_DIM; tcfg.ffn_dim = FFN_DIM;
    tcfg.num_double_blocks = N_DOUBLE; tcfg.num_single_blocks = N_SINGLE;
    tcfg.rope_theta = 2000.0f; tcfg.router = &router;
    tcfg.precision = precision;
    std::printf("Precision:     %s\n", use_fp8 ? "MXFP8" : "NVFP4");

    auto t_build0 = std::chrono::steady_clock::now();
    f2k::cuda::FluxTransformer model(tcfg);
    if (!model.ok()) { std::fprintf(stderr, "transformer ctor: %s\n", model.last_error()); return 1; }
    auto t_build1 = std::chrono::steady_clock::now();
    std::printf("FluxTransformer ready in %.2fs (workspace=%.1f MiB)\n",
                std::chrono::duration<double>(t_build1 - t_build0).count(),
                model.workspace_size_bytes() / 1048576.0);

    f2k::cuda::VAEDecoder::Config vcfg{};
    vcfg.N = 1; vcfg.H_lat = H_LAT; vcfg.W_lat = W_LAT;
    vcfg.prefix = "decoder";
    vcfg.reader = &vae_r;
    auto t_vae0 = std::chrono::steady_clock::now();
    f2k::cuda::VAEDecoder vae(vcfg);
    if (!vae.ok()) { std::fprintf(stderr, "vae ctor: %s\n", vae.last_error()); return 1; }
    auto t_vae1 = std::chrono::steady_clock::now();
    std::printf("VAEDecoder       ready in %.2fs (workspace=%.1f MiB)\n",
                std::chrono::duration<double>(t_vae1 - t_vae0).count(),
                vae.workspace_size_bytes() / 1048576.0);

    if (use_qwen) {
        f2k::cuda::QwenEncoder::Config qcfg{};
        qcfg.seq = SEQ_TXT;
        qcfg.loader = qwen_ld.get();
        // diffusers Flux2 uses hidden_states[9,18,27]; HF's hidden_states[0]
        // is the embedding, so hidden_states[k] = output of layer k-1.
        qcfg.capture_layers = { 8, 17, 26 };
        auto t_q0 = std::chrono::steady_clock::now();
        qwen = std::make_unique<f2k::cuda::QwenEncoder>(qcfg);
        if (!qwen->ok()) { std::fprintf(stderr, "qwen ctor: %s\n", qwen->last_error()); return 1; }
        auto t_q1 = std::chrono::steady_clock::now();
        std::printf("Qwen3Encoder     ready in %.2fs (workspace=%.1f MiB; capture={9,18,27})\n",
                    std::chrono::duration<double>(t_q1 - t_q0).count(),
                    qwen->workspace_size_bytes() / 1048576.0);
    }

    // ============================================================
    // 3. Allocate device buffers
    // ============================================================
    const size_t latent_elems = (size_t)1 * 32 * H_LAT * W_LAT;
    const size_t token_elems  = (size_t)1 * SEQ_IMG * IN_CH;
    const size_t txt_elems    = (size_t)1 * SEQ_TXT * T5_DIM;
    const size_t pixel_elems  = (size_t)1 * 3 * vae.output_H() * vae.output_W();

    void *d_latent, *d_tokens, *d_velocity, *d_txt, *d_temb, *d_pixels;
    void *d_t_ws, *d_v_ws, *d_q_ws = nullptr;
    int32_t *d_token_ids = nullptr;
    cudaMalloc(&d_latent,   latent_elems * 2);
    cudaMalloc(&d_tokens,   token_elems  * 2);
    cudaMalloc(&d_velocity, token_elems  * 2);
    cudaMalloc(&d_txt,      txt_elems    * 2);
    cudaMalloc(&d_temb,     TIME_DIM     * 2);
    cudaMalloc(&d_pixels,   pixel_elems  * 2);
    cudaMalloc(&d_t_ws,     model.workspace_size_bytes());
    cudaMalloc(&d_v_ws,     vae.workspace_size_bytes());
    if (use_qwen) {
        cudaMalloc(&d_token_ids, SEQ_TXT * sizeof(int32_t));
        cudaMalloc(&d_q_ws, qwen->workspace_size_bytes());
    }

    // ============================================================
    // 4. Initial random latent + text conditioning
    // ============================================================
    std::mt19937 rng(seed);
    std::normal_distribution<float> dn(0.0f, 1.0f);   // N(0,1) — what the model was distilled on
    std::printf("Seed: 0x%08x  (N(0,1) latent noise)\n", seed);

    std::vector<__nv_bfloat16> h_latent(latent_elems);
    for (auto& v : h_latent) v = fp32_to_bf16(dn(rng));
    cudaMemcpy(d_latent, h_latent.data(), latent_elems * 2, cudaMemcpyHostToDevice);

  if (!decode_latent_path.empty()) {
    // -------- VAE isolation mode: decode a ground-truth [32,H_LAT,W_LAT] latent --------
    std::ifstream f(decode_latent_path, std::ios::binary);
    if (!f) { std::fprintf(stderr, "open decode_latent %s failed\n", decode_latent_path.c_str()); return 1; }
    int32_t chw[3]; f.read(reinterpret_cast<char*>(chw), 3 * sizeof(int32_t));
    if (chw[0] != 32 || chw[1] != H_LAT || chw[2] != W_LAT) {
        std::fprintf(stderr, "decode_latent shape (%d,%d,%d) != expected (32,%d,%d)\n",
                     chw[0], chw[1], chw[2], H_LAT, W_LAT); return 1;
    }
    std::vector<float> lf(latent_elems);
    f.read(reinterpret_cast<char*>(lf.data()), latent_elems * sizeof(float));
    if (!f) { std::fprintf(stderr, "truncated decode_latent\n"); return 1; }
    for (size_t i = 0; i < latent_elems; ++i) h_latent[i] = fp32_to_bf16(lf[i]);
    cudaMemcpy(d_latent, h_latent.data(), latent_elems * 2, cudaMemcpyHostToDevice);
    std::printf("Decode-latent mode: loaded %s [32,%d,%d], skipping transformer\n",
                decode_latent_path.c_str(), H_LAT, W_LAT);
  } else {
    if (!embeds_path.empty()) {
        // Load precomputed [seq, 12288] BF16 tensor directly.
        std::ifstream f(embeds_path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "open embeds %s failed\n", embeds_path.c_str()); return 1; }
        int32_t hdr[2]; f.read(reinterpret_cast<char*>(hdr), 2 * sizeof(int32_t));
        if (hdr[0] != SEQ_TXT || hdr[1] != T5_DIM) {
            std::fprintf(stderr, "embeds shape (%d,%d) != expected (%d,%d)\n",
                         hdr[0], hdr[1], SEQ_TXT, T5_DIM);
            return 1;
        }
        std::vector<__nv_bfloat16> h_emb(txt_elems);
        f.read(reinterpret_cast<char*>(h_emb.data()), txt_elems * sizeof(__nv_bfloat16));
        if (!f) { std::fprintf(stderr, "truncated embeds\n"); return 1; }
        cudaMemcpy(d_txt, h_emb.data(), txt_elems * sizeof(__nv_bfloat16), cudaMemcpyHostToDevice);
        std::printf("Text embeds:   loaded from %s (%d × %d)\n",
                    embeds_path.c_str(), hdr[0], hdr[1]);
    } else if (use_qwen) {
        // Read tokens: int32 seq_len; int32[seq_len] ids
        std::ifstream f(tokens_path, std::ios::binary);
        if (!f) { std::fprintf(stderr, "open tokens %s failed\n", tokens_path.c_str()); return 1; }
        int32_t hdr; f.read(reinterpret_cast<char*>(&hdr), sizeof(int32_t));
        if (hdr != SEQ_TXT) {
            std::fprintf(stderr, "token file seq=%d != expected %d\n", hdr, SEQ_TXT);
            return 1;
        }
        std::vector<int32_t> h_ids(SEQ_TXT);
        f.read(reinterpret_cast<char*>(h_ids.data()), SEQ_TXT * sizeof(int32_t));
        if (!f) { std::fprintf(stderr, "truncated token file\n"); return 1; }
        cudaMemcpy(d_token_ids, h_ids.data(), SEQ_TXT * sizeof(int32_t), cudaMemcpyHostToDevice);

        auto t_te0 = std::chrono::steady_clock::now();
        if (!qwen->forward(d_token_ids, d_txt, d_q_ws, qwen->workspace_size_bytes())) {
            std::fprintf(stderr, "qwen forward: %s\n", qwen->last_error()); return 1;
        }
        cudaDeviceSynchronize();
        auto t_te1 = std::chrono::steady_clock::now();
        std::printf("Text encode:   %.2f ms  (tokens=%s)\n",
                    std::chrono::duration<double, std::milli>(t_te1 - t_te0).count(),
                    tokens_path.c_str());
    } else {
        // Fallback: random text buffer (produces noise-like images).
        std::vector<__nv_bfloat16> h_txt(txt_elems);
        for (auto& v : h_txt) v = fp32_to_bf16(dn(rng) * 0.5f);
        cudaMemcpy(d_txt, h_txt.data(), txt_elems * 2, cudaMemcpyHostToDevice);
        std::printf("Text encode:   SKIPPED (no --tokens; using random)\n");
    }

    // Patchify the initial latent into tokens that the transformer expects.
    if (!f2k::cuda::patchify_bf16(d_latent, d_tokens, 1, 32, H_LAT, W_LAT, PATCH)) {
        std::fprintf(stderr, "patchify failed\n"); return 1;
    }

    // ============================================================
    // 5. Flow-matching denoise loop
    // ============================================================
    f2k::cuda::FlowMatchScheduler sched =
        f2k::cuda::FlowMatchScheduler::flux2_dynamic(N_STEPS, SEQ_IMG);
    std::printf("Denoising %d steps (Flux2 dynamic-shift) ...\n", N_STEPS);
    for (int i = 0; i <= N_STEPS; ++i) std::printf("  t[%d]=%.4f\n", i, sched.t(i));
    const auto t_loop0 = std::chrono::steady_clock::now();
    for (int i = 0; i < N_STEPS; ++i) {
        const float t  = sched.t(i);
        const float dt = sched.dt(i);
        const auto h_t = f2k::cuda::compute_timestep_embedding(t * 1000.0f, TIME_DIM);
        cudaMemcpy(d_temb, h_t.data(), TIME_DIM * 2, cudaMemcpyHostToDevice);

        if (!model.forward(d_tokens, d_txt, d_temb, d_velocity,
                           d_t_ws, model.workspace_size_bytes())) {
            std::fprintf(stderr, "step %d: %s\n", i, model.last_error()); return 1;
        }
        if (!f2k::cuda::axpy_bf16(d_tokens, d_velocity, dt, token_elems)) {
            std::fprintf(stderr, "step %d axpy\n", i); return 1;
        }
        cudaDeviceSynchronize();
        std::printf("  step %d  t=%.3f dt=%+.3f\n", i, t, dt);
    }
    const auto t_loop1 = std::chrono::steady_clock::now();
    std::printf("Denoise loop: %.2f s\n", std::chrono::duration<double>(t_loop1 - t_loop0).count());

    // ============================================================
    // 6. VAE latent de-normalization (inverse of the diffusers `vae.bn`).
    //    The transformer denoises in the NORMALIZED patch-latent space, where
    //    the clean latent is ~N(0,1). Before decoding, diffusers undoes the
    //    BatchNorm it applied at encode time, on the 128-channel patchified
    //    latent: x = x * sqrt(running_var + eps) + running_mean (see
    //    pipeline_flux2_klein.py: latents * latents_bn_std + latents_bn_mean,
    //    applied BEFORE _unpatchify_latents). The 128 token channels are exactly
    //    bn's channels (c*p*p + ph*p + pw), so we apply the affine per-column
    //    across all SEQ_IMG rows, then unpatchify. Skipping this leaves the 4
    //    sub-pixels of every patch mis-scaled → a per-patch grid artifact.
    {
        const f2k::TensorView* bn_mean = vae_r.find("bn.running_mean");
        const f2k::TensorView* bn_var  = vae_r.find("bn.running_var");
        if (!bn_mean || !bn_var) {
            std::fprintf(stderr, "VAE bn.running_{mean,var} not found\n"); return 1;
        }
        constexpr float bn_eps = 1e-4f;   // vae config batch_norm_eps
        const auto* mean_bf = reinterpret_cast<const __nv_bfloat16*>(bn_mean->data);
        const auto* var_bf  = reinterpret_cast<const __nv_bfloat16*>(bn_var->data);
        std::vector<float> mean(IN_CH), std_(IN_CH);
        for (int c = 0; c < IN_CH; ++c) {
            mean[c] = bf16_to_fp32(mean_bf[c]);
            std_[c] = std::sqrt(bf16_to_fp32(var_bf[c]) + bn_eps);
        }
        std::vector<__nv_bfloat16> h_tok(token_elems);
        cudaMemcpy(h_tok.data(), d_tokens, token_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyDeviceToHost);
        for (int r = 0; r < SEQ_IMG; ++r)
            for (int c = 0; c < IN_CH; ++c) {
                const size_t i = (size_t)r * IN_CH + c;
                h_tok[i] = fp32_to_bf16(bf16_to_fp32(h_tok[i]) * std_[c] + mean[c]);
            }
        cudaMemcpy(d_tokens, h_tok.data(), token_elems * sizeof(__nv_bfloat16),
                   cudaMemcpyHostToDevice);
    }

    // ============================================================
    // 6b. Unpatchify back into [1, 32, H_LAT, W_LAT]
    // ============================================================
    if (!f2k::cuda::unpatchify_bf16(d_tokens, d_latent, 1, 32, H_LAT, W_LAT, PATCH)) {
        std::fprintf(stderr, "unpatchify failed\n"); return 1;
    }
  } // end transformer path (skipped in --decode_latent mode)

    // ============================================================
    // 7. VAE decode → pixels  (VAEDecoder now applies post_quant_conv itself)
    // ============================================================
    const auto t_dec0 = std::chrono::steady_clock::now();
    if (!vae.forward(d_latent, d_pixels, d_v_ws, vae.workspace_size_bytes())) {
        std::fprintf(stderr, "vae: %s\n", vae.last_error()); return 1;
    }
    cudaDeviceSynchronize();
    const auto t_dec1 = std::chrono::steady_clock::now();
    std::printf("VAE decode:   %.2f ms\n",
                std::chrono::duration<double, std::milli>(t_dec1 - t_dec0).count());

    // ============================================================
    // 8. Copy back, convert, write PPM
    // ============================================================
    std::vector<__nv_bfloat16> h_pixels(pixel_elems);
    cudaMemcpy(h_pixels.data(), d_pixels, pixel_elems * 2, cudaMemcpyDeviceToHost);

    std::vector<float> h_pixels_f(pixel_elems);
    float mn = 1e9f, mx = -1e9f;
    int nans = 0;
    for (size_t i = 0; i < pixel_elems; ++i) {
        const float v = bf16_to_fp32(h_pixels[i]);
        if (!std::isfinite(v)) ++nans;
        h_pixels_f[i] = v;
        mn = std::min(mn, v); mx = std::max(mx, v);
    }
    std::printf("Pixel range: [%.3f, %.3f]  nans=%d\n", mn, mx, nans);

    if (!write_ppm(out_path, h_pixels_f.data(), 3, vae.output_H(), vae.output_W())) {
        std::fprintf(stderr, "write %s failed\n", out_path.c_str()); return 1;
    }
    std::printf("Wrote %s (%d x %d)\n", out_path.c_str(), vae.output_W(), vae.output_H());

    cudaFree(d_latent); cudaFree(d_tokens); cudaFree(d_velocity);
    cudaFree(d_txt); cudaFree(d_temb); cudaFree(d_pixels);
    cudaFree(d_t_ws); cudaFree(d_v_ws);
    if (d_token_ids) cudaFree(d_token_ids);
    if (d_q_ws)      cudaFree(d_q_ws);
    return 0;
}
