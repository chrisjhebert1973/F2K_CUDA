// Sampler helpers for FLUX.2-klein flow-matching denoising:
//
//   compute_timestep_embedding(t, dim)  →  BF16 [dim] sinusoidal embedding,
//       compatible with what the model was trained on (cat(cos, sin)).
//
//   FlowMatchScheduler                  →  precomputes the t schedule
//       for an Euler integrator. v1 is a plain linear schedule (the FLUX
//       seq-length-dependent shift can land on top of this later).
//
//   axpy_bf16(y, x, alpha, n)           →  y[i] += alpha * x[i], BF16
//       in-place. Used for the per-step `latent += dt * v_pred` update.

#pragma once

#include <cuda_bf16.h>

#include <cmath>
#include <cstddef>
#include <vector>

struct CUstream_st;
using cudaStream_t = CUstream_st*;

namespace f2k::cuda {

// Sinusoidal timestep embedding. `t` is the timestep value (typically the
// raw flow-matching t in [0, 1] multiplied by 1000 — matches what FLUX.1
// reference code does), `dim` is even, `max_period` defaults to 10000.
// Returns host BF16 vector of length `dim`: cat(cos(t * freqs), sin(...)).
inline std::vector<__nv_bfloat16>
compute_timestep_embedding(float t, int dim, float max_period = 10000.0f) {
    std::vector<__nv_bfloat16> out(dim);
    const int half = dim / 2;
    const float log_max = std::log(max_period);
    for (int i = 0; i < half; ++i) {
        const float exponent = -log_max * static_cast<float>(i) / static_cast<float>(half);
        const float arg = t * std::exp(exponent);
        out[i]        = __float2bfloat16(std::cos(arg));
        out[half + i] = __float2bfloat16(std::sin(arg));
    }
    return out;
}

class FlowMatchScheduler {
public:
    explicit FlowMatchScheduler(int num_steps)
        : ts_(num_steps + 1) {
        for (int i = 0; i <= num_steps; ++i) {
            ts_[i] = 1.0f - static_cast<float>(i) / static_cast<float>(num_steps);
        }
    }

    // Flux2 dynamic-shifting schedule (matches diffusers
    // FlowMatchEulerDiscreteScheduler with use_dynamic_shifting=true and
    // time_shift_type="exponential"). Linear sigmas are remapped through:
    //   mu     = base_shift + (max_shift - base_shift) *
    //              (image_seq_len - base_seq_len) / (max_seq_len - base_seq_len)
    //   sigma' = exp(mu) / (exp(mu) + (1/sigma - 1))      (sigma > 0)
    //   sigma' = 0                                        (sigma = 0)
    static FlowMatchScheduler flux2_dynamic(int num_steps,
                                             int image_seq_len,
                                             int base_seq_len = 256,
                                             int max_seq_len  = 4096,
                                             float base_shift = 0.5f,
                                             float max_shift  = 1.15f) {
        FlowMatchScheduler s(num_steps);
        const float frac = static_cast<float>(image_seq_len - base_seq_len) /
                           static_cast<float>(max_seq_len - base_seq_len);
        const float mu  = base_shift + (max_shift - base_shift) * frac;
        const float emu = std::exp(mu);
        for (int i = 0; i <= num_steps; ++i) {
            const float sigma = s.ts_[i];
            s.ts_[i] = (sigma > 0.0f)
                          ? emu / (emu + (1.0f / sigma - 1.0f))
                          : 0.0f;
        }
        return s;
    }

    int   num_steps() const { return static_cast<int>(ts_.size()) - 1; }
    float t(int i)    const { return ts_[i]; }
    float dt(int i)   const { return ts_[i + 1] - ts_[i]; }  // negative (going 1 → 0)

private:
    std::vector<float> ts_;
};

// y[i] += alpha * x[i].
bool axpy_bf16(void* y, const void* x, float alpha, size_t n,
               cudaStream_t stream = nullptr);

} // namespace f2k::cuda
