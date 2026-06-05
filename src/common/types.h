#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <string>
#include <vector>

namespace f2k {

struct Image {
    std::vector<uint8_t> rgb; // tightly packed RGB8, size = width * height * 3
    int width = 0;
    int height = 0;

    bool empty() const { return rgb.empty(); }
};

struct RefImage {
    Image image;
    float weight = 1.0f;
    bool enabled = true;
};

enum class Sampler : int {
    FlowMatch = 0,
    Euler,
    DpmppSde,
    Count
};

inline const char* sampler_name(Sampler s) {
    switch (s) {
        case Sampler::FlowMatch: return "Flow Match";
        case Sampler::Euler:     return "Euler";
        case Sampler::DpmppSde:  return "DPM++ SDE";
        default:                 return "?";
    }
}

struct GenerationRequest {
    std::string positive_prompt;
    std::string negative_prompt;

    std::vector<RefImage> refs;     // up to 4 for FLUX.2-klein

    int width  = 1024;
    int height = 1024;
    int steps  = 8;

    // FLUX-specific: distilled CFG (the native one) vs true CFG (negative-aware).
    float guidance_distilled = 3.5f;
    float guidance_true      = 1.0f; // 1.0 == off
    float sigma_shift        = 3.0f; // flow-matching schedule shift

    uint64_t seed           = 0;
    bool     randomize_seed = true;

    int     num_images = 1;
    Sampler sampler    = Sampler::FlowMatch;
};

struct ProgressEvent {
    int   step;
    int   total_steps;
    Image preview; // empty if no preview this step
};

using ProgressCallback = std::function<void(ProgressEvent)>;
using DoneCallback     = std::function<void(Image)>;
using ErrorCallback    = std::function<void(std::string)>;

} // namespace f2k
