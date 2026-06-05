#include "backend/stub_backend.h"

#include "common/log.h"

#include <chrono>
#include <cstdint>
#include <random>

namespace f2k {

StubBackend::StubBackend() = default;

StubBackend::~StubBackend() {
    cancel();
    if (worker_.joinable()) worker_.join();
}

bool StubBackend::load() {
    log_info("StubBackend: load() — nothing to do");
    return true;
}

void StubBackend::cancel() {
    cancel_requested_.store(true);
}

void StubBackend::generate(const GenerationRequest& req,
                           ProgressCallback         on_progress,
                           DoneCallback             on_done,
                           ErrorCallback            on_error) {
    if (busy_.exchange(true)) {
        if (on_error) on_error("StubBackend: already generating");
        return;
    }
    if (worker_.joinable()) worker_.join();
    cancel_requested_.store(false);
    worker_ = std::thread(&StubBackend::run, this,
                          req, std::move(on_progress),
                          std::move(on_done), std::move(on_error));
}

static Image seeded_noise(int w, int h, uint64_t seed) {
    Image img;
    img.width  = w;
    img.height = h;
    img.rgb.resize(static_cast<size_t>(w) * h * 3);
    std::mt19937_64 rng(seed);
    for (auto& b : img.rgb) b = static_cast<uint8_t>(rng() & 0xff);
    return img;
}

void StubBackend::run(GenerationRequest req,
                      ProgressCallback   on_progress,
                      DoneCallback       on_done,
                      ErrorCallback      on_error) {
    using namespace std::chrono_literals;
    log_info("StubBackend: starting fake generation");

    const int total = std::max(1, req.steps);
    for (int step = 1; step <= total; ++step) {
        if (cancel_requested_.load()) {
            if (on_error) on_error("Cancelled");
            busy_.store(false);
            return;
        }
        std::this_thread::sleep_for(150ms);
        if (on_progress) {
            ProgressEvent ev{step, total, {}};
            // Send a coarse preview every few steps to exercise the UI path.
            if (step % 2 == 0 || step == total) {
                ev.preview = seeded_noise(req.width / 4, req.height / 4,
                                          req.seed ^ static_cast<uint64_t>(step));
            }
            on_progress(std::move(ev));
        }
    }

    Image result = seeded_noise(req.width, req.height, req.seed);
    if (on_done) on_done(std::move(result));
    busy_.store(false);
    log_info("StubBackend: done");
}

} // namespace f2k
