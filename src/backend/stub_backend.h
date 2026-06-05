#pragma once

#include "backend/backend.h"

#include <atomic>
#include <thread>

namespace f2k {

// StubBackend simulates the inference pipeline by sleeping per step and
// returning a deterministic seeded noise image. Used to develop the UI
// against the real Backend interface before the CUDA backend exists.
class StubBackend final : public Backend {
public:
    StubBackend();
    ~StubBackend() override;

    bool load() override;
    void generate(const GenerationRequest& req,
                  ProgressCallback         on_progress,
                  DoneCallback             on_done,
                  ErrorCallback            on_error) override;
    void cancel() override;
    bool busy() const override { return busy_.load(); }
    const char* name() const override { return "Stub (no inference)"; }

private:
    void run(GenerationRequest req,
             ProgressCallback   on_progress,
             DoneCallback       on_done,
             ErrorCallback      on_error);

    std::thread       worker_;
    std::atomic<bool> busy_{false};
    std::atomic<bool> cancel_requested_{false};
};

} // namespace f2k
