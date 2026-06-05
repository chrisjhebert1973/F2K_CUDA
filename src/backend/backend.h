#pragma once

#include "common/types.h"

namespace f2k {

// Backend is the abstract interface the UI calls into. The first concrete
// implementation is StubBackend (returns noise); CUDA backend lands later.
//
// Threading contract: generate() returns immediately and runs asynchronously
// on a worker thread the backend owns. Callbacks fire on that worker thread,
// so the UI is responsible for marshalling back to the main thread (we do
// this with a simple thread-safe queue in App).
class Backend {
public:
    virtual ~Backend() = default;

    // Load model weights. May be slow. Returns false if loading failed.
    virtual bool load() = 0;

    // Kick off a generation. Returns immediately. Callbacks fire on the
    // backend worker thread.
    virtual void generate(const GenerationRequest& req,
                          ProgressCallback         on_progress,
                          DoneCallback             on_done,
                          ErrorCallback            on_error) = 0;

    // Request cancellation of any in-flight generate(). Best-effort.
    virtual void cancel() = 0;

    // True if a generate() is currently running.
    virtual bool busy() const = 0;

    virtual const char* name() const = 0;
};

} // namespace f2k
