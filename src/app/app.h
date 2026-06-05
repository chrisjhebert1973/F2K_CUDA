#pragma once

#include "backend/backend.h"
#include "common/types.h"

#include <memory>
#include <mutex>
#include <optional>
#include <queue>
#include <string>
#include <variant>
#include <vector>

// Forward declarations to keep Vulkan/GLFW headers out of this file.
struct GLFWwindow;

namespace f2k {

// Events queued from backend worker threads back to the UI thread.
struct BackendProgress { ProgressEvent ev; };
struct BackendDone     { Image image; };
struct BackendError    { std::string msg; };
using  BackendEvent    = std::variant<BackendProgress, BackendDone, BackendError>;

class App {
public:
    App();
    ~App();
    int run();

private:
    bool init_window();
    bool init_vulkan();
    void init_imgui();
    void shutdown();

    void frame();
    void draw_ui();
    void draw_params_panel();
    void draw_refs_panel();
    void draw_result_panel();
    void draw_log_panel();

    void start_generation();
    void drain_backend_events();

    // ---- backend ----
    std::unique_ptr<Backend> backend_;
    GenerationRequest        req_;
    std::optional<Image>     last_result_;
    int                      progress_step_  = 0;
    int                      progress_total_ = 0;
    Image                    last_preview_;
    std::mutex               event_mtx_;
    std::queue<BackendEvent> events_;

    // ---- window ----
    GLFWwindow* window_ = nullptr;
    int         fb_width_ = 0;
    int         fb_height_ = 0;
};

} // namespace f2k
