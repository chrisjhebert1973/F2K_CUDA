#include "app/app.h"

#include "backend/stub_backend.h"
#include "common/log.h"

// Vulkan must come before GLFW so glfwCreateWindowSurface is visible.
#include <vulkan/vulkan.h>
#include <GLFW/glfw3.h>

#include <imgui.h>
#include <backends/imgui_impl_glfw.h>
#include <backends/imgui_impl_vulkan.h>

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>
#include <vector>

// ---------------------------------------------------------------------------
// Vulkan globals. Keeping these file-scoped avoids leaking Vulkan headers
// through app.h and matches the structure of ImGui's reference example.
// ---------------------------------------------------------------------------
namespace {

VkAllocationCallbacks*   g_alloc       = nullptr;
VkInstance               g_instance    = VK_NULL_HANDLE;
VkPhysicalDevice         g_phys        = VK_NULL_HANDLE;
VkDevice                 g_device      = VK_NULL_HANDLE;
uint32_t                 g_queue_family = (uint32_t)-1;
VkQueue                  g_queue       = VK_NULL_HANDLE;
VkDescriptorPool         g_desc_pool   = VK_NULL_HANDLE;
ImGui_ImplVulkanH_Window g_window_data{};
uint32_t                 g_min_image_count = 2;
bool                     g_swapchain_rebuild = false;

#define VK_CHECK(x)                                                          \
    do {                                                                     \
        VkResult _vk_err = (x);                                              \
        if (_vk_err < 0) {                                                   \
            std::fprintf(stderr, "vulkan error %d at %s:%d (%s)\n",          \
                         (int)_vk_err, __FILE__, __LINE__, #x);              \
            std::abort();                                                    \
        }                                                                    \
    } while (0)

static void glfw_error(int code, const char* msg) {
    std::fprintf(stderr, "GLFW error %d: %s\n", code, msg);
}

static bool is_ext_available(const std::vector<VkExtensionProperties>& props,
                             const char* name) {
    for (const auto& p : props)
        if (std::strcmp(p.extensionName, name) == 0) return true;
    return false;
}

static void setup_vulkan(std::vector<const char*> instance_extensions) {
    // Instance
    {
        VkInstanceCreateInfo create{};
        create.sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO;

        uint32_t props_count = 0;
        vkEnumerateInstanceExtensionProperties(nullptr, &props_count, nullptr);
        std::vector<VkExtensionProperties> props(props_count);
        vkEnumerateInstanceExtensionProperties(nullptr, &props_count, props.data());

        if (is_ext_available(props, VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME))
            instance_extensions.push_back(VK_KHR_GET_PHYSICAL_DEVICE_PROPERTIES_2_EXTENSION_NAME);

        create.enabledExtensionCount   = (uint32_t)instance_extensions.size();
        create.ppEnabledExtensionNames = instance_extensions.data();
        VK_CHECK(vkCreateInstance(&create, g_alloc, &g_instance));
    }

    // Pick physical device — prefer discrete, else first.
    {
        uint32_t count = 0;
        VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &count, nullptr));
        if (count == 0) {
            std::fprintf(stderr, "no vulkan devices found\n");
            std::abort();
        }
        std::vector<VkPhysicalDevice> devs(count);
        VK_CHECK(vkEnumeratePhysicalDevices(g_instance, &count, devs.data()));
        g_phys = devs[0];
        for (auto d : devs) {
            VkPhysicalDeviceProperties pr;
            vkGetPhysicalDeviceProperties(d, &pr);
            if (pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_DISCRETE_GPU ||
                pr.deviceType == VK_PHYSICAL_DEVICE_TYPE_INTEGRATED_GPU) {
                g_phys = d;
                break;
            }
        }
        VkPhysicalDeviceProperties pr;
        vkGetPhysicalDeviceProperties(g_phys, &pr);
        f2k::log_info(std::string("vulkan device: ") + pr.deviceName);
    }

    // Queue family with graphics
    {
        uint32_t count = 0;
        vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &count, nullptr);
        std::vector<VkQueueFamilyProperties> qf(count);
        vkGetPhysicalDeviceQueueFamilyProperties(g_phys, &count, qf.data());
        for (uint32_t i = 0; i < count; ++i) {
            if (qf[i].queueFlags & VK_QUEUE_GRAPHICS_BIT) {
                g_queue_family = i;
                break;
            }
        }
        if (g_queue_family == (uint32_t)-1) {
            std::fprintf(stderr, "no graphics queue family\n");
            std::abort();
        }
    }

    // Logical device
    {
        std::vector<const char*> device_exts = { VK_KHR_SWAPCHAIN_EXTENSION_NAME };

        const float prio = 1.0f;
        VkDeviceQueueCreateInfo qci{};
        qci.sType            = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO;
        qci.queueFamilyIndex = g_queue_family;
        qci.queueCount       = 1;
        qci.pQueuePriorities = &prio;

        VkDeviceCreateInfo dci{};
        dci.sType                   = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO;
        dci.queueCreateInfoCount    = 1;
        dci.pQueueCreateInfos       = &qci;
        dci.enabledExtensionCount   = (uint32_t)device_exts.size();
        dci.ppEnabledExtensionNames = device_exts.data();
        VK_CHECK(vkCreateDevice(g_phys, &dci, g_alloc, &g_device));
        vkGetDeviceQueue(g_device, g_queue_family, 0, &g_queue);
    }

    // Descriptor pool (sized generously; ImGui's example sizes it small,
    // but we want room for backend-owned image descriptors later).
    {
        const VkDescriptorPoolSize pool_sizes[] = {
            { VK_DESCRIPTOR_TYPE_COMBINED_IMAGE_SAMPLER, 256 },
        };
        VkDescriptorPoolCreateInfo dpci{};
        dpci.sType         = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO;
        dpci.flags         = VK_DESCRIPTOR_POOL_CREATE_FREE_DESCRIPTOR_SET_BIT;
        dpci.maxSets       = 256;
        dpci.poolSizeCount = (uint32_t)std::size(pool_sizes);
        dpci.pPoolSizes    = pool_sizes;
        VK_CHECK(vkCreateDescriptorPool(g_device, &dpci, g_alloc, &g_desc_pool));
    }
}

static void setup_vulkan_window(ImGui_ImplVulkanH_Window* wd,
                                VkSurfaceKHR surface, int w, int h) {
    wd->Surface = surface;

    VkBool32 supported = VK_FALSE;
    vkGetPhysicalDeviceSurfaceSupportKHR(g_phys, g_queue_family, surface, &supported);
    if (!supported) {
        std::fprintf(stderr, "selected queue family doesn't support presentation\n");
        std::abort();
    }

    const VkFormat req_formats[] = {
        VK_FORMAT_B8G8R8A8_UNORM,
        VK_FORMAT_R8G8B8A8_UNORM,
        VK_FORMAT_B8G8R8_UNORM,
        VK_FORMAT_R8G8B8_UNORM,
    };
    const VkColorSpaceKHR req_color = VK_COLOR_SPACE_SRGB_NONLINEAR_KHR;
    wd->SurfaceFormat = ImGui_ImplVulkanH_SelectSurfaceFormat(
        g_phys, surface, req_formats, (size_t)std::size(req_formats), req_color);

    VkPresentModeKHR present_modes[] = { VK_PRESENT_MODE_FIFO_KHR };
    wd->PresentMode = ImGui_ImplVulkanH_SelectPresentMode(
        g_phys, surface, present_modes, (size_t)std::size(present_modes));

    ImGui_ImplVulkanH_CreateOrResizeWindow(
        g_instance, g_phys, g_device, wd, g_queue_family, g_alloc,
        w, h, g_min_image_count, VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
}

static void cleanup_vulkan_window() {
    ImGui_ImplVulkanH_DestroyWindow(g_instance, g_device, &g_window_data, g_alloc);
}

static void cleanup_vulkan() {
    vkDestroyDescriptorPool(g_device, g_desc_pool, g_alloc);
    vkDestroyDevice(g_device, g_alloc);
    vkDestroyInstance(g_instance, g_alloc);
}

static void frame_render(ImGui_ImplVulkanH_Window* wd, ImDrawData* dd) {
    VkSemaphore image_acquired = wd->FrameSemaphores[wd->SemaphoreIndex].ImageAcquiredSemaphore;
    VkSemaphore render_complete = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;

    VkResult err = vkAcquireNextImageKHR(g_device, wd->Swapchain, UINT64_MAX,
                                         image_acquired, VK_NULL_HANDLE,
                                         &wd->FrameIndex);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_swapchain_rebuild = true;
        return;
    }
    VK_CHECK(err);

    auto* fd = &wd->Frames[wd->FrameIndex];
    VK_CHECK(vkWaitForFences(g_device, 1, &fd->Fence, VK_TRUE, UINT64_MAX));
    VK_CHECK(vkResetFences(g_device, 1, &fd->Fence));

    VK_CHECK(vkResetCommandPool(g_device, fd->CommandPool, 0));
    VkCommandBufferBeginInfo cbbi{};
    cbbi.sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO;
    cbbi.flags = VK_COMMAND_BUFFER_USAGE_ONE_TIME_SUBMIT_BIT;
    VK_CHECK(vkBeginCommandBuffer(fd->CommandBuffer, &cbbi));

    VkRenderPassBeginInfo rpbi{};
    rpbi.sType                    = VK_STRUCTURE_TYPE_RENDER_PASS_BEGIN_INFO;
    rpbi.renderPass               = wd->RenderPass;
    rpbi.framebuffer              = fd->Framebuffer;
    rpbi.renderArea.extent.width  = (uint32_t)wd->Width;
    rpbi.renderArea.extent.height = (uint32_t)wd->Height;
    rpbi.clearValueCount          = 1;
    rpbi.pClearValues             = &wd->ClearValue;
    vkCmdBeginRenderPass(fd->CommandBuffer, &rpbi, VK_SUBPASS_CONTENTS_INLINE);

    ImGui_ImplVulkan_RenderDrawData(dd, fd->CommandBuffer);

    vkCmdEndRenderPass(fd->CommandBuffer);
    VK_CHECK(vkEndCommandBuffer(fd->CommandBuffer));

    VkPipelineStageFlags wait_stage = VK_PIPELINE_STAGE_COLOR_ATTACHMENT_OUTPUT_BIT;
    VkSubmitInfo si{};
    si.sType                = VK_STRUCTURE_TYPE_SUBMIT_INFO;
    si.waitSemaphoreCount   = 1;
    si.pWaitSemaphores      = &image_acquired;
    si.pWaitDstStageMask    = &wait_stage;
    si.commandBufferCount   = 1;
    si.pCommandBuffers      = &fd->CommandBuffer;
    si.signalSemaphoreCount = 1;
    si.pSignalSemaphores    = &render_complete;
    VK_CHECK(vkQueueSubmit(g_queue, 1, &si, fd->Fence));
}

static void frame_present(ImGui_ImplVulkanH_Window* wd) {
    if (g_swapchain_rebuild) return;
    VkSemaphore render_complete = wd->FrameSemaphores[wd->SemaphoreIndex].RenderCompleteSemaphore;
    VkPresentInfoKHR pi{};
    pi.sType              = VK_STRUCTURE_TYPE_PRESENT_INFO_KHR;
    pi.waitSemaphoreCount = 1;
    pi.pWaitSemaphores    = &render_complete;
    pi.swapchainCount     = 1;
    pi.pSwapchains        = &wd->Swapchain;
    pi.pImageIndices      = &wd->FrameIndex;
    VkResult err = vkQueuePresentKHR(g_queue, &pi);
    if (err == VK_ERROR_OUT_OF_DATE_KHR || err == VK_SUBOPTIMAL_KHR) {
        g_swapchain_rebuild = true;
        return;
    }
    VK_CHECK(err);
    wd->SemaphoreIndex = (wd->SemaphoreIndex + 1) % wd->SemaphoreCount;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// App
// ---------------------------------------------------------------------------
namespace f2k {

App::App() : backend_(std::make_unique<StubBackend>()) {
    req_.positive_prompt = "a tiny robot painting a Vermeer in a sunlit studio";
    req_.refs.resize(4);
}

App::~App() = default;

bool App::init_window() {
    glfwSetErrorCallback(glfw_error);
    if (!glfwInit()) {
        std::fprintf(stderr, "glfwInit failed\n");
        return false;
    }
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);
    window_ = glfwCreateWindow(1600, 1000, "F2KTest — FLUX.2-klein on Spark",
                               nullptr, nullptr);
    if (!window_) {
        std::fprintf(stderr, "glfwCreateWindow failed (no display? try DISPLAY=:0 or run under wayland)\n");
        return false;
    }
    return true;
}

bool App::init_vulkan() {
    if (!glfwVulkanSupported()) {
        std::fprintf(stderr, "GLFW reports vulkan not supported\n");
        return false;
    }
    uint32_t ext_count = 0;
    const char** glfw_exts = glfwGetRequiredInstanceExtensions(&ext_count);
    std::vector<const char*> instance_exts(glfw_exts, glfw_exts + ext_count);
    setup_vulkan(std::move(instance_exts));

    VkSurfaceKHR surface;
    VK_CHECK(glfwCreateWindowSurface(g_instance, window_, g_alloc, &surface));

    glfwGetFramebufferSize(window_, &fb_width_, &fb_height_);
    setup_vulkan_window(&g_window_data, surface, fb_width_, fb_height_);
    return true;
}

void App::init_imgui() {
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO();
    io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;
    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForVulkan(window_, true);
    ImGui_ImplVulkan_InitInfo init{};
    init.Instance        = g_instance;
    init.PhysicalDevice  = g_phys;
    init.Device          = g_device;
    init.QueueFamily     = g_queue_family;
    init.Queue           = g_queue;
    init.DescriptorPool  = g_desc_pool;
    init.MinImageCount   = g_min_image_count;
    init.ImageCount      = g_window_data.ImageCount;
    init.Allocator       = g_alloc;
    init.PipelineInfoMain.RenderPass  = g_window_data.RenderPass;
    init.PipelineInfoMain.Subpass     = 0;
    init.PipelineInfoMain.MSAASamples = VK_SAMPLE_COUNT_1_BIT;
    ImGui_ImplVulkan_Init(&init);
}

void App::shutdown() {
    if (g_device != VK_NULL_HANDLE) vkDeviceWaitIdle(g_device);
    ImGui_ImplVulkan_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();
    cleanup_vulkan_window();
    cleanup_vulkan();
    if (window_) {
        glfwDestroyWindow(window_);
        window_ = nullptr;
    }
    glfwTerminate();
}

int App::run() {
    if (!init_window()) return 1;
    if (!init_vulkan()) return 1;
    init_imgui();
    backend_->load();

    log_info("F2KTest scaffold ready. Backend: " + std::string(backend_->name()));

    while (!glfwWindowShouldClose(window_)) {
        glfwPollEvents();
        if (glfwGetWindowAttrib(window_, GLFW_ICONIFIED)) {
            glfwWaitEventsTimeout(0.1);
            continue;
        }

        if (g_swapchain_rebuild) {
            int w, h;
            glfwGetFramebufferSize(window_, &w, &h);
            if (w > 0 && h > 0) {
                ImGui_ImplVulkan_SetMinImageCount(g_min_image_count);
                ImGui_ImplVulkanH_CreateOrResizeWindow(
                    g_instance, g_phys, g_device, &g_window_data,
                    g_queue_family, g_alloc, w, h, g_min_image_count,
                    VK_IMAGE_USAGE_COLOR_ATTACHMENT_BIT);
                g_window_data.FrameIndex = 0;
                g_swapchain_rebuild = false;
            }
        }

        drain_backend_events();

        ImGui_ImplVulkan_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        ImGui::DockSpaceOverViewport(0, ImGui::GetMainViewport());
        draw_ui();

        ImGui::Render();
        ImDrawData* dd = ImGui::GetDrawData();
        const bool minimized = (dd->DisplaySize.x <= 0.0f || dd->DisplaySize.y <= 0.0f);
        if (!minimized) {
            g_window_data.ClearValue.color.float32[0] = 0.05f;
            g_window_data.ClearValue.color.float32[1] = 0.06f;
            g_window_data.ClearValue.color.float32[2] = 0.08f;
            g_window_data.ClearValue.color.float32[3] = 1.0f;
            frame_render(&g_window_data, dd);
            frame_present(&g_window_data);
        }
    }

    shutdown();
    return 0;
}

// ---------------------------------------------------------------------------
// UI
// ---------------------------------------------------------------------------

void App::draw_ui() {
    draw_params_panel();
    draw_refs_panel();
    draw_result_panel();
    draw_log_panel();
}

void App::draw_params_panel() {
    ImGui::Begin("Generation");

    ImGui::TextDisabled("Prompt");
    {
        char buf[4096];
        std::snprintf(buf, sizeof(buf), "%s", req_.positive_prompt.c_str());
        if (ImGui::InputTextMultiline("##pos", buf, sizeof(buf), ImVec2(-1, 96)))
            req_.positive_prompt = buf;
    }

    ImGui::TextDisabled("Negative prompt (only matters if true CFG > 1)");
    {
        char buf[2048];
        std::snprintf(buf, sizeof(buf), "%s", req_.negative_prompt.c_str());
        if (ImGui::InputTextMultiline("##neg", buf, sizeof(buf), ImVec2(-1, 48)))
            req_.negative_prompt = buf;
    }

    ImGui::Separator();

    ImGui::SliderInt("Width",  &req_.width,  256, 2048);
    ImGui::SliderInt("Height", &req_.height, 256, 2048);
    ImGui::SliderInt("Steps",  &req_.steps,  1,   50);

    ImGui::SliderFloat("Guidance (distilled)", &req_.guidance_distilled, 1.0f, 10.0f, "%.2f");
    ImGui::SliderFloat("Guidance (true CFG)",  &req_.guidance_true,      1.0f, 8.0f,  "%.2f");
    ImGui::SliderFloat("Sigma shift",          &req_.sigma_shift,        0.5f, 6.0f,  "%.2f");

    ImGui::Checkbox("Randomize seed", &req_.randomize_seed);
    if (!req_.randomize_seed) {
        ImGui::SameLine();
        long long s = (long long)req_.seed;
        if (ImGui::InputScalar("seed", ImGuiDataType_S64, &s))
            req_.seed = (uint64_t)s;
    }

    ImGui::SliderInt("Images", &req_.num_images, 1, 8);

    {
        const char* current = sampler_name(req_.sampler);
        if (ImGui::BeginCombo("Sampler", current)) {
            for (int i = 0; i < (int)Sampler::Count; ++i) {
                bool sel = ((int)req_.sampler == i);
                if (ImGui::Selectable(sampler_name((Sampler)i), sel))
                    req_.sampler = (Sampler)i;
            }
            ImGui::EndCombo();
        }
    }

    ImGui::Separator();

    const bool busy = backend_->busy();
    ImGui::BeginDisabled(busy);
    if (ImGui::Button("Generate", ImVec2(-1, 32))) start_generation();
    ImGui::EndDisabled();
    if (busy) {
        ImGui::SameLine();
        if (ImGui::SmallButton("Cancel")) backend_->cancel();
    }

    if (progress_total_ > 0) {
        float frac = (float)progress_step_ / (float)progress_total_;
        ImGui::ProgressBar(frac, ImVec2(-1, 0));
        ImGui::Text("step %d / %d", progress_step_, progress_total_);
    }

    ImGui::End();
}

void App::draw_refs_panel() {
    ImGui::Begin("Reference images");
    ImGui::TextWrapped("FLUX.2-klein is an editor: up to 4 reference images can condition the generation. "
                       "Image upload is stubbed in this scaffold.");
    for (int i = 0; i < (int)req_.refs.size(); ++i) {
        ImGui::PushID(i);
        ImGui::Separator();
        ImGui::Checkbox("enabled", &req_.refs[i].enabled);
        ImGui::SameLine();
        ImGui::Text("Slot %d", i + 1);
        ImGui::SliderFloat("weight", &req_.refs[i].weight, 0.0f, 2.0f, "%.2f");
        ImGui::BeginDisabled(true);
        ImGui::Button("Load image...");
        ImGui::EndDisabled();
        ImGui::PopID();
    }
    ImGui::End();
}

void App::draw_result_panel() {
    ImGui::Begin("Result");
    if (!last_preview_.empty()) {
        ImGui::Text("Live preview (%dx%d)", last_preview_.width, last_preview_.height);
    }
    if (last_result_) {
        ImGui::Text("Final image (%dx%d)", last_result_->width, last_result_->height);
    } else {
        ImGui::TextDisabled("(no image yet — click Generate)");
    }
    // TODO: upload preview/result as a Vulkan texture via ImGui_ImplVulkan_AddTexture
    //       and render it here. Stubbed for now.
    ImGui::End();
}

void App::draw_log_panel() {
    ImGui::Begin("Log");
    if (ImGui::SmallButton("clear")) Log::instance().clear();
    ImGui::Separator();
    ImGui::BeginChild("##loglines", ImVec2(0, 0), ImGuiChildFlags_None,
                      ImGuiWindowFlags_HorizontalScrollbar);
    Log::instance().for_each([](const LogLine& l) {
        ImVec4 col;
        switch (l.level) {
            case LogLevel::Info:  col = ImVec4(0.85f, 0.85f, 0.85f, 1.0f); break;
            case LogLevel::Warn:  col = ImVec4(1.00f, 0.80f, 0.30f, 1.0f); break;
            case LogLevel::Error: col = ImVec4(1.00f, 0.35f, 0.35f, 1.0f); break;
        }
        ImGui::PushStyleColor(ImGuiCol_Text, col);
        ImGui::TextUnformatted(l.text.c_str());
        ImGui::PopStyleColor();
    });
    if (ImGui::GetScrollY() >= ImGui::GetScrollMaxY() - 4.0f)
        ImGui::SetScrollHereY(1.0f);
    ImGui::EndChild();
    ImGui::End();
}

// ---------------------------------------------------------------------------
// Backend bridge
// ---------------------------------------------------------------------------

void App::start_generation() {
    if (backend_->busy()) return;
    if (req_.randomize_seed) {
        req_.seed = (uint64_t)glfwGetTimerValue();
    }
    progress_step_  = 0;
    progress_total_ = req_.steps;
    last_preview_   = {};
    last_result_.reset();

    backend_->generate(
        req_,
        [this](ProgressEvent ev) {
            std::lock_guard l(event_mtx_);
            events_.push(BackendProgress{std::move(ev)});
        },
        [this](Image img) {
            std::lock_guard l(event_mtx_);
            events_.push(BackendDone{std::move(img)});
        },
        [this](std::string msg) {
            std::lock_guard l(event_mtx_);
            events_.push(BackendError{std::move(msg)});
        });
}

void App::drain_backend_events() {
    std::queue<BackendEvent> local;
    {
        std::lock_guard l(event_mtx_);
        std::swap(local, events_);
    }
    while (!local.empty()) {
        auto& e = local.front();
        if (auto* p = std::get_if<BackendProgress>(&e)) {
            progress_step_  = p->ev.step;
            progress_total_ = p->ev.total_steps;
            if (!p->ev.preview.empty()) last_preview_ = std::move(p->ev.preview);
        } else if (auto* d = std::get_if<BackendDone>(&e)) {
            last_result_ = std::move(d->image);
            log_info("generation done");
        } else if (auto* er = std::get_if<BackendError>(&e)) {
            log_error("backend: " + er->msg);
            progress_total_ = 0;
        }
        local.pop();
    }
}

} // namespace f2k
