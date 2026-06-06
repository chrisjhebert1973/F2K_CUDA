cmake_minimum_required(VERSION 3.28)

project(F2KTest
    VERSION 0.0.1
    DESCRIPTION "FLUX.2-klein on DGX Spark: custom CUDA inference + ImGui frontend"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

# CUDA is enabled lazily — the scaffold compiles without it. Once the real
# backend lands, set F2K_ENABLE_CUDA=ON and we light up the CUDA language.
option(F2K_ENABLE_CUDA "Enable CUDA + CUTLASS code paths" OFF)

if(F2K_ENABLE_CUDA)
    # MUST be set before enable_language(CUDA); otherwise CMake's auto-probe
    # writes some random fallback (Turing 75 in this case) and our default
    # never wins. 121a = Blackwell GB10 with arch-specific instructions
    # (tcgen05, block-scaled MMA) which CUTLASS's NVFP4 MMA requires.
    set(CMAKE_CUDA_ARCHITECTURES "121a" CACHE STRING
        "CUDA architectures — Blackwell GB10 / DGX Spark" FORCE)
    enable_language(CUDA)
    set(CMAKE_CUDA_STANDARD 20)
    set(CMAKE_CUDA_STANDARD_REQUIRED ON)
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --expt-relaxed-constexpr --expt-extended-lambda -lineinfo")
    # CUTLASS templates push the compile-time work hard; parallelism helps.
    set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} --threads 0")
endif()

# ---------------------------------------------------------------------------
# Dependencies
# ---------------------------------------------------------------------------

find_package(Vulkan REQUIRED)
find_package(Threads REQUIRED)

# GLFW: build only the library we need.
set(GLFW_BUILD_DOCS     OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_TESTS    OFF CACHE BOOL "" FORCE)
set(GLFW_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
set(GLFW_INSTALL        OFF CACHE BOOL "" FORCE)
add_subdirectory(third_party/glfw EXCLUDE_FROM_ALL)

# ImGui: vendored sources, compiled as a static lib.
set(IMGUI_DIR ${CMAKE_SOURCE_DIR}/third_party/imgui)
add_library(imgui STATIC
    ${IMGUI_DIR}/imgui.cpp
    ${IMGUI_DIR}/imgui_draw.cpp
    ${IMGUI_DIR}/imgui_tables.cpp
    ${IMGUI_DIR}/imgui_widgets.cpp
    ${IMGUI_DIR}/imgui_demo.cpp
    ${IMGUI_DIR}/backends/imgui_impl_glfw.cpp
    ${IMGUI_DIR}/backends/imgui_impl_vulkan.cpp
)
target_include_directories(imgui PUBLIC
    ${IMGUI_DIR}
    ${IMGUI_DIR}/backends
)
target_link_libraries(imgui PUBLIC glfw Vulkan::Vulkan)
target_compile_definitions(imgui PUBLIC
    IMGUI_IMPL_VULKAN_NO_PROTOTYPES=0
    GLFW_INCLUDE_NONE
)

# stb: header-only.
add_library(stb INTERFACE)
target_include_directories(stb INTERFACE ${CMAKE_SOURCE_DIR}/third_party/stb)

# CUTLASS: header-only. We do NOT add_subdirectory because its top-level
# CMake builds the full test/benchmark suite.
add_library(cutlass INTERFACE)
target_include_directories(cutlass INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/include
    ${CMAKE_SOURCE_DIR}/third_party/cutlass/tools/util/include
)

# nlohmann/json: header-only (single header in single_include/).
add_library(nlohmann_json INTERFACE)
target_include_directories(nlohmann_json INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/json/single_include
)

# ---------------------------------------------------------------------------
# Common library: format I/O code shared between the UI binary and tests.
# ---------------------------------------------------------------------------

add_library(f2k_common STATIC
    src/common/safetensors.cpp
    src/common/f2k_format.cpp
    src/common/f2k_model_loader.cpp
    src/common/tensor_router.cpp
)
target_include_directories(f2k_common PUBLIC src)
target_link_libraries(f2k_common PUBLIC nlohmann_json)
target_compile_options(f2k_common PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic>
)

# ---------------------------------------------------------------------------
# F2K executable
# ---------------------------------------------------------------------------

add_executable(f2k
    src/main.cpp
    src/app/app.cpp
    src/backend/stub_backend.cpp
)
target_include_directories(f2k PRIVATE src)
target_link_libraries(f2k PRIVATE
    f2k_common
    imgui
    glfw
    Vulkan::Vulkan
    stb
    Threads::Threads
)
target_compile_options(f2k PRIVATE
    $<$<CXX_COMPILER_ID:GNU,Clang>:-Wall -Wextra -Wpedantic -Wno-missing-field-initializers>
)

# When CUDA is enabled, the backend lib will pull in CUTLASS + cuBLAS + cuDNN.
if(F2K_ENABLE_CUDA)
    # cuDNN — for the VAE decoder convs and norms. The Blackwell (sm_121)
    # tensor-core conv engines only exist in cuDNN 9 and are reachable only via
    # the backend graph API (see conv2d.cu). The unversioned `libcudnn.so` dev
    # symlink points at the legacy cuDNN 8 (no sm_121 kernels — ~170x slower),
    # so we link libcudnn.so.9 explicitly. Headers in /usr/include are already
    # v9 (cudnn_version.h reports CUDNN_MAJOR 9).
    find_library(CUDNN_LIB NAMES cudnn.so.9 libcudnn.so.9 cudnn
                 HINTS /usr/lib/aarch64-linux-gnu)
    if(EXISTS /usr/lib/aarch64-linux-gnu/libcudnn.so.9)
        set(CUDNN_LIB /usr/lib/aarch64-linux-gnu/libcudnn.so.9)
    endif()
    if(NOT CUDNN_LIB)
        message(FATAL_ERROR "cuDNN not found. Install libcudnn9-cuda-13 + libcudnn9-dev-cuda-13.")
    endif()
    set(CUDNN_INCLUDE_DIRS /usr/include /usr/include/aarch64-linux-gnu)
    message(STATUS "cuDNN: ${CUDNN_LIB}")

    add_library(f2k_cuda STATIC
        src/backend/cuda/fp4_gemm.cu
        src/backend/cuda/fp8_gemm.cu
        src/backend/cuda/linear.cu
        src/backend/cuda/swiglu.cu
        src/backend/cuda/attention.cu
        src/backend/cuda/joint_attention.cu
        src/backend/cuda/mmdit_block.cu
        src/backend/cuda/mmdit_stack.cu
        src/backend/cuda/single_stream_block.cu
        src/backend/cuda/double_stream_block.cu
        src/backend/cuda/modulation_mlp.cu
        src/backend/cuda/embeddings.cu
        src/backend/cuda/flux_transformer.cu
        src/backend/cuda/sampler.cu
        src/backend/cuda/conv2d.cu
        src/backend/cuda/kernels/seq_ops.cu
        src/backend/cuda/kernels/groupnorm.cu
        src/backend/cuda/kernels/upsample.cu
        src/backend/cuda/kernels/rmsnorm.cu
        src/backend/cuda/kernels/rope.cu
        src/backend/cuda/kernels/silu_mul.cu
        src/backend/cuda/kernels/modulation.cu
        src/backend/cuda/kernels/qkv_mlp_split.cu
        src/backend/cuda/vae_resnet.cu
        src/backend/cuda/vae_attn.cu
        src/backend/cuda/vae_decoder.cu
        src/backend/cuda/kernels/transpose_chw.cu
        src/backend/cuda/kernels/patchify.cu
        src/backend/cuda/kernels/embed_lookup.cu
        src/backend/cuda/kernels/rope_4axis.cu
        src/backend/cuda/kernels/layernorm.cu
        src/backend/cuda/qwen_attention.cu
        src/backend/cuda/qwen_layer.cu
        src/backend/cuda/qwen_encoder.cu
    )
    target_include_directories(f2k_cuda PUBLIC src ${CUDNN_INCLUDE_DIRS})
    target_link_libraries(f2k_cuda PUBLIC cutlass ${CUDNN_LIB} f2k_common)
    set_target_properties(f2k_cuda PROPERTIES
        POSITION_INDEPENDENT_CODE ON
        CUDA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(f2k PRIVATE f2k_cuda)
endif()

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

enable_testing()

add_executable(test_round_trip tests/test_round_trip.cpp)
target_link_libraries(test_round_trip PRIVATE f2k_common)
add_test(NAME round_trip COMMAND test_round_trip)

add_executable(test_f2k_model_loader tests/test_f2k_model_loader.cpp)
target_link_libraries(test_f2k_model_loader PRIVATE f2k_common)
add_test(NAME f2k_model_loader COMMAND test_f2k_model_loader)

add_executable(test_tensor_router tests/test_tensor_router.cpp)
target_link_libraries(test_tensor_router PRIVATE f2k_common)
add_test(NAME tensor_router COMMAND test_tensor_router)

# Converter tool — gated on CUDA because it uses CUTLASS host-side types
# (cutlass::float_e2m1_t / float_ue4m3_t) for proper FP4/FP8 rounding.
if(F2K_ENABLE_CUDA)
    add_executable(f2k_convert tools/f2k_convert.cu)
    target_link_libraries(f2k_convert PRIVATE f2k_common cutlass)
endif()

if(F2K_ENABLE_CUDA)
    add_executable(test_fp4_gemm tests/test_fp4_gemm.cu)
    target_link_libraries(test_fp4_gemm PRIVATE f2k_cuda)
    add_test(NAME fp4_gemm COMMAND test_fp4_gemm)

    add_executable(test_fp8_gemm tests/test_fp8_gemm.cu)
    target_link_libraries(test_fp8_gemm PRIVATE f2k_cuda)
    add_test(NAME fp8_gemm COMMAND test_fp8_gemm)

    add_executable(test_linear_fp8 tests/test_linear_fp8.cu)
    target_link_libraries(test_linear_fp8 PRIVATE f2k_cuda)
    add_test(NAME linear_fp8 COMMAND test_linear_fp8)

    add_executable(test_rmsnorm tests/test_rmsnorm.cu)
    target_link_libraries(test_rmsnorm PRIVATE f2k_cuda)
    add_test(NAME rmsnorm COMMAND test_rmsnorm)

    add_executable(test_linear tests/test_linear.cu)
    target_link_libraries(test_linear PRIVATE f2k_cuda)
    add_test(NAME linear COMMAND test_linear)

    add_executable(test_linear_preq tests/test_linear_preq.cu)
    target_link_libraries(test_linear_preq PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME linear_preq COMMAND test_linear_preq)

    add_executable(test_single_stream_block tests/test_single_stream_block.cu)
    target_link_libraries(test_single_stream_block PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME single_stream_block COMMAND test_single_stream_block)

    add_executable(test_double_stream_block tests/test_double_stream_block.cu)
    target_link_libraries(test_double_stream_block PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME double_stream_block COMMAND test_double_stream_block)

    add_executable(test_modulation_mlp tests/test_modulation_mlp.cu)
    target_link_libraries(test_modulation_mlp PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME modulation_mlp COMMAND test_modulation_mlp)

    add_executable(test_embeddings tests/test_embeddings.cu)
    target_link_libraries(test_embeddings PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME embeddings COMMAND test_embeddings)

    add_executable(test_flux_transformer tests/test_flux_transformer.cu)
    target_link_libraries(test_flux_transformer PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME flux_transformer COMMAND test_flux_transformer)

    add_executable(test_denoise_loop tests/test_denoise_loop.cu)
    target_link_libraries(test_denoise_loop PRIVATE f2k_cuda f2k_common cutlass)
    add_test(NAME denoise_loop COMMAND test_denoise_loop)

    add_executable(test_conv2d tests/test_conv2d.cu)
    target_link_libraries(test_conv2d PRIVATE f2k_cuda)
    add_test(NAME conv2d COMMAND test_conv2d)

    add_executable(test_groupnorm tests/test_groupnorm.cu)
    target_link_libraries(test_groupnorm PRIVATE f2k_cuda)
    add_test(NAME groupnorm COMMAND test_groupnorm)

    add_executable(test_upsample tests/test_upsample.cu)
    target_link_libraries(test_upsample PRIVATE f2k_cuda)
    add_test(NAME upsample COMMAND test_upsample)

    add_executable(test_vae_resnet tests/test_vae_resnet.cu)
    target_link_libraries(test_vae_resnet PRIVATE f2k_cuda)
    add_test(NAME vae_resnet COMMAND test_vae_resnet)

    add_executable(test_vae_attn tests/test_vae_attn.cu)
    target_link_libraries(test_vae_attn PRIVATE f2k_cuda)
    add_test(NAME vae_attn COMMAND test_vae_attn)

    add_executable(dump_vae tools/dump_vae.cpp)
    target_link_libraries(dump_vae PRIVATE f2k_common)

    add_executable(x_dim tools/x_dim.cpp)
    target_link_libraries(x_dim PRIVATE f2k_common)

    add_executable(dump_qwen tools/dump_qwen.cpp)
    target_link_libraries(dump_qwen PRIVATE f2k_common)

    add_executable(generate tools/generate.cu)
    target_link_libraries(generate PRIVATE f2k_cuda f2k_common)

    add_executable(bench_attention tools/bench_attention.cu)
    target_link_libraries(bench_attention PRIVATE f2k_cuda)

    add_executable(probe_ctx_emb tools/probe_ctx_emb.cu)
    target_link_libraries(probe_ctx_emb PRIVATE f2k_cuda f2k_common)

    add_executable(cmp_transformer tools/cmp_transformer.cu)
    target_link_libraries(cmp_transformer PRIVATE f2k_cuda f2k_common)

    add_executable(test_vae_decoder tests/test_vae_decoder.cu)
    target_link_libraries(test_vae_decoder PRIVATE f2k_cuda f2k_common)
    add_test(NAME vae_decoder COMMAND test_vae_decoder)

    add_executable(test_patchify tests/test_patchify.cu)
    target_link_libraries(test_patchify PRIVATE f2k_cuda)
    add_test(NAME patchify COMMAND test_patchify)

    add_executable(test_rope_4axis tests/test_rope_4axis.cu)
    target_link_libraries(test_rope_4axis PRIVATE f2k_cuda)
    add_test(NAME rope_4axis COMMAND test_rope_4axis)

    add_executable(test_qwen_attention tests/test_qwen_attention.cu)
    target_link_libraries(test_qwen_attention PRIVATE f2k_cuda)
    add_test(NAME qwen_attention COMMAND test_qwen_attention)

    add_executable(test_qwen_encoder tests/test_qwen_encoder.cu)
    target_link_libraries(test_qwen_encoder PRIVATE f2k_cuda f2k_common)
    add_test(NAME qwen_encoder COMMAND test_qwen_encoder)

    add_executable(test_qwen_golden tests/test_qwen_golden.cu)
    target_link_libraries(test_qwen_golden PRIVATE f2k_cuda f2k_common)
    # Not added to ctest because it depends on /tmp/qwen3_golden.bin existing.

    add_executable(test_rope tests/test_rope.cu)
    target_link_libraries(test_rope PRIVATE f2k_cuda)
    add_test(NAME rope COMMAND test_rope)

    add_executable(test_swiglu tests/test_swiglu.cu)
    target_link_libraries(test_swiglu PRIVATE f2k_cuda)
    add_test(NAME swiglu COMMAND test_swiglu)

    add_executable(test_attention tests/test_attention.cu)
    target_link_libraries(test_attention PRIVATE f2k_cuda)
    add_test(NAME attention COMMAND test_attention)

    add_executable(test_modulation tests/test_modulation.cu)
    target_link_libraries(test_modulation PRIVATE f2k_cuda)
    add_test(NAME modulation COMMAND test_modulation)

    add_executable(test_mmdit_block tests/test_mmdit_block.cu)
    target_link_libraries(test_mmdit_block PRIVATE f2k_cuda)
    add_test(NAME mmdit_block COMMAND test_mmdit_block)

    add_executable(test_mmdit_stack tests/test_mmdit_stack.cu)
    target_link_libraries(test_mmdit_stack PRIVATE f2k_cuda)
    add_test(NAME mmdit_stack COMMAND test_mmdit_stack)

    add_executable(test_joint_attention tests/test_joint_attention.cu)
    target_link_libraries(test_joint_attention PRIVATE f2k_cuda)
    add_test(NAME joint_attention COMMAND test_joint_attention)
endif()
