cmake_minimum_required(VERSION 3.21)
project(lotus_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

option(LOTUS_WITH_VULKAN
    "Build the dependency-free Vulkan executor" ON)

if(LOTUS_WITH_VULKAN)
    find_package(Vulkan REQUIRED COMPONENTS glslc)
    set(LOTUS_GENERATED_DIR "${CMAKE_CURRENT_BINARY_DIR}/generated")
    file(MAKE_DIRECTORY "${LOTUS_GENERATED_DIR}")
    function(lotus_compile_shader NAME)
        set(SOURCE "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/${NAME}.glsl")
        set(SPIRV "${LOTUS_GENERATED_DIR}/${NAME}.spv")
        set(HEADER "${LOTUS_GENERATED_DIR}/${NAME}_spv.h")
        add_custom_command(
            OUTPUT "${SPIRV}"
            COMMAND Vulkan::glslc -fshader-stage=compute
                --target-env=vulkan1.3 --target-spv=spv1.6 -Werror
                "-I${CMAKE_CURRENT_SOURCE_DIR}/src/shaders"
                -o "${SPIRV}" "${SOURCE}"
            DEPENDS "${SOURCE}"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/indexing.h"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv2d_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/linear_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/project_tokens_common.glsl"
                "${CMAKE_CURRENT_SOURCE_DIR}/src/shaders/conv_transpose_nonoverlap_common.glsl"
            VERBATIM)
        add_custom_command(
            OUTPUT "${HEADER}"
            COMMAND "${CMAKE_COMMAND}" "-DINPUT=${SPIRV}"
                "-DOUTPUT=${HEADER}" "-DVARIABLE=lotus_${NAME}_spv"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            DEPENDS "${SPIRV}" "${CMAKE_CURRENT_SOURCE_DIR}/cmake/embed_binary.cmake"
            VERBATIM)
        set(LOTUS_SHADER_HEADERS
            ${LOTUS_SHADER_HEADERS} "${HEADER}" PARENT_SCOPE)
    endfunction()
    foreach(SHADER
            add_scaled add add_position bilinear_align_true
            bilinear_align_true_image bmm bmm_score_half bmm_value_half
            conv2d conv2d8 conv2d_half conv2d8_half
            conv_transpose_nonoverlap conv_transpose_nonoverlap_half
            elementwise gelu layer_norm linear linear16 linear_half
            linear16_half prepare_tokens position_bicubic project_tokens
            project_tokens_half relu softmax_lastdim softmax_lastdim_half)
        lotus_compile_shader(${SHADER})
    endforeach()
endif()

add_library(lotus_model STATIC
    src/model_bundle.cpp
    src/prompt_cache.cpp
    src/safetensors.cpp
    src/tensor_cpu.cpp
    src/transformer_cpu.cpp
    src/unet_cpu.cpp
    src/vae_cpu.cpp
)
if(LOTUS_WITH_VULKAN)
    target_sources(lotus_model PRIVATE
        src/vulkan.cpp src/operators.cpp src/gpu_model.cpp
        ${LOTUS_SHADER_HEADERS})
    target_include_directories(
        lotus_model PRIVATE "${LOTUS_GENERATED_DIR}")
    target_link_libraries(lotus_model PRIVATE Vulkan::Vulkan)
    target_compile_definitions(
        lotus_model PUBLIC LOTUS_WITH_VULKAN=1)
endif()
target_include_directories(lotus_model PUBLIC src)
set_target_properties(lotus_model PROPERTIES POSITION_INDEPENDENT_CODE ON)

add_library(lotus_native SHARED src/lotus.cpp)
target_compile_definitions(lotus_native PRIVATE LOTUS_NATIVE_BUILD)
target_include_directories(lotus_native PUBLIC include)
target_link_libraries(lotus_native PRIVATE lotus_model)
set_target_properties(
    lotus_native PROPERTIES
    OUTPUT_NAME "lotus_native"
    WINDOWS_EXPORT_ALL_SYMBOLS OFF)

add_executable(lotus_model_probe tools/model_probe.cpp)
target_link_libraries(lotus_model_probe PRIVATE lotus_model)

add_executable(lotus_vae_probe tools/vae_probe.cpp)
target_link_libraries(lotus_vae_probe PRIVATE lotus_model)

add_executable(lotus_unet_probe tools/unet_probe.cpp)
target_link_libraries(lotus_unet_probe PRIVATE lotus_model)

add_executable(lotus_full_graph_probe tools/full_graph_probe.cpp)
target_include_directories(lotus_full_graph_probe PRIVATE include)
target_link_libraries(lotus_full_graph_probe PRIVATE lotus_native)

add_executable(lotus_size_probe tools/size_probe.cpp)
target_include_directories(lotus_size_probe PRIVATE include)
target_link_libraries(lotus_size_probe PRIVATE lotus_native)

add_executable(lotus_c_abi_smoke tests/c_abi_smoke.c)
target_include_directories(lotus_c_abi_smoke PRIVATE include)
target_link_libraries(lotus_c_abi_smoke PRIVATE lotus_native)

enable_testing()
add_test(NAME lotus_model_probe_usage COMMAND lotus_model_probe)
set_tests_properties(lotus_model_probe_usage PROPERTIES WILL_FAIL TRUE)
add_test(NAME lotus_c_abi_smoke COMMAND lotus_c_abi_smoke)
