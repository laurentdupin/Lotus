cmake_minimum_required(VERSION 3.21)
project(lotus_native LANGUAGES C CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lotus_model STATIC
    src/model_bundle.cpp
    src/prompt_cache.cpp
    src/safetensors.cpp
    src/tensor_cpu.cpp
    src/transformer_cpu.cpp
    src/unet_cpu.cpp
    src/vae_cpu.cpp
)
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
