cmake_minimum_required(VERSION 3.21)
project(lotus_native LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

add_library(lotus_model STATIC
    src/model_bundle.cpp
    src/safetensors.cpp
    src/tensor_cpu.cpp
    src/vae_cpu.cpp
)
target_include_directories(lotus_model PUBLIC src)

add_executable(lotus_model_probe tools/model_probe.cpp)
target_link_libraries(lotus_model_probe PRIVATE lotus_model)

add_executable(lotus_vae_probe tools/vae_probe.cpp)
target_link_libraries(lotus_vae_probe PRIVATE lotus_model)

enable_testing()
add_test(NAME lotus_model_probe_usage COMMAND lotus_model_probe)
set_tests_properties(lotus_model_probe_usage PROPERTIES WILL_FAIL TRUE)
