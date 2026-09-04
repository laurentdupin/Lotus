#include "lotus_native.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <iostream>
#include <string>
#include <vector>

int main(int argc, char** argv) {
    if (argc < 5 || argc > 7) {
        std::cerr
            << "usage: lotus_size_probe "
               "snapshot-root prompt-cache width height "
               "[gpu-device [iterations]]\n";
        return 2;
    }
    const std::uint32_t width =
        static_cast<std::uint32_t>(std::stoul(argv[3]));
    const std::uint32_t height =
        static_cast<std::uint32_t>(std::stoul(argv[4]));
    std::vector<float> rgb(std::uint64_t(width) * height * 3);
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            rgb[(std::uint64_t(y) * width + x) * 3 + 0] =
                std::sin(x * 0.071f) * 0.5f + 0.5f;
            rgb[(std::uint64_t(y) * width + x) * 3 + 1] =
                std::cos(y * 0.053f) * 0.5f + 0.5f;
            rgb[(std::uint64_t(y) * width + x) * 3 + 2] =
                std::sin((x + y) * 0.037f) * 0.5f + 0.5f;
        }
    }
    std::vector<float> depth(std::uint64_t(width) * height);
    lotus_context* context = nullptr;
    int status = argc >= 6
        ? lotus_create_vulkan(
            argv[1], argv[2],
            static_cast<std::uint32_t>(std::stoul(argv[5])), &context)
        : lotus_create(argv[1], argv[2], &context);
    const std::uint32_t iterations = argc == 7
        ? static_cast<std::uint32_t>(std::stoul(argv[6]))
        : 1u;
    std::vector<double> samples;
    if (status == LOTUS_OK) {
        for (std::uint32_t iteration = 0; iteration < iterations; ++iteration) {
            const auto start = std::chrono::steady_clock::now();
            status = lotus_infer_rgb_f32(
                context, rgb.data(), width, height, 17, depth.data());
            samples.push_back(std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count());
            if (status != LOTUS_OK) break;
        }
    }
    if (status != LOTUS_OK) {
        std::cerr << lotus_last_error() << "\n";
        lotus_destroy(context);
        return 1;
    }
    double sum = 0.0;
    for (float value : depth) {
        if (!std::isfinite(value) || value < 0.0f || value > 1.0f) {
            lotus_destroy(context);
            return 3;
        }
        sum += value;
    }
    std::cout << "width=" << width << "\nheight=" << height
              << "\nmean=" << sum / depth.size();
    if (!samples.empty()) {
        std::sort(samples.begin(), samples.end());
        std::cout << "\nmedian_ms=" << samples[samples.size() / 2]
                  << "\nminimum_ms=" << samples.front()
                  << "\nmaximum_ms=" << samples.back();
    }
    std::cout << "\n";
    lotus_destroy(context);
    return 0;
}
