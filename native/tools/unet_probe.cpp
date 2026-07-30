#include "model_bundle.h"
#include "unet_cpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> load(const std::string& path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<float> values(count);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid fixture: " + path);
    }
    return values;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: lotus_unet_probe snapshot-root fixture-dir\n";
        return 2;
    }
    try {
        const lotus_native::ModelBundle bundle(argv[1]);
        const std::string root = argv[2];
        const std::vector<float> rgb =
            load(root + "/rgb_latent.bin", 4 * 8 * 8);
        const std::vector<float> noise =
            load(root + "/initial_latent.bin", 4 * 8 * 8);
        lotus_native::ImageTensor sample{8, 8, 8, {}};
        sample.values.reserve(rgb.size() + noise.size());
        sample.values.insert(sample.values.end(), rgb.begin(), rgb.end());
        sample.values.insert(sample.values.end(), noise.begin(), noise.end());
        const lotus_native::TokenTensor prompt{
            77, 1024, load(root + "/prompt.bin", 77 * 1024)};
        const std::vector<float> task = load(root + "/task.bin", 4);
        const lotus_native::ImageTensor prediction =
            lotus_native::unet_predict(
                bundle.unet(), sample, 999, prompt, task);
        const std::vector<float> reference =
            load(root + "/unet_prediction.bin", 4 * 8 * 8);
        double error = 0.0;
        double magnitude = 0.0;
        float maximum = 0.0f;
        for (std::size_t i = 0; i < reference.size(); ++i) {
            const float difference =
                std::abs(prediction.values[i] - reference[i]);
            error += difference;
            magnitude += std::abs(reference[i]);
            maximum = std::max(maximum, difference);
        }
        const double relative = error / magnitude;
        std::cout << "relative_l1=" << relative
                  << "\nmaximum_absolute=" << maximum << "\n";
        return relative <= 0.01 ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
