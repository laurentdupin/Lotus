#include "model_bundle.h"
#include "vae_cpu.h"

#include <algorithm>
#include <cmath>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {

std::vector<float> load(const char* path, std::size_t count) {
    std::ifstream stream(path, std::ios::binary);
    std::vector<float> values(count);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(count * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error(std::string("invalid fixture: ") + path);
    }
    return values;
}

void compare(
    const char* label,
    const std::vector<float>& actual,
    const std::vector<float>& reference) {
    if (actual.size() != reference.size()) {
        throw std::runtime_error("comparison size mismatch");
    }
    double error = 0.0;
    double magnitude = 0.0;
    float maximum = 0.0f;
    for (std::size_t i = 0; i < actual.size(); ++i) {
        const float difference = std::abs(actual[i] - reference[i]);
        error += difference;
        magnitude += std::abs(reference[i]);
        maximum = std::max(maximum, difference);
    }
    const double relative = error / std::max(magnitude, 1.0e-30);
    std::cout << label << "_relative_l1=" << relative
              << "\n" << label << "_maximum_absolute=" << maximum << "\n";
    if (relative > 0.01) {
        throw std::runtime_error(
            std::string(label) + " exceeded the one-percent gate");
    }
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 3) {
        std::cerr << "usage: lotus_vae_probe snapshot-root fixture-dir\n";
        return 2;
    }
    try {
        const lotus_native::ModelBundle bundle(argv[1]);
        const std::string fixture = argv[2];
        lotus_native::ImageTensor rgb{
            3, 64, 64, load((fixture + "/rgb.bin").c_str(), 3 * 64 * 64)};
        const lotus_native::Posterior posterior =
            lotus_native::vae_encode(bundle.vae(), rgb);
        compare(
            "posterior_mean", posterior.mean.values,
            load((fixture + "/posterior_mean.bin").c_str(), 4 * 8 * 8));
        compare(
            "posterior_logvar", posterior.log_variance.values,
            load(
                (fixture + "/posterior_logvar.bin").c_str(),
                4 * 8 * 8));

        lotus_native::ImageTensor latent{
            4, 8, 8,
            load((fixture + "/unet_prediction.bin").c_str(), 4 * 8 * 8)};
        for (float& value : latent.values) {
            value /= 0.18215f;
        }
        const lotus_native::ImageTensor decoded =
            lotus_native::vae_decode(bundle.vae(), latent);
        compare(
            "decoded", decoded.values,
            load((fixture + "/decoded.bin").c_str(), 3 * 64 * 64));
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
