#include "lotus_native.h"

#include "model_bundle.h"
#include "prompt_cache.h"
#include "unet_cpu.h"
#include "vae_cpu.h"

#include <algorithm>
#include <cmath>
#include <memory>
#include <new>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

struct lotus_context {
    std::unique_ptr<lotus_native::ModelBundle> model;
    lotus_native::TokenTensor prompt;
};

namespace {

thread_local std::string last_error;

int fail(int code, const char* message) {
    last_error = message;
    return code;
}

int fail(int code, const std::exception& error) {
    last_error = error.what();
    return code;
}

lotus_native::ImageTensor infer(
    lotus_context& context,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* initial_noise,
    const float* posterior_noise) {
    const std::uint32_t latent_width = width / 8;
    const std::uint32_t latent_height = height / 8;
    if (width < 8 || height < 8 || latent_width == 0 || latent_height == 0) {
        throw std::invalid_argument("Lotus input dimensions are too small");
    }
    lotus_native::ImageTensor image{
        3, height, width,
        std::vector<float>(std::uint64_t(3) * height * width)};
    for (std::uint32_t y = 0; y < height; ++y) {
        for (std::uint32_t x = 0; x < width; ++x) {
            for (std::uint32_t c = 0; c < 3; ++c) {
                image.values[
                    (std::uint64_t(c) * height + y) * width + x] =
                    rgb[(std::uint64_t(y) * width + x) * 3 + c] *
                        2.0f -
                    1.0f;
            }
        }
    }
    const lotus_native::Posterior posterior =
        lotus_native::vae_encode(context.model->vae(), image);
    const std::uint64_t latent_elements =
        std::uint64_t(4) * latent_height * latent_width;
    if (posterior.mean.values.size() != latent_elements) {
        throw std::runtime_error("Lotus latent dimensions are inconsistent");
    }
    lotus_native::ImageTensor rgb_latent{
        4, latent_height, latent_width, posterior.mean.values};
    for (std::uint64_t i = 0; i < latent_elements; ++i) {
        rgb_latent.values[i] =
            (rgb_latent.values[i] +
             std::exp(0.5f * posterior.log_variance.values[i]) *
                 posterior_noise[i]) *
            0.18215f;
    }
    lotus_native::ImageTensor sample{
        8, latent_height, latent_width, {}};
    sample.values.reserve(static_cast<std::size_t>(latent_elements * 2));
    sample.values.insert(
        sample.values.end(),
        rgb_latent.values.begin(), rgb_latent.values.end());
    sample.values.insert(
        sample.values.end(),
        initial_noise, initial_noise + latent_elements);
    const std::vector<float> task = {
        std::sin(1.0f), 0.0f, std::cos(1.0f), 1.0f};
    lotus_native::ImageTensor prediction = lotus_native::unet_predict(
        context.model->unet(), sample, 999, context.prompt, task);
    for (float& value : prediction.values) {
        value /= 0.18215f;
    }
    return lotus_native::vae_decode(context.model->vae(), prediction);
}

void output_depth(
    const lotus_native::ImageTensor& decoded,
    std::uint32_t target_width,
    std::uint32_t target_height,
    float* output) {
    for (std::uint32_t y = 0; y < target_height; ++y) {
        const std::uint32_t source_y =
            std::min(
                decoded.height - 1,
                static_cast<std::uint32_t>(
                    std::uint64_t(y) * decoded.height / target_height));
        for (std::uint32_t x = 0; x < target_width; ++x) {
            const std::uint32_t source_x =
                std::min(
                    decoded.width - 1,
                    static_cast<std::uint32_t>(
                        std::uint64_t(x) * decoded.width / target_width));
            float sum = 0.0f;
            for (std::uint32_t c = 0; c < 3; ++c) {
                const float raw = decoded.values[
                    (std::uint64_t(c) * decoded.height + source_y) *
                        decoded.width +
                    source_x];
                sum += std::clamp(raw * 0.5f + 0.5f, 0.0f, 1.0f);
            }
            output[std::uint64_t(y) * target_width + x] = sum / 3.0f;
        }
    }
}

}  // namespace

extern "C" {

std::uint32_t lotus_native_abi_version(void) {
    return LOTUS_NATIVE_ABI_VERSION;
}

const char* lotus_last_error(void) {
    return last_error.c_str();
}

int lotus_create(
    const char* snapshot_root,
    const char* prompt_cache,
    lotus_context** output) {
    if (!snapshot_root || !prompt_cache || !output) {
        return fail(LOTUS_INVALID_ARGUMENT, "invalid Lotus create argument");
    }
    *output = nullptr;
    try {
        auto context = std::make_unique<lotus_context>();
        context->model =
            std::make_unique<lotus_native::ModelBundle>(
                snapshot_root, false);
        context->prompt =
            lotus_native::load_empty_prompt_cache(prompt_cache);
        *output = context.release();
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_MODEL_ERROR, error);
    }
}

void lotus_destroy(lotus_context* context) {
    delete context;
}

int lotus_infer_rgb_f32_with_noise(
    lotus_context* context,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* initial_noise,
    const float* posterior_noise,
    float* depth) {
    if (!context || !rgb || !initial_noise || !posterior_noise || !depth) {
        return fail(LOTUS_INVALID_ARGUMENT, "invalid Lotus inference argument");
    }
    try {
        output_depth(
            infer(
                *context, rgb, width, height,
                initial_noise, posterior_noise),
            width, height, depth);
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_RUNTIME_ERROR, error);
    }
}

int lotus_infer_rgb_f32(
    lotus_context* context,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    std::uint64_t seed,
    float* depth) {
    if (!context || !rgb || !depth || width < 8 || height < 8) {
        return fail(LOTUS_INVALID_ARGUMENT, "invalid Lotus inference argument");
    }
    const std::uint64_t count =
        std::uint64_t(4) * (width / 8) * (height / 8);
    std::mt19937_64 generator(seed);
    std::normal_distribution<float> normal;
    std::vector<float> initial(static_cast<std::size_t>(count));
    std::vector<float> posterior(static_cast<std::size_t>(count));
    for (float& value : initial) {
        value = normal(generator);
    }
    for (float& value : posterior) {
        value = normal(generator);
    }
    return lotus_infer_rgb_f32_with_noise(
        context, rgb, width, height,
        initial.data(), posterior.data(), depth);
}

}  // extern "C"
