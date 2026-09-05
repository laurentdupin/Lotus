#include "lotus_native.h"

#include "model_bundle.h"
#include "prompt_cache.h"
#include "unet_cpu.h"
#include "vae_cpu.h"
#include "inferbridge/native_harness_diffusion_shape.h"
#include "lotus_internal.h"
#if defined(LOTUS_WITH_METAL)
#include "metal_executor.h"
#endif
#if defined(LOTUS_WITH_VULKAN)
#include "gpu_model.h"
#include "lotus_gpu.h"
#include "operators.h"
#include "vulkan.h"
#endif

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
#if defined(LOTUS_WITH_METAL)
    std::unique_ptr<lotus_native::MetalExecutor> metal;
#endif
#if defined(LOTUS_WITH_VULKAN)
    std::unique_ptr<lotus_native::VulkanContext> vulkan;
    std::unique_ptr<lotus_native::GpuModel> gpu_unet;
    std::unique_ptr<lotus_native::GpuModel> gpu_vae;
    std::unique_ptr<lotus_native::VulkanOperators> operators;
#endif
};

namespace lotus_native {
#if defined(LOTUS_WITH_METAL)
class MetalContextExternalGpu final : public ExternalGpu {
public:
    explicit MetalContextExternalGpu(lotus_context* context) : context_(context) {}
    ExternalGpuCapabilities capabilities() const override { return {true, 0u, 3u}; }
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request) override {
        if (!context_ || !context_->metal)
            throw std::invalid_argument("Lotus Metal context is unavailable");
        return context_->metal->submit_texture(request);
    }
    void transfer_counters(std::uint64_t& upload,
                           std::uint64_t& download) const override {
        upload = 0u; download = 0u;
    }
private:
    lotus_context* context_ = nullptr;
};
#endif

std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    lotus_context* context, const std::string& cache_path) {
#if defined(LOTUS_WITH_METAL)
    if (!context || !context->metal)
        throw std::invalid_argument("Lotus Metal context is unavailable");
    context->metal->set_cache_path(cache_path);
    return std::make_shared<MetalContextExternalGpu>(context);
#else
    (void)context;
    (void)cache_path;
    throw std::invalid_argument("Lotus was built without Metal");
#endif
}
}

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
    const std::uint32_t input_channels = static_cast<std::uint32_t>(
        context.model->unet().tensor("conv_in.weight").dimensions[1]);
    lotus_native::ImageTensor sample{
        input_channels, latent_height, latent_width, {}};
    if (input_channels == 8) {
        sample.values.reserve(static_cast<std::size_t>(latent_elements * 2));
        sample.values.insert(
            sample.values.end(),
            rgb_latent.values.begin(), rgb_latent.values.end());
        sample.values.insert(
            sample.values.end(),
            initial_noise, initial_noise + latent_elements);
    } else if (input_channels == 4) {
        sample.values = std::move(rgb_latent.values);
    } else {
        throw std::runtime_error("unsupported Lotus UNet input channels");
    }
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

void inferbridge_shape(
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t& processing_width,
    std::uint32_t& processing_height) {
    inferbridge::native_harness::fit_diffusion_shape(
        width, height, processing_width, processing_height);
}

std::vector<float> preprocess_bgra(
    const std::uint8_t* bgra,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_stride,
    std::uint32_t processing_width,
    std::uint32_t processing_height) {
    std::vector<float> output(
        std::uint64_t(processing_width) * processing_height * 3);
    for (std::uint32_t y = 0; y < processing_height; ++y) {
        const std::uint32_t source_y = std::min(
            height - 1,
            static_cast<std::uint32_t>(
                std::uint64_t(y) * height / processing_height));
        const std::uint8_t* row =
            bgra + std::uint64_t(source_y) * row_stride;
        for (std::uint32_t x = 0; x < processing_width; ++x) {
            const std::uint32_t source_x = std::min(
                width - 1,
                static_cast<std::uint32_t>(
                    std::uint64_t(x) * width / processing_width));
            for (std::uint32_t c = 0; c < 3; ++c) {
                output[
                    (std::uint64_t(y) * processing_width + x) * 3 + c] =
                    static_cast<float>(row[std::uint64_t(source_x) * 4 + c]) /
                    255.0f;
            }
        }
    }
    return output;
}

void resize_and_normalize(
    const float* processing_depth,
    std::uint32_t processing_width,
    std::uint32_t processing_height,
    std::uint32_t width,
    std::uint32_t height,
    float* depth) {
    for (std::uint32_t y = 0; y < height; ++y) {
        const std::uint32_t source_y = std::min(
            processing_height - 1,
            static_cast<std::uint32_t>(
                std::uint64_t(y) * processing_height / height));
        for (std::uint32_t x = 0; x < width; ++x) {
            const std::uint32_t source_x = std::min(
                processing_width - 1,
                static_cast<std::uint32_t>(
                    std::uint64_t(x) * processing_width / width));
            const float value = processing_depth[
                std::uint64_t(source_y) * processing_width + source_x];
            depth[std::uint64_t(y) * width + x] = value;
        }
    }
    const std::uint64_t count = std::uint64_t(width) * height;
    float minimum = depth[0];
    float maximum = depth[0];
    for (std::uint64_t i = 1; i < count; ++i) {
        minimum = std::min(minimum, depth[i]);
        maximum = std::max(maximum, depth[i]);
    }
    const float range = maximum - minimum;
    for (std::uint64_t i = 0; i < count; ++i) {
        depth[i] = range > 0.0f ? (depth[i] - minimum) / range : 0.0f;
    }
}

}  // namespace

int lotus_get_transfer_counters(
    uint64_t* upload_bytes, uint64_t* download_bytes) {
    if (!upload_bytes || !download_bytes)
        return fail(LOTUS_INVALID_ARGUMENT, "invalid transfer counter output");
#if defined(LOTUS_WITH_VULKAN)
    lotus_native::global_transfer_counters(*upload_bytes, *download_bytes);
#else
    *upload_bytes = 0u;
    *download_bytes = 0u;
#endif
    return LOTUS_OK;
}

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
        const std::uint32_t input_channels = static_cast<std::uint32_t>(
            context->model->unet().tensor(
                "conv_in.weight").dimensions[1]);
        context->prompt = lotus_native::load_empty_prompt_cache(
            prompt_cache, input_channels);
        *output = context.release();
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_MODEL_ERROR, error);
    }
}

int lotus_create_vulkan(
    const char* snapshot_root,
    const char* prompt_cache,
    std::uint32_t device_index,
    lotus_context** output) {
    if (!snapshot_root || !prompt_cache || !output) {
        return fail(LOTUS_INVALID_ARGUMENT, "invalid Lotus create argument");
    }
    *output = nullptr;
#if defined(LOTUS_WITH_METAL)
    (void)device_index;
    try {
        auto context = std::make_unique<lotus_context>();
        context->model = std::make_unique<lotus_native::ModelBundle>(
            snapshot_root, false);
        const std::uint32_t input_channels = static_cast<std::uint32_t>(
            context->model->unet().tensor("conv_in.weight").dimensions[1]);
        context->prompt = lotus_native::load_empty_prompt_cache(
            prompt_cache, input_channels);
        context->metal = std::make_unique<lotus_native::MetalExecutor>(
            *context->model, context->prompt);
        *output = context.release();
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_MODEL_ERROR, error);
    }
#elif !defined(LOTUS_WITH_VULKAN)
    (void)device_index;
    return fail(LOTUS_RUNTIME_ERROR, "this DLL was built without Vulkan");
#else
    try {
        auto context = std::make_unique<lotus_context>();
        context->model =
            std::make_unique<lotus_native::ModelBundle>(
                snapshot_root, false);
        const std::uint32_t input_channels = static_cast<std::uint32_t>(
            context->model->unet().tensor(
                "conv_in.weight").dimensions[1]);
        context->prompt = lotus_native::load_empty_prompt_cache(
            prompt_cache, input_channels);
        context->vulkan =
            std::make_unique<lotus_native::VulkanContext>(device_index);
        context->gpu_unet = std::make_unique<lotus_native::GpuModel>(
            context->model->unet(), *context->vulkan);
        context->gpu_vae = std::make_unique<lotus_native::GpuModel>(
            context->model->vae(), *context->vulkan);
        context->operators =
            std::make_unique<lotus_native::VulkanOperators>(
                *context->vulkan);
        *output = context.release();
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_MODEL_ERROR, error);
    }
#endif
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
#if defined(LOTUS_WITH_METAL)
        if (context->metal) {
            output_depth(
                context->metal->infer(
                    rgb, width, height, initial_noise, posterior_noise),
                width, height, depth);
            last_error.clear();
            return LOTUS_OK;
        }
#endif
#if defined(LOTUS_WITH_VULKAN)
        if (context->vulkan) {
            lotus_native::VulkanBuffer output =
                lotus_native::lotus_infer_gpu(
                    *context->vulkan, *context->gpu_unet,
                    *context->gpu_vae, *context->operators,
                    context->prompt, rgb, width, height,
                    initial_noise, posterior_noise);
            context->vulkan->download(
                output, depth,
                std::uint64_t(width) * height * sizeof(float));
            last_error.clear();
            return LOTUS_OK;
        }
#endif
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

int lotus_inferbridge_image_shape(
    std::uint32_t source_width,
    std::uint32_t source_height,
    std::uint32_t* processing_width,
    std::uint32_t* processing_height) {
    if (!processing_width || !processing_height) {
        return fail(LOTUS_INVALID_ARGUMENT, "invalid Lotus shape argument");
    }
    try {
        inferbridge_shape(
            source_width, source_height,
            *processing_width, *processing_height);
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_INVALID_ARGUMENT, error);
    }
}

int lotus_infer_bgra8_f32_with_noise(
    lotus_context* context,
    const std::uint8_t* bgra,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_stride_bytes,
    const float* initial_noise,
    const float* posterior_noise,
    float* depth) {
    if (!context || !bgra || !initial_noise || !posterior_noise || !depth ||
        width == 0 || height == 0 ||
        row_stride_bytes < std::uint64_t(width) * 4) {
        return fail(
            LOTUS_INVALID_ARGUMENT, "invalid Lotus BGRA inference argument");
    }
    try {
        std::uint32_t processing_width = 0;
        std::uint32_t processing_height = 0;
        inferbridge_shape(
            width, height, processing_width, processing_height);
        std::vector<float> rgb = preprocess_bgra(
            bgra, width, height, row_stride_bytes,
            processing_width, processing_height);
        std::vector<float> processing_depth(
            std::uint64_t(processing_width) * processing_height);
        const int result = lotus_infer_rgb_f32_with_noise(
            context, rgb.data(), processing_width, processing_height,
            initial_noise, posterior_noise, processing_depth.data());
        if (result != LOTUS_OK) {
            return result;
        }
        resize_and_normalize(
            processing_depth.data(), processing_width, processing_height,
            width, height, depth);
        last_error.clear();
        return LOTUS_OK;
    } catch (const std::exception& error) {
        return fail(LOTUS_RUNTIME_ERROR, error);
    }
}

int lotus_infer_bgra8_f32(
    lotus_context* context,
    const std::uint8_t* bgra,
    std::uint32_t width,
    std::uint32_t height,
    std::uint32_t row_stride_bytes,
    std::uint64_t seed,
    float* depth) {
    std::uint32_t processing_width = 0;
    std::uint32_t processing_height = 0;
    const int shape_result = lotus_inferbridge_image_shape(
        width, height, &processing_width, &processing_height);
    if (shape_result != LOTUS_OK) {
        return shape_result;
    }
    const std::uint64_t count =
        std::uint64_t(4) * (processing_width / 8) *
        (processing_height / 8);
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
    return lotus_infer_bgra8_f32_with_noise(
        context, bgra, width, height, row_stride_bytes,
        initial.data(), posterior.data(), depth);
}

}  // extern "C"

#include "linux_capture.inl"
