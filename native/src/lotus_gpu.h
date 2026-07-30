#pragma once

#include "gpu_model.h"
#include "operators.h"
#include "prompt_cache.h"

#include <string>

namespace lotus_native {

struct GpuImage {
    VulkanBuffer buffer;
    std::uint32_t channels = 0;
    std::uint32_t height = 0;
    std::uint32_t width = 0;
};

VulkanBuffer lotus_infer_gpu(
    VulkanContext& context,
    GpuModel& unet,
    GpuModel& vae,
    VulkanOperators& operators,
    const TokenTensor& prompt,
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* initial_noise,
    const float* posterior_noise);

GpuImage lotus_vae_encode_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* normalized_rgb_nchw, std::uint32_t width,
    std::uint32_t height);
GpuImage lotus_unet_gpu(
    VulkanContext& context, GpuModel& unet, VulkanOperators& operators,
    const TokenTensor& prompt, const float* sample_nchw,
    std::uint32_t width, std::uint32_t height);
GpuImage lotus_vae_decode_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* latent_nchw, std::uint32_t width, std::uint32_t height);
GpuImage lotus_spatial_attention_gpu(
    VulkanContext& context, GpuModel& vae, VulkanOperators& operators,
    const float* input_nchw, std::uint32_t channels,
    std::uint32_t width, std::uint32_t height,
    const std::string& prefix);

}  // namespace lotus_native
