#if defined(__linux__) && !defined(__ANDROID__) && defined(LOTUS_WITH_VULKAN)
#include "linux_capture.h"
#include <inferbridge/linux_capture_preprocess.h>
ibr_linux_capture_capabilities
lotus_linux_capture_capabilities(lotus_context *context) {
  return context->vulkan ? context->vulkan->linux_capture_capabilities()
                         : ibr_linux_capture_capabilities{};
}
void lotus_infer_linux_capture(
    lotus_context *context,
    const inferbridge::linux_capture::LinuxDmaBufImage &source, uint64_t seed,
    float *output) {
  if (!context->vulkan)
    throw std::runtime_error("lotus Vulkan context unavailable");
  auto &vk = *context->vulkan;
  uint32_t width = 0, height = 0;
  inferbridge_shape(source.width, source.height, width, height);
  auto input = inferbridge::linux_capture::capture_tensor(
      vk, source, width, height,
      {3, true, {.5f, .5f, .5f, 0}, {.5f, .5f, .5f, 1}});
  const uint64_t count = uint64_t(4) * (width / 8) * (height / 8);
  std::mt19937_64 generator(seed);
  std::normal_distribution<float> normal;
  const auto noise = [&] {
    std::vector<float> values(count);
    for (auto &value : values)
      value = normal(generator);
    auto buffer = vk.create_device_buffer(count * sizeof(float));
    vk.upload(buffer, values.data(), count * sizeof(float));
    return buffer;
  };
  auto first = noise();
  lotus_native::LotusGpuGraph graph(vk, *context->gpu_unet, *context->gpu_vae,
                                    *context->operators, context->prompt);
  auto second = noise();
  auto result =
      graph.infer_device(std::move(input), width, height, std::move(first),
                         std::move(second), width, height);
  std::vector<float> depth(uint64_t(width) * height);
  vk.download(result, depth.data(), depth.size() * sizeof(float));
  resize_and_normalize(depth.data(), width, height, source.width, source.height,
                       output);
}
#endif
