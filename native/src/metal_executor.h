#pragma once

#include "model_bundle.h"
#include "transformer_cpu.h"
#include "external_gpu.h"

#include <cstdint>
#include <memory>

namespace lotus_native {

class MetalExecutor {
public:
    MetalExecutor(const ModelBundle& model, const TokenTensor& prompt);
    ~MetalExecutor();
    MetalExecutor(const MetalExecutor&) = delete;
    MetalExecutor& operator=(const MetalExecutor&) = delete;

    ImageTensor infer(
        const float* rgb,
        std::uint32_t width,
        std::uint32_t height,
        const float* initial_noise,
        const float* posterior_noise);
    std::shared_ptr<ExternalJob> submit_texture(
        const ExternalTextureRequest& request);

private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace lotus_native
