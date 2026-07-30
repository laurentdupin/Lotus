#pragma once

#include "transformer_cpu.h"

namespace lotus_native {

ImageTensor unet_predict(
    const SafeTensors& model,
    const ImageTensor& sample,
    std::uint32_t timestep,
    const TokenTensor& prompt,
    const std::vector<float>& class_labels);

}  // namespace lotus_native
