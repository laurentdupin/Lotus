#pragma once

#include "transformer_cpu.h"

#include <string>

namespace lotus_native {

TokenTensor load_empty_prompt_cache(
    const std::string& path_utf8,
    std::uint32_t unet_input_channels);

}  // namespace lotus_native
