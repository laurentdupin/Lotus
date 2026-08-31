#pragma once

#include "external_gpu.h"
#include "lotus_native.h"

#include <memory>
#include <string>

namespace lotus_native {
std::shared_ptr<ExternalGpu> create_metal_external_gpu(
    lotus_context* context, const std::string& cache_path);
}
