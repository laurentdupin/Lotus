#include "external_gpu.h"

#include <stdexcept>

namespace lotus_native {

std::shared_ptr<ExternalGpu> create_external_gpu(
    const std::string&, const std::string&, std::uint32_t) {
    throw std::runtime_error(
        "Lotus external D3D12 interop is unavailable in this build");
}

ExternalGpuCapabilities probe_external_gpu(std::uint32_t) {
    return {};
}

}  // namespace lotus_native
