#include "model_bundle.h"

#include <filesystem>
#include <stdexcept>

namespace lotus_native {
namespace {

std::string model_path(
    const std::string& root,
    const char* component,
    const char* file) {
    const std::filesystem::path path =
        std::filesystem::u8path(root) / component / file;
    if (!std::filesystem::is_regular_file(path)) {
        throw std::runtime_error(
            "missing Lotus model component: " + path.u8string());
    }
    return path.u8string();
}

}  // namespace

ModelBundle::ModelBundle(const std::string& root)
    : unet_(std::make_unique<SafeTensors>(
          model_path(root, "unet", "diffusion_pytorch_model.safetensors"))),
      vae_(std::make_unique<SafeTensors>(
          model_path(root, "vae", "diffusion_pytorch_model.safetensors"))),
      text_encoder_(std::make_unique<SafeTensors>(
          model_path(root, "text_encoder", "model.safetensors"))) {}

}  // namespace lotus_native
