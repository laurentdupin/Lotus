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

ModelBundle::ModelBundle(
    const std::string& root,
    bool load_text_encoder)
    : unet_(std::make_unique<SafeTensors>(
          model_path(root, "unet", "diffusion_pytorch_model.safetensors"))),
      vae_(std::make_unique<SafeTensors>(
          model_path(root, "vae", "diffusion_pytorch_model.safetensors"))) {
    if (load_text_encoder) {
        text_encoder_ = std::make_unique<SafeTensors>(
            model_path(root, "text_encoder", "model.safetensors"));
    }
}

const SafeTensors& ModelBundle::text_encoder() const {
    if (!text_encoder_) {
        throw std::runtime_error("Lotus text encoder was not loaded");
    }
    return *text_encoder_;
}

}  // namespace lotus_native
