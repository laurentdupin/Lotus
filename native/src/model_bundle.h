#pragma once

#include "safetensors.h"

#include <memory>
#include <string>

namespace lotus_native {

class ModelBundle {
public:
    explicit ModelBundle(const std::string& snapshot_root_utf8);

    const SafeTensors& unet() const { return *unet_; }
    const SafeTensors& vae() const { return *vae_; }
    const SafeTensors& text_encoder() const { return *text_encoder_; }

private:
    std::unique_ptr<SafeTensors> unet_;
    std::unique_ptr<SafeTensors> vae_;
    std::unique_ptr<SafeTensors> text_encoder_;
};

}  // namespace lotus_native
