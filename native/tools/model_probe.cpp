#include "model_bundle.h"

#include <iostream>

namespace {

bool shape(
    const lotus_native::TensorView& tensor,
    std::initializer_list<std::uint64_t> expected) {
    if (tensor.rank != expected.size()) {
        return false;
    }
    std::size_t index = 0;
    for (const std::uint64_t dimension : expected) {
        if (tensor.dimensions[index++] != dimension) {
            return false;
        }
    }
    return true;
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::cerr << "usage: lotus_model_probe snapshot-root\n";
        return 2;
    }

    try {
        const lotus_native::ModelBundle model(argv[1]);
        const bool valid =
            model.unet().tensor_count() == 690 &&
            model.vae().tensor_count() == 248 &&
            model.text_encoder().tensor_count() == 372 &&
            shape(model.unet().tensor("conv_in.weight"), {320, 8, 3, 3}) &&
            shape(model.unet().tensor("conv_out.weight"), {4, 320, 3, 3}) &&
            shape(
                model.unet().tensor("class_embedding.linear_1.weight"),
                {1280, 4}) &&
            shape(
                model.vae().tensor("encoder.conv_in.weight"),
                {128, 3, 3, 3}) &&
            shape(model.vae().tensor("quant_conv.weight"), {8, 8, 1, 1}) &&
            shape(
                model.text_encoder().tensor(
                    "text_model.embeddings.token_embedding.weight"),
                {49408, 1024});

        std::cout << "unet_tensors=" << model.unet().tensor_count()
                  << "\nvae_tensors=" << model.vae().tensor_count()
                  << "\ntext_encoder_tensors="
                  << model.text_encoder().tensor_count() << "\n";
        return valid ? 0 : 3;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
