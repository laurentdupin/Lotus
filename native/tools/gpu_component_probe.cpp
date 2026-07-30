#include "gpu_model.h"
#include "lotus_gpu.h"
#include "prompt_cache.h"
#include "tensor_cpu.h"
#include "transformer_cpu.h"

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

std::vector<float> load(const std::filesystem::path& path, std::size_t count) {
    std::vector<float> values(count);
    std::ifstream stream(path, std::ios::binary);
    stream.read(
        reinterpret_cast<char*>(values.data()),
        static_cast<std::streamsize>(values.size() * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid fixture");
    }
    return values;
}

void compare(
    const char* name, const std::vector<float>& actual,
    const std::vector<float>& reference) {
    double error = 0.0, magnitude = 0.0;
    float maximum = 0.0f;
    for (std::size_t i = 0; i < reference.size(); ++i) {
        const float difference = std::abs(actual[i] - reference[i]);
        error += difference;
        magnitude += std::abs(reference[i]);
        maximum = std::max(maximum, difference);
    }
    std::cout << name << "_relative_l1=" << error / magnitude
              << "\n" << name << "_maximum_absolute=" << maximum << "\n";
}

int main(int argc, char** argv) {
    if (argc != 5) return 2;
    try {
        const std::filesystem::path snapshot = argv[1];
        const std::filesystem::path fixture = argv[3];
        lotus_native::SafeTensors unet(
            (snapshot / "unet/diffusion_pytorch_model.safetensors").string());
        lotus_native::SafeTensors vae(
            (snapshot / "vae/diffusion_pytorch_model.safetensors").string());
        lotus_native::VulkanContext context(std::stoul(argv[4]));
        lotus_native::VulkanOperators operators(context);
        {
            constexpr std::uint32_t tokens = 64;
            constexpr std::uint32_t dimensions = 512;
            std::vector<float> values(tokens * dimensions);
            for (std::size_t i = 0; i < values.size(); ++i) {
                values[i] = std::sin(static_cast<float>(i) * 0.007f);
            }
            std::vector<float> reference(values.size());
            std::vector<float> scores(tokens);
            const float scale = 1.0f / std::sqrt(float(dimensions));
            for (std::uint32_t query = 0; query < tokens; ++query) {
                float maximum = -INFINITY;
                for (std::uint32_t key = 0; key < tokens; ++key) {
                    float sum = 0.0f;
                    for (std::uint32_t d = 0; d < dimensions; ++d) {
                        sum += values[query * dimensions + d] *
                            values[key * dimensions + d];
                    }
                    scores[key] = sum * scale;
                    maximum = std::max(maximum, scores[key]);
                }
                float denominator = 0.0f;
                for (float& score : scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (std::uint32_t d = 0; d < dimensions; ++d) {
                    float sum = 0.0f;
                    for (std::uint32_t key = 0; key < tokens; ++key) {
                        sum += scores[key] / denominator *
                            values[key * dimensions + d];
                    }
                    reference[query * dimensions + d] = sum;
                }
            }
            auto input_buffer =
                context.create_device_buffer(values.size() * sizeof(float));
            auto output_buffer =
                context.create_device_buffer(values.size() * sizeof(float));
            context.upload(
                input_buffer, values.data(), values.size() * sizeof(float));
            operators.attention_separate(
                output_buffer, input_buffer, input_buffer, input_buffer,
                tokens, tokens, 1, dimensions);
            std::vector<float> actual(values.size());
            context.download(
                output_buffer, actual.data(), actual.size() * sizeof(float));
            compare("attention512", actual, reference);
        }
        lotus_native::GpuModel gpu_vae(vae, context);
        {
            lotus_native::ImageTensor input{
                512, 8, 8, std::vector<float>(512 * 8 * 8)};
            for (std::size_t i = 0; i < input.values.size(); ++i) {
                input.values[i] =
                    std::sin(static_cast<float>(i) * 0.013f);
            }
            const std::string prefix = "decoder.mid_block.attentions.0";
            auto normalized = input;
            lotus_native::group_norm(
                vae, normalized, prefix + ".group_norm.weight",
                prefix + ".group_norm.bias");
            auto normalized_buffer = context.create_device_buffer(
                input.values.size() * sizeof(float));
            context.upload(
                normalized_buffer, input.values.data(),
                input.values.size() * sizeof(float));
            operators.group_norm(
                normalized_buffer,
                gpu_vae.tensor(prefix + ".group_norm.weight").buffer,
                gpu_vae.tensor(prefix + ".group_norm.bias").buffer,
                512, 64, 1.0e-6f);
            std::vector<float> normalized_actual(input.values.size());
            context.download(
                normalized_buffer, normalized_actual.data(),
                normalized_actual.size() * sizeof(float));
            compare(
                "group_norm", normalized_actual, normalized.values);
            lotus_native::TokenTensor normalized_tokens{
                64, 512, std::vector<float>(512 * 64)};
            for (std::uint32_t token = 0; token < 64; ++token) {
                for (std::uint32_t channel = 0; channel < 512; ++channel) {
                    normalized_tokens.values[token * 512 + channel] =
                        normalized.values[channel * 64 + token];
                }
            }
            const auto q_reference = lotus_native::linear(
                vae, normalized_tokens, prefix + ".to_q.weight",
                prefix + ".to_q.bias");
            auto token_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            auto q_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            operators.nchw_tokens(
                token_buffer, normalized_buffer, 64, 512, false);
            operators.linear(
                q_buffer, token_buffer,
                gpu_vae.tensor(prefix + ".to_q.weight").buffer,
                gpu_vae.tensor(prefix + ".to_q.bias").buffer,
                64, 512, 512, false);
            std::vector<float> q_actual(512 * 64);
            context.download(
                q_buffer, q_actual.data(), q_actual.size() * sizeof(float));
            compare("spatial_q", q_actual, q_reference.values);
            const auto k_reference = lotus_native::linear(
                vae, normalized_tokens, prefix + ".to_k.weight",
                prefix + ".to_k.bias");
            const auto v_reference = lotus_native::linear(
                vae, normalized_tokens, prefix + ".to_v.weight",
                prefix + ".to_v.bias");
            std::vector<float> attended_reference(512 * 64);
            std::vector<float> attention_scores(64);
            for (std::uint32_t query = 0; query < 64; ++query) {
                float maximum = -INFINITY;
                for (std::uint32_t key = 0; key < 64; ++key) {
                    float sum = 0.0f;
                    for (std::uint32_t d = 0; d < 512; ++d) {
                        sum += q_reference.values[query * 512 + d] *
                            k_reference.values[key * 512 + d];
                    }
                    attention_scores[key] = sum / std::sqrt(512.0f);
                    maximum = std::max(maximum, attention_scores[key]);
                }
                float denominator = 0.0f;
                for (float& score : attention_scores) {
                    score = std::exp(score - maximum);
                    denominator += score;
                }
                for (std::uint32_t d = 0; d < 512; ++d) {
                    float sum = 0.0f;
                    for (std::uint32_t key = 0; key < 64; ++key) {
                        sum += attention_scores[key] / denominator *
                            v_reference.values[key * 512 + d];
                    }
                    attended_reference[query * 512 + d] = sum;
                }
            }
            auto k_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            auto v_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            auto attended_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            operators.linear(
                k_buffer, token_buffer,
                gpu_vae.tensor(prefix + ".to_k.weight").buffer,
                gpu_vae.tensor(prefix + ".to_k.bias").buffer,
                64, 512, 512, false);
            operators.linear(
                v_buffer, token_buffer,
                gpu_vae.tensor(prefix + ".to_v.weight").buffer,
                gpu_vae.tensor(prefix + ".to_v.bias").buffer,
                64, 512, 512, false);
            operators.attention_separate(
                attended_buffer, q_buffer, k_buffer, v_buffer,
                64, 64, 1, 512);
            std::vector<float> attended_actual(512 * 64);
            context.download(
                attended_buffer, attended_actual.data(),
                attended_actual.size() * sizeof(float));
            compare(
                "spatial_attended",
                attended_actual, attended_reference);
            lotus_native::TokenTensor attended_tokens{
                64, 512, attended_reference};
            const auto projected_reference = lotus_native::linear(
                vae, attended_tokens, prefix + ".to_out.0.weight",
                prefix + ".to_out.0.bias");
            auto projected_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            operators.linear(
                projected_buffer, attended_buffer,
                gpu_vae.tensor(prefix + ".to_out.0.weight").buffer,
                gpu_vae.tensor(prefix + ".to_out.0.bias").buffer,
                64, 512, 512, false);
            std::vector<float> projected_actual(512 * 64);
            context.download(
                projected_buffer, projected_actual.data(),
                projected_actual.size() * sizeof(float));
            compare(
                "spatial_projected",
                projected_actual, projected_reference.values);
            auto spatial_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            auto original_buffer =
                context.create_device_buffer(512 * 64 * sizeof(float));
            context.upload(
                original_buffer, input.values.data(),
                input.values.size() * sizeof(float));
            operators.nchw_tokens(
                spatial_buffer, projected_buffer, 64, 512, true);
            operators.add(
                spatial_buffer, spatial_buffer, original_buffer,
                512 * 64);
            std::vector<float> spatial_actual(512 * 64);
            context.download(
                spatial_buffer, spatial_actual.data(),
                spatial_actual.size() * sizeof(float));
            std::vector<float> spatial_reference(input.values);
            for (std::uint32_t token = 0; token < 64; ++token) {
                for (std::uint32_t channel = 0; channel < 512; ++channel) {
                    spatial_reference[channel * 64 + token] +=
                        projected_reference.values[token * 512 + channel];
                }
            }
            compare(
                "spatial_layout", spatial_actual, spatial_reference);
            const auto reference =
                lotus_native::spatial_attention(vae, input, prefix);
            auto output = lotus_native::lotus_spatial_attention_gpu(
                context, gpu_vae, operators, input.values.data(),
                512, 8, 8, prefix);
            std::vector<float> actual(reference.values.size());
            context.download(
                output.buffer, actual.data(), actual.size() * sizeof(float));
            compare("spatial_attention", actual, reference.values);
        }
        {
            auto rgb = load(fixture / "rgb.bin", 3 * 64 * 64);
            auto output = lotus_native::lotus_vae_encode_gpu(
                context, gpu_vae, operators, rgb.data(), 64, 64);
            std::vector<float> actual(8 * 8 * 8);
            context.download(
                output.buffer, actual.data(), actual.size() * sizeof(float));
            actual.resize(4 * 8 * 8);
            compare(
                "encoder_mean", actual,
                load(fixture / "posterior_mean.bin", actual.size()));
        }
        lotus_native::GpuModel gpu_unet(unet, context);
        {
            auto rgb_latent = load(fixture / "rgb_latent.bin", 4 * 8 * 8);
            auto initial = load(fixture / "initial_latent.bin", 4 * 8 * 8);
            rgb_latent.insert(
                rgb_latent.end(), initial.begin(), initial.end());
            auto prompt =
                lotus_native::load_empty_prompt_cache(argv[2]);
            auto output = lotus_native::lotus_unet_gpu(
                context, gpu_unet, operators, prompt,
                rgb_latent.data(), 8, 8);
            std::vector<float> actual(4 * 8 * 8);
            context.download(
                output.buffer, actual.data(), actual.size() * sizeof(float));
            compare(
                "unet", actual,
                load(fixture / "unet_prediction.bin", actual.size()));
        }
        {
            auto prediction =
                load(fixture / "unet_prediction.bin", 4 * 8 * 8);
            for (float& value : prediction) value /= 0.18215f;
            auto output = lotus_native::lotus_vae_decode_gpu(
                context, gpu_vae, operators, prediction.data(), 8, 8);
            std::vector<float> actual(3 * 64 * 64);
            context.download(
                output.buffer, actual.data(), actual.size() * sizeof(float));
            compare(
                "decoder", actual,
                load(fixture / "decoded.bin", actual.size()));
        }
        return 0;
    } catch (const std::exception& error) {
        std::cerr << error.what() << "\n";
        return 1;
    }
}
