#include "prompt_cache.h"

#include <array>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <stdexcept>

namespace lotus_native {
namespace {

constexpr std::array<char, 8> magic = {
    'L', 'O', 'T', 'U', 'S', 'P', '0', '1'};
constexpr std::uint32_t version = 1;
constexpr std::uint32_t header_bytes = 512;
constexpr std::uint32_t tokens = 77;
constexpr std::uint32_t dimensions = 1024;
constexpr const char revision[] =
    "d8f1b0835745cb1ef5c8b89528c8d37bc764a37b";
constexpr const char unet_hash[] =
    "41a8d50a989a757016251e170f39b6ac90e1ff324a90ae88c9a762ec549d0f34";
constexpr const char vae_hash[] =
    "fc436097e5cbb105c44df4403ae84acf7e81de1f64acfedc334888e9a8eea223";
constexpr const char text_hash[] =
    "8a0859e9385019944df829693ffdef5bbde394950ff561cbfdd61fbb3ea76fb5";

std::uint32_t u32(const char* data) {
    return std::uint32_t(static_cast<unsigned char>(data[0])) |
        (std::uint32_t(static_cast<unsigned char>(data[1])) << 8) |
        (std::uint32_t(static_cast<unsigned char>(data[2])) << 16) |
        (std::uint32_t(static_cast<unsigned char>(data[3])) << 24);
}

void expect_string(
    const char* actual,
    std::size_t bytes,
    const char* expected,
    const char* field) {
    if (std::strlen(expected) + 1 > bytes ||
        std::strncmp(actual, expected, bytes) != 0) {
        throw std::runtime_error(
            std::string("Lotus prompt cache ") + field + " mismatch");
    }
}

}  // namespace

TokenTensor load_empty_prompt_cache(const std::string& path) {
    std::ifstream stream(
        std::filesystem::u8path(path), std::ios::binary);
    std::array<char, header_bytes> header{};
    stream.read(header.data(), header.size());
    if (!stream ||
        !std::equal(magic.begin(), magic.end(), header.begin()) ||
        u32(header.data() + 8) != version ||
        u32(header.data() + 12) != header_bytes ||
        u32(header.data() + 16) != tokens ||
        u32(header.data() + 20) != dimensions) {
        throw std::runtime_error("invalid Lotus prompt cache header");
    }
    expect_string(header.data() + 24, 41, revision, "revision");
    expect_string(header.data() + 65, 65, unet_hash, "UNet hash");
    expect_string(header.data() + 130, 65, vae_hash, "VAE hash");
    expect_string(header.data() + 195, 65, text_hash, "text hash");

    TokenTensor prompt{
        tokens, dimensions,
        std::vector<float>(std::uint64_t(tokens) * dimensions)};
    stream.read(
        reinterpret_cast<char*>(prompt.values.data()),
        static_cast<std::streamsize>(
            prompt.values.size() * sizeof(float)));
    if (!stream || stream.peek() != std::char_traits<char>::eof()) {
        throw std::runtime_error("invalid Lotus prompt cache payload");
    }
    return prompt;
}

}  // namespace lotus_native
