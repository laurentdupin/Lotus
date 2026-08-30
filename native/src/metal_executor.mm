#include "metal_executor.h"
#include "inferbridge/native_harness_precision.h"

#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace lotus_native {
namespace {

MPSShape* shape(std::initializer_list<NSInteger> values) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:values.size()];
    for (NSInteger value : values) [result addObject:@(value)];
    return result;
}

MPSShape* shape(const TensorView& tensor) {
    NSMutableArray<NSNumber*>* result =
        [NSMutableArray arrayWithCapacity:tensor.rank];
    for (std::uint32_t i = 0; i < tensor.rank; ++i)
        [result addObject:@(tensor.dimensions[i])];
    return result;
}

NSString* ns(const std::string& value) {
    return [NSString stringWithUTF8String:value.c_str()];
}

int dimension(MPSGraphTensor* value, int index) {
    return [value.shape[index] intValue];
}

class GraphBuilder {
public:
    GraphBuilder(
        const ModelBundle& model,
        const TokenTensor& prompt,
        bool fp16,
        int width,
        int height)
        : unet_(model.unet()), vae_(model.vae()), prompt_(prompt),
          fp16_(fp16), width_(width), height_(height), graph_([MPSGraph new]) {}

    void build() {
        const int latent_width = width_ / 8;
        const int latent_height = height_ / 8;
        rgb_ = [graph_ placeholderWithShape:shape({1, 3, height_, width_})
                                      dataType:MPSDataTypeFloat32 name:@"rgb"];
        initial_noise_ = [graph_ placeholderWithShape:
            shape({1, 4, latent_height, latent_width})
            dataType:MPSDataTypeFloat32 name:@"initial_noise"];
        posterior_noise_ = [graph_ placeholderWithShape:
            shape({1, 4, latent_height, latent_width})
            dataType:MPSDataTypeFloat32 name:@"posterior_noise"];

        MPSGraphTensor* posterior = vae_encode(internal(rgb_));
        MPSGraphTensor* mean = [graph_ sliceTensor:posterior dimension:1
            start:0 length:4 name:nil];
        MPSGraphTensor* log_variance = [graph_ sliceTensor:posterior dimension:1
            start:4 length:4 name:nil];
        log_variance = [graph_ clampWithTensor:log_variance
            minValueTensor:scalar(-30.0f) maxValueTensor:scalar(20.0f) name:nil];
        MPSGraphTensor* deviation = [graph_ exponentWithTensor:
            multiply(log_variance, scalar(0.5f)) name:nil];
        MPSGraphTensor* latent = multiply(add(mean, multiply(
            deviation, internal(posterior_noise_))), scalar(0.18215f));

        const int input_channels = static_cast<int>(
            unet_.tensor("conv_in.weight").dimensions[1]);
        MPSGraphTensor* sample = latent;
        if (input_channels == 8) {
            sample = [graph_ concatTensors:@[latent, internal(initial_noise_)]
                                 dimension:1 name:nil];
        } else if (input_channels != 4) {
            throw std::runtime_error("unsupported Lotus UNet input channels");
        }
        MPSGraphTensor* prediction = unet(sample);
        prediction = multiply(prediction, scalar(1.0f / 0.18215f));
        decoded_ = external(vae_decode(prediction));
    }

    MPSGraph* graph() const { return graph_; }
    MPSGraphTensor* rgb() const { return rgb_; }
    MPSGraphTensor* initial_noise() const { return initial_noise_; }
    MPSGraphTensor* posterior_noise() const { return posterior_noise_; }
    MPSGraphTensor* decoded() const { return decoded_; }

private:
    MPSGraphTensor* internal(MPSGraphTensor* value) {
        return fp16_ && value.dataType != MPSDataTypeFloat16
            ? [graph_ castTensor:value toType:MPSDataTypeFloat16 name:nil]
            : value;
    }

    MPSGraphTensor* external(MPSGraphTensor* value) {
        return value.dataType == MPSDataTypeFloat32 ? value
            : [graph_ castTensor:value toType:MPSDataTypeFloat32 name:nil];
    }

    MPSGraphTensor* constant(
        const SafeTensors& model, const std::string& name) {
        const TensorView& tensor = model.tensor(name);
        if (fp16_) {
            auto& cache = &model == &unet_ ? unet_half_ : vae_half_;
            auto found = cache.find(name);
            if (found == cache.end()) {
                found = cache.emplace(
                    name, inferbridge::native::pack_fp16(
                        tensor.data, static_cast<std::size_t>(tensor.elements)))
                    .first;
            }
            NSData* half_data = [NSData dataWithBytesNoCopy:
                found->second.data()
                length:found->second.size() * sizeof(std::uint16_t)
                freeWhenDone:NO];
            return [graph_ constantWithData:half_data
                shape:shape(tensor) dataType:MPSDataTypeFloat16];
        }
        NSData* data = [NSData dataWithBytesNoCopy:
            const_cast<float*>(tensor.data)
            length:tensor.elements * sizeof(float) freeWhenDone:NO];
        return [graph_ constantWithData:data
            shape:shape(tensor) dataType:MPSDataTypeFloat32];
    }

    MPSGraphTensor* prompt_constant() {
        if (fp16_) {
            if (prompt_half_.empty())
                prompt_half_ = inferbridge::native::pack_fp16(
                    prompt_.values.data(), prompt_.values.size());
            NSData* data = [NSData dataWithBytesNoCopy:prompt_half_.data()
                length:prompt_half_.size() * sizeof(std::uint16_t)
                freeWhenDone:NO];
            return [graph_ constantWithData:data
                shape:shape({1, static_cast<NSInteger>(prompt_.tokens),
                             static_cast<NSInteger>(prompt_.dimensions)})
                dataType:MPSDataTypeFloat16];
        }
        NSData* data = [NSData dataWithBytesNoCopy:
            const_cast<float*>(prompt_.values.data())
            length:prompt_.values.size() * sizeof(float) freeWhenDone:NO];
        return internal([graph_ constantWithData:data
            shape:shape({1, static_cast<NSInteger>(prompt_.tokens),
                         static_cast<NSInteger>(prompt_.dimensions)})
            dataType:MPSDataTypeFloat32]);
    }

    MPSGraphTensor* scalar(float value) {
        return internal([graph_ constantWithScalar:value
            dataType:MPSDataTypeFloat32]);
    }

    MPSGraphTensor* add(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ additionWithPrimaryTensor:a secondaryTensor:b name:nil];
    }

    MPSGraphTensor* multiply(MPSGraphTensor* a, MPSGraphTensor* b) {
        return [graph_ multiplicationWithPrimaryTensor:a secondaryTensor:b
            name:nil];
    }

    MPSGraphConvolution2DOpDescriptor* descriptor(
        int stride, int before, int after) {
        return [MPSGraphConvolution2DOpDescriptor
            descriptorWithStrideInX:stride strideInY:stride
            dilationRateInX:1 dilationRateInY:1 groups:1
            paddingLeft:before paddingRight:after
            paddingTop:before paddingBottom:after
            paddingStyle:MPSGraphPaddingStyleExplicit
            dataLayout:MPSGraphTensorNamedDataLayoutNCHW
            weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    }

    MPSGraphTensor* conv(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix,
        int stride = 1,
        int before = 1,
        int after = 1) {
        MPSGraphTensor* result = [graph_ convolution2DWithSourceTensor:value
            weightsTensor:constant(model, prefix + ".weight")
            descriptor:descriptor(stride, before, after) name:ns(prefix)];
        if (model.contains(prefix + ".bias")) {
            const int channels = static_cast<int>(
                model.tensor(prefix + ".bias").elements);
            result = add(result, [graph_ reshapeTensor:
                constant(model, prefix + ".bias")
                withShape:shape({1, channels, 1, 1}) name:nil]);
        }
        return result;
    }

    MPSGraphTensor* linear(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix) {
        MPSGraphTensor* weight = [graph_ transposeTensor:
            constant(model, prefix + ".weight")
            dimension:0 withDimension:1 name:nil];
        MPSGraphTensor* result = [graph_
            matrixMultiplicationWithPrimaryTensor:value
            secondaryTensor:weight name:ns(prefix)];
        if (model.contains(prefix + ".bias"))
            result = add(result, constant(model, prefix + ".bias"));
        return result;
    }

    MPSGraphTensor* silu(MPSGraphTensor* value) {
        return multiply(value, [graph_ sigmoidWithTensor:value name:nil]);
    }

    MPSGraphTensor* gelu(MPSGraphTensor* value) {
        MPSGraphTensor* error = [graph_ erfWithTensor:
            multiply(value, scalar(0.7071067811865475f)) name:nil];
        return multiply(multiply(value, scalar(0.5f)),
                        add(error, scalar(1.0f)));
    }

    MPSGraphTensor* group_norm(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix,
        float epsilon) {
        const int channels = dimension(value, 1);
        const int height = dimension(value, 2);
        const int width = dimension(value, 3);
        MPSGraphTensor* grouped = [graph_ reshapeTensor:value
            withShape:shape({1, 32, channels / 32, height, width}) name:nil];
        NSArray<NSNumber*>* axes = @[@2, @3, @4];
        MPSGraphTensor* mean = [graph_ meanOfTensor:grouped axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:grouped
            meanTensor:mean axes:axes name:nil];
        MPSGraphTensor* normalized = [graph_ normalizationWithTensor:grouped
            meanTensor:mean varianceTensor:variance gammaTensor:nil
            betaTensor:nil epsilon:epsilon name:ns(prefix)];
        normalized = [graph_ reshapeTensor:normalized
            withShape:shape({1, channels, height, width}) name:nil];
        MPSGraphTensor* gamma = [graph_ reshapeTensor:
            constant(model, prefix + ".weight")
            withShape:shape({1, channels, 1, 1}) name:nil];
        MPSGraphTensor* beta = [graph_ reshapeTensor:
            constant(model, prefix + ".bias")
            withShape:shape({1, channels, 1, 1}) name:nil];
        return add(multiply(normalized, gamma), beta);
    }

    MPSGraphTensor* layer_norm(
        const SafeTensors& model,
        MPSGraphTensor* value,
        const std::string& prefix) {
        NSArray<NSNumber*>* axes = @[@(-1)];
        MPSGraphTensor* mean = [graph_ meanOfTensor:value axes:axes name:nil];
        MPSGraphTensor* variance = [graph_ varianceOfTensor:value
            meanTensor:mean axes:axes name:nil];
        return [graph_ normalizationWithTensor:value meanTensor:mean
            varianceTensor:variance gammaTensor:constant(model, prefix + ".weight")
            betaTensor:constant(model, prefix + ".bias") epsilon:1.0e-5f
            name:ns(prefix)];
    }

    MPSGraphTensor* nearest(
        MPSGraphTensor* value, int target_height, int target_width) {
        const int source_height = dimension(value, 2);
        const int source_width = dimension(value, 3);
        std::vector<std::int32_t> y(static_cast<std::size_t>(target_height));
        std::vector<std::int32_t> x(static_cast<std::size_t>(target_width));
        for (int i = 0; i < target_height; ++i)
            y[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>(
                    std::int64_t(i) * source_height / target_height);
        for (int i = 0; i < target_width; ++i)
            x[static_cast<std::size_t>(i)] =
                static_cast<std::int32_t>(
                    std::int64_t(i) * source_width / target_width);
        NSData* y_data = [NSData dataWithBytes:y.data()
            length:y.size() * sizeof(std::int32_t)];
        NSData* x_data = [NSData dataWithBytes:x.data()
            length:x.size() * sizeof(std::int32_t)];
        MPSGraphTensor* y_indices = [graph_ constantWithData:y_data
            shape:shape({target_height}) dataType:MPSDataTypeInt32];
        MPSGraphTensor* x_indices = [graph_ constantWithData:x_data
            shape:shape({target_width}) dataType:MPSDataTypeInt32];
        value = [graph_ gatherWithUpdatesTensor:value indicesTensor:y_indices
            axis:2 batchDimensions:0 name:nil];
        return [graph_ gatherWithUpdatesTensor:value indicesTensor:x_indices
            axis:3 batchDimensions:0 name:nil];
    }

    MPSGraphTensor* resnet(
        const SafeTensors& model,
        MPSGraphTensor* input,
        const std::string& prefix) {
        MPSGraphTensor* hidden = silu(group_norm(
            model, input, prefix + ".norm1", 1.0e-6f));
        hidden = conv(model, hidden, prefix + ".conv1");
        hidden = silu(group_norm(
            model, hidden, prefix + ".norm2", 1.0e-6f));
        hidden = conv(model, hidden, prefix + ".conv2");
        MPSGraphTensor* residual = input;
        if (model.contains(prefix + ".conv_shortcut.weight"))
            residual = conv(model, input, prefix + ".conv_shortcut", 1, 0, 0);
        return add(hidden, residual);
    }

    MPSGraphTensor* spatial_attention(
        MPSGraphTensor* input, const std::string& prefix) {
        const int channels = dimension(input, 1);
        const int height = dimension(input, 2);
        const int width = dimension(input, 3);
        const int tokens = height * width;
        MPSGraphTensor* normalized = group_norm(
            vae_, input, prefix + ".group_norm", 1.0e-6f);
        normalized = [graph_ reshapeTensor:normalized
            withShape:shape({1, channels, tokens}) name:nil];
        normalized = [graph_ transposeTensor:normalized dimension:1
            withDimension:2 name:nil];
        MPSGraphTensor* q = linear(vae_, normalized, prefix + ".to_q");
        MPSGraphTensor* k = linear(vae_, normalized, prefix + ".to_k");
        MPSGraphTensor* v = linear(vae_, normalized, prefix + ".to_v");
        k = [graph_ transposeTensor:k dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* scores = [graph_
            matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
        scores = multiply(scores, scalar(1.0f / std::sqrt(float(channels))));
        scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
        MPSGraphTensor* attended = [graph_
            matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v
            name:nil];
        attended = linear(vae_, attended, prefix + ".to_out.0");
        attended = [graph_ transposeTensor:attended dimension:1
            withDimension:2 name:nil];
        attended = [graph_ reshapeTensor:attended
            withShape:shape({1, channels, height, width}) name:nil];
        return add(input, attended);
    }

    MPSGraphTensor* vae_mid(MPSGraphTensor* hidden, const std::string& prefix) {
        hidden = resnet(vae_, hidden, prefix + ".resnets.0");
        hidden = spatial_attention(hidden, prefix + ".attentions.0");
        return resnet(vae_, hidden, prefix + ".resnets.1");
    }

    MPSGraphTensor* vae_encode(MPSGraphTensor* image) {
        MPSGraphTensor* hidden = conv(vae_, image, "encoder.conv_in");
        for (int block = 0; block < 4; ++block) {
            const std::string root = "encoder.down_blocks." +
                std::to_string(block);
            for (int layer = 0; layer < 2; ++layer)
                hidden = resnet(vae_, hidden, root + ".resnets." +
                    std::to_string(layer));
            if (block != 3)
                hidden = conv(vae_, hidden, root + ".downsamplers.0.conv",
                    2, 0, 1);
        }
        hidden = vae_mid(hidden, "encoder.mid_block");
        hidden = silu(group_norm(vae_, hidden,
            "encoder.conv_norm_out", 1.0e-6f));
        hidden = conv(vae_, hidden, "encoder.conv_out");
        return conv(vae_, hidden, "quant_conv", 1, 0, 0);
    }

    MPSGraphTensor* vae_decode(MPSGraphTensor* latent) {
        MPSGraphTensor* hidden = conv(
            vae_, latent, "post_quant_conv", 1, 0, 0);
        hidden = conv(vae_, hidden, "decoder.conv_in");
        hidden = vae_mid(hidden, "decoder.mid_block");
        for (int block = 0; block < 4; ++block) {
            const std::string root = "decoder.up_blocks." +
                std::to_string(block);
            for (int layer = 0; layer < 3; ++layer)
                hidden = resnet(vae_, hidden, root + ".resnets." +
                    std::to_string(layer));
            if (block != 3) {
                hidden = nearest(hidden, dimension(hidden, 2) * 2,
                    dimension(hidden, 3) * 2);
                hidden = conv(vae_, hidden, root + ".upsamplers.0.conv");
            }
        }
        hidden = silu(group_norm(vae_, hidden,
            "decoder.conv_norm_out", 1.0e-6f));
        return conv(vae_, hidden, "decoder.conv_out");
    }

    MPSGraphTensor* attention(
        MPSGraphTensor* query_input,
        MPSGraphTensor* key_value_input,
        const std::string& prefix,
        int heads) {
        const int query_tokens = dimension(query_input, 1);
        const int key_tokens = dimension(key_value_input, 1);
        MPSGraphTensor* q = linear(unet_, query_input, prefix + ".to_q");
        MPSGraphTensor* k = linear(unet_, key_value_input, prefix + ".to_k");
        MPSGraphTensor* v = linear(unet_, key_value_input, prefix + ".to_v");
        const int channels = dimension(q, 2);
        const int head_dimensions = channels / heads;
        q = [graph_ reshapeTensor:q
            withShape:shape({1, query_tokens, heads, head_dimensions}) name:nil];
        k = [graph_ reshapeTensor:k
            withShape:shape({1, key_tokens, heads, head_dimensions}) name:nil];
        v = [graph_ reshapeTensor:v
            withShape:shape({1, key_tokens, heads, head_dimensions}) name:nil];
        q = [graph_ transposeTensor:q permutation:@[@0, @2, @1, @3] name:nil];
        k = [graph_ transposeTensor:k permutation:@[@0, @2, @3, @1] name:nil];
        v = [graph_ transposeTensor:v permutation:@[@0, @2, @1, @3] name:nil];
        MPSGraphTensor* scores = [graph_
            matrixMultiplicationWithPrimaryTensor:q secondaryTensor:k name:nil];
        scores = multiply(scores,
            scalar(1.0f / std::sqrt(float(head_dimensions))));
        scores = [graph_ softMaxWithTensor:scores axis:-1 name:nil];
        MPSGraphTensor* attended = [graph_
            matrixMultiplicationWithPrimaryTensor:scores secondaryTensor:v
            name:nil];
        attended = [graph_ transposeTensor:attended
            permutation:@[@0, @2, @1, @3] name:nil];
        attended = [graph_ reshapeTensor:attended
            withShape:shape({1, query_tokens, channels}) name:nil];
        return linear(unet_, attended, prefix + ".to_out.0");
    }

    MPSGraphTensor* feed_forward(
        MPSGraphTensor* input, const std::string& prefix) {
        MPSGraphTensor* projected = linear(
            unet_, input, prefix + ".net.0.proj");
        const int dimensions = dimension(projected, 2) / 2;
        MPSGraphTensor* value = [graph_ sliceTensor:projected dimension:2
            start:0 length:dimensions name:nil];
        MPSGraphTensor* gate = [graph_ sliceTensor:projected dimension:2
            start:dimensions length:dimensions name:nil];
        return linear(unet_, multiply(value, gelu(gate)), prefix + ".net.2");
    }

    MPSGraphTensor* transformer(
        MPSGraphTensor* input,
        const std::string& prefix,
        int heads) {
        const int channels = dimension(input, 1);
        const int height = dimension(input, 2);
        const int width = dimension(input, 3);
        const int tokens = height * width;
        MPSGraphTensor* normalized = group_norm(
            unet_, input, prefix + ".norm", 1.0e-6f);
        normalized = [graph_ reshapeTensor:normalized
            withShape:shape({1, channels, tokens}) name:nil];
        normalized = [graph_ transposeTensor:normalized
            dimension:1 withDimension:2 name:nil];
        MPSGraphTensor* current = linear(unet_, normalized, prefix + ".proj_in");
        const std::string block = prefix + ".transformer_blocks.0";
        MPSGraphTensor* normalized_tokens =
            layer_norm(unet_, current, block + ".norm1");
        current = add(current, attention(normalized_tokens, normalized_tokens,
            block + ".attn1", heads));
        current = add(current, attention(
            layer_norm(unet_, current, block + ".norm2"),
            prompt_constant(), block + ".attn2", heads));
        current = add(current, feed_forward(
            layer_norm(unet_, current, block + ".norm3"), block + ".ff"));
        current = linear(unet_, current, prefix + ".proj_out");
        current = [graph_ transposeTensor:current
            dimension:1 withDimension:2 name:nil];
        current = [graph_ reshapeTensor:current
            withShape:shape({1, channels, height, width}) name:nil];
        return add(input, current);
    }

    MPSGraphTensor* embedding_mlp(
        MPSGraphTensor* input, const std::string& prefix) {
        return linear(unet_, silu(linear(
            unet_, input, prefix + ".linear_1")), prefix + ".linear_2");
    }

    MPSGraphTensor* time_embedding() {
        std::vector<float> values(320);
        for (int i = 0; i < 160; ++i) {
            const float frequency = std::exp(
                -std::log(10000.0f) * static_cast<float>(i) / 160.0f);
            const float argument = 999.0f * frequency;
            values[i] = std::cos(argument);
            values[160 + i] = std::sin(argument);
        }
        NSData* data = [NSData dataWithBytes:values.data()
            length:values.size() * sizeof(float)];
        MPSGraphTensor* timestep = internal([graph_ constantWithData:data
            shape:shape({1, 320}) dataType:MPSDataTypeFloat32]);
        const float labels[4] = {
            std::sin(1.0f), 0.0f, std::cos(1.0f), 1.0f};
        data = [NSData dataWithBytes:labels length:sizeof(labels)];
        MPSGraphTensor* task = internal([graph_ constantWithData:data
            shape:shape({1, 4}) dataType:MPSDataTypeFloat32]);
        return add(embedding_mlp(timestep, "time_embedding"),
                   embedding_mlp(task, "class_embedding"));
    }

    MPSGraphTensor* unet_resnet(
        MPSGraphTensor* input,
        MPSGraphTensor* time,
        const std::string& prefix) {
        MPSGraphTensor* hidden = silu(group_norm(
            unet_, input, prefix + ".norm1", 1.0e-5f));
        hidden = conv(unet_, hidden, prefix + ".conv1");
        MPSGraphTensor* projected = linear(
            unet_, silu(time), prefix + ".time_emb_proj");
        projected = [graph_ reshapeTensor:projected
            withShape:shape({1, dimension(hidden, 1), 1, 1}) name:nil];
        hidden = add(hidden, projected);
        hidden = silu(group_norm(
            unet_, hidden, prefix + ".norm2", 1.0e-5f));
        hidden = conv(unet_, hidden, prefix + ".conv2");
        MPSGraphTensor* residual = input;
        if (unet_.contains(prefix + ".conv_shortcut.weight"))
            residual = conv(unet_, input, prefix + ".conv_shortcut", 1, 0, 0);
        return add(hidden, residual);
    }

    MPSGraphTensor* unet(MPSGraphTensor* sample) {
        MPSGraphTensor* time = time_embedding();
        MPSGraphTensor* hidden = conv(unet_, sample, "conv_in");
        std::vector<MPSGraphTensor*> skips;
        skips.push_back(hidden);
        constexpr int down_heads[3] = {5, 10, 20};
        for (int block = 0; block < 4; ++block) {
            const std::string root = "down_blocks." + std::to_string(block);
            for (int layer = 0; layer < 2; ++layer) {
                hidden = unet_resnet(hidden, time,
                    root + ".resnets." + std::to_string(layer));
                if (block < 3)
                    hidden = transformer(hidden,
                        root + ".attentions." + std::to_string(layer),
                        down_heads[block]);
                skips.push_back(hidden);
            }
            if (block != 3) {
                hidden = conv(unet_, hidden, root + ".downsamplers.0.conv",
                    2, 1, 1);
                skips.push_back(hidden);
            }
        }
        hidden = unet_resnet(hidden, time, "mid_block.resnets.0");
        hidden = transformer(hidden, "mid_block.attentions.0", 20);
        hidden = unet_resnet(hidden, time, "mid_block.resnets.1");

        constexpr int up_heads[4] = {0, 20, 10, 5};
        for (int block = 0; block < 4; ++block) {
            const std::string root = "up_blocks." + std::to_string(block);
            for (int layer = 0; layer < 3; ++layer) {
                MPSGraphTensor* skip = skips.back();
                skips.pop_back();
                hidden = [graph_ concatTensors:@[hidden, skip]
                                     dimension:1 name:nil];
                hidden = unet_resnet(hidden, time,
                    root + ".resnets." + std::to_string(layer));
                if (block != 0)
                    hidden = transformer(hidden,
                        root + ".attentions." + std::to_string(layer),
                        up_heads[block]);
            }
            if (block != 3) {
                hidden = nearest(hidden, dimension(skips.back(), 2),
                    dimension(skips.back(), 3));
                hidden = conv(unet_, hidden, root + ".upsamplers.0.conv");
            }
        }
        hidden = silu(group_norm(unet_, hidden, "conv_norm_out", 1.0e-5f));
        return conv(unet_, hidden, "conv_out");
    }

    const SafeTensors& unet_;
    const SafeTensors& vae_;
    const TokenTensor& prompt_;
    bool fp16_;
    int width_;
    int height_;
    MPSGraph* graph_;
    MPSGraphTensor* rgb_ = nil;
    MPSGraphTensor* initial_noise_ = nil;
    MPSGraphTensor* posterior_noise_ = nil;
    MPSGraphTensor* decoded_ = nil;
    std::unordered_map<std::string, std::vector<std::uint16_t>> unet_half_;
    std::unordered_map<std::string, std::vector<std::uint16_t>> vae_half_;
    std::vector<std::uint16_t> prompt_half_;
};

struct PlanKey {
    int width;
    int height;
    bool operator==(const PlanKey& other) const {
        return width == other.width && height == other.height;
    }
};

struct PlanHash {
    std::size_t operator()(const PlanKey& key) const {
        return static_cast<std::size_t>(key.width) * 65537u + key.height;
    }
};

struct Plan {
    MPSGraph* graph = nil;
    MPSGraphExecutable* executable = nil;
};

}  // namespace

class MetalExecutor::Impl {
public:
    Impl(const ModelBundle& model, const TokenTensor& prompt)
        : model_(model), prompt_(prompt) {
        const auto precision = inferbridge::native::requested_precision();
        if (precision == inferbridge::native::Precision::int8)
            throw std::invalid_argument("Lotus Metal does not support INT8 yet");
        fp16_ = precision == inferbridge::native::Precision::fp16 ||
            precision == inferbridge::native::Precision::automatic;
        device_ = MTLCreateSystemDefaultDevice();
        if (device_ == nil) throw std::runtime_error("Metal is unavailable");
        queue_ = [device_ newCommandQueue];
        graph_device_ = [MPSGraphDevice deviceWithMTLDevice:device_];
        if (queue_ == nil || graph_device_ == nil)
            throw std::runtime_error("could not initialize Lotus Metal");
    }

    ImageTensor infer(
        const float* rgb,
        std::uint32_t width,
        std::uint32_t height,
        const float* initial_noise,
        const float* posterior_noise) {
        std::vector<float> image(
            static_cast<std::size_t>(3) * width * height);
        for (std::uint32_t y = 0; y < height; ++y)
            for (std::uint32_t x = 0; x < width; ++x)
                for (std::uint32_t c = 0; c < 3; ++c)
                    image[(static_cast<std::size_t>(c) * height + y) * width + x] =
                        rgb[(static_cast<std::size_t>(y) * width + x) * 3 + c] *
                            2.0f - 1.0f;
        const std::size_t latent_elements =
            static_cast<std::size_t>(4) * (width / 8) * (height / 8);

        std::lock_guard<std::mutex> guard(mutex_);
        @autoreleasepool {
            const Plan& plan = get_plan(width, height);
            id<MTLBuffer> rgb_buffer = [device_ newBufferWithBytes:image.data()
                length:image.size() * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> initial_buffer = [device_ newBufferWithBytes:initial_noise
                length:latent_elements * sizeof(float)
                options:MTLResourceStorageModeShared];
            id<MTLBuffer> posterior_buffer = [device_ newBufferWithBytes:posterior_noise
                length:latent_elements * sizeof(float)
                options:MTLResourceStorageModeShared];
            if (rgb_buffer == nil || initial_buffer == nil || posterior_buffer == nil)
                throw std::bad_alloc();
            NSArray<MPSGraphTensorData*>* inputs = @[
                [[MPSGraphTensorData alloc] initWithMTLBuffer:rgb_buffer
                    shape:shape({1, 3, static_cast<NSInteger>(height),
                                 static_cast<NSInteger>(width)})
                    dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:initial_buffer
                    shape:shape({1, 4, static_cast<NSInteger>(height / 8),
                                 static_cast<NSInteger>(width / 8)})
                    dataType:MPSDataTypeFloat32],
                [[MPSGraphTensorData alloc] initWithMTLBuffer:posterior_buffer
                    shape:shape({1, 4, static_cast<NSInteger>(height / 8),
                                 static_cast<NSInteger>(width / 8)})
                    dataType:MPSDataTypeFloat32]
            ];
            MPSGraphExecutableExecutionDescriptor* execution =
                [MPSGraphExecutableExecutionDescriptor new];
            execution.waitUntilCompleted = YES;
            NSArray<MPSGraphTensorData*>* results = [plan.executable
                runWithMTLCommandQueue:queue_ inputsArray:inputs
                resultsArray:nil executionDescriptor:execution];
            if (results.count != 1)
                throw std::runtime_error("Lotus Metal returned incomplete output");
            const std::uint32_t decoded_width = width / 8 * 8;
            const std::uint32_t decoded_height = height / 8 * 8;
            ImageTensor output{3, decoded_height, decoded_width,
                std::vector<float>(static_cast<std::size_t>(3) *
                    decoded_width * decoded_height)};
            [results[0].mpsndarray readBytes:output.values.data() strideBytes:nil];
            return output;
        }
    }

private:
    const Plan& get_plan(std::uint32_t width, std::uint32_t height) {
        const PlanKey key{static_cast<int>(width), static_cast<int>(height)};
        auto found = plans_.find(key);
        if (found != plans_.end()) return found->second;
        MPSGraphCompilationDescriptor* descriptor =
            [MPSGraphCompilationDescriptor new];
        descriptor.optimizationLevel = MPSGraphOptimizationLevel0;
        descriptor.waitForCompilationCompletion = YES;
        NSURL* package = cache_url(key);
        MPSGraphExecutable* executable = nil;
        if (@available(macOS 14.0, *)) {
            if (package != nil && [[NSFileManager defaultManager]
                    fileExistsAtPath:package.path]) {
                @try {
                    executable = [[MPSGraphExecutable alloc]
                        initWithMPSGraphPackageAtURL:package
                        compilationDescriptor:descriptor];
                } @catch (NSException*) {
                    [[NSFileManager defaultManager]
                        removeItemAtURL:package error:nil];
                    executable = nil;
                }
            }
        }
        MPSGraph* graph = nil;
        if (executable == nil) {
            GraphBuilder builder(model_, prompt_, fp16_, key.width, key.height);
            builder.build();
            graph = builder.graph();
            NSMutableDictionary<MPSGraphTensor*, MPSGraphShapedType*>* feeds =
                [NSMutableDictionary dictionary];
            feeds[builder.rgb()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 3, key.height, key.width})
                dataType:MPSDataTypeFloat32];
            feeds[builder.initial_noise()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 4, key.height / 8, key.width / 8})
                dataType:MPSDataTypeFloat32];
            feeds[builder.posterior_noise()] = [[MPSGraphShapedType alloc]
                initWithShape:shape({1, 4, key.height / 8, key.width / 8})
                dataType:MPSDataTypeFloat32];
            executable = [builder.graph()
                compileWithDevice:graph_device_ feeds:feeds
                targetTensors:@[builder.decoded()] targetOperations:nil
                compilationDescriptor:descriptor];
            if (@available(macOS 14.0, *)) {
                if (executable != nil && package != nil) {
                    @try {
                        [executable serializeToMPSGraphPackageAtURL:package
                                                         descriptor:nil];
                    } @catch (NSException*) {
                        [[NSFileManager defaultManager]
                            removeItemAtURL:package error:nil];
                    }
                }
            }
        }
        if (executable == nil)
            throw std::runtime_error("failed to compile Lotus Metal graph");
        executable.options = MPSGraphOptionsSynchronizeResults;
        return plans_.emplace(key, Plan{graph, executable})
            .first->second;
    }

    NSURL* cache_url(const PlanKey& key) const {
        if (@available(macOS 14.0, *)) {
            NSArray<NSString*>* directories =
                NSSearchPathForDirectoriesInDomains(
                    NSCachesDirectory, NSUserDomainMask, YES);
            if (directories.count == 0) return nil;
            NSString* directory = [directories.firstObject
                stringByAppendingPathComponent:
                    @"DepthExtractor/LotusMetalGraphCache-v1"];
            if (![[NSFileManager defaultManager]
                    createDirectoryAtPath:directory
                    withIntermediateDirectories:YES attributes:nil error:nil])
                return nil;
            const NSOperatingSystemVersion os =
                NSProcessInfo.processInfo.operatingSystemVersion;
            const int input_channels = static_cast<int>(
                model_.unet().tensor("conv_in.weight").dimensions[1]);
            const char* model_identity = input_channels == 8
                ? "41a8d50a989a757016251e170f39b6ac90e1ff324a90ae88c9a762ec549d0f34"
                : "66f32e128f0f85a6f2d72893f56c663f8abde5a76678180b0eb51b1f3ed44899";
            const std::string name =
                std::string(model_identity) + "-" +
                std::to_string(key.width) + "x" +
                std::to_string(key.height) + "-" +
                (fp16_ ? "fp16" : "fp32") + "-" +
                std::to_string(device_.registryID) + "-macos" +
                std::to_string(os.majorVersion) + "." +
                std::to_string(os.minorVersion) + ".mpsgraphpackage";
            return [NSURL fileURLWithPath:[directory
                stringByAppendingPathComponent:ns(name)]];
        }
        return nil;
    }

    const ModelBundle& model_;
    const TokenTensor& prompt_;
    bool fp16_ = false;
    id<MTLDevice> device_ = nil;
    id<MTLCommandQueue> queue_ = nil;
    MPSGraphDevice* graph_device_ = nil;
    std::unordered_map<PlanKey, Plan, PlanHash> plans_;
    std::mutex mutex_;
};

MetalExecutor::MetalExecutor(
    const ModelBundle& model, const TokenTensor& prompt)
    : impl_(std::make_unique<Impl>(model, prompt)) {}

MetalExecutor::~MetalExecutor() = default;

ImageTensor MetalExecutor::infer(
    const float* rgb,
    std::uint32_t width,
    std::uint32_t height,
    const float* initial_noise,
    const float* posterior_noise) {
    return impl_->infer(
        rgb, width, height, initial_noise, posterior_noise);
}

}  // namespace lotus_native
