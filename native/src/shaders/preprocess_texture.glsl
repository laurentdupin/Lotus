#version 450 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(set = 0, binding = 0) uniform sampler2D source_image;
layout(set = 0, binding = 1, std430) writeonly buffer Output {
    float data[];
} output_buffer;
layout(push_constant) uniform Parameters {
    uint source_width;
    uint source_height;
    uint target_width;
    uint target_height;
} p;
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= p.target_width || y >= p.target_height) return;
    uint sx = min(p.source_width - 1, x * p.source_width / p.target_width);
    uint sy = min(p.source_height - 1, y * p.source_height / p.target_height);
    vec3 rgb = texelFetch(source_image, ivec2(sx, sy), 0).rgb;
    uint spatial = p.target_width * p.target_height;
    uint position = y * p.target_width + x;
    output_buffer.data[position] = rgb.b * 2.0 - 1.0;
    output_buffer.data[spatial + position] = rgb.g * 2.0 - 1.0;
    output_buffer.data[2 * spatial + position] = rgb.r * 2.0 - 1.0;
}
