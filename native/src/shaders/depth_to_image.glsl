#version 450 core
layout(local_size_x = 16, local_size_y = 16) in;
layout(r32f, set = 0, binding = 0) uniform writeonly image2D output_image;
layout(set = 0, binding = 1, std430) readonly buffer Depth {
    float data[];
} depth;
layout(push_constant) uniform Parameters { uint width; uint height; } p;
void main() {
    uint x = gl_GlobalInvocationID.x;
    uint y = gl_GlobalInvocationID.y;
    if (x >= p.width || y >= p.height) return;
    imageStore(output_image, ivec2(x, y), vec4(depth.data[y * p.width + x]));
}
