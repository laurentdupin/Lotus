#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) buffer Depth { float data[]; } depth;
layout(set = 0, binding = 1, std430) readonly buffer Range {
    float data[];
} range_buffer;
layout(push_constant) uniform Parameters { uint count; } p;
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    float range = range_buffer.data[1] - range_buffer.data[0];
    depth.data[i] = range > 0.0
        ? (depth.data[i] - range_buffer.data[0]) / range
        : 0.0;
}
