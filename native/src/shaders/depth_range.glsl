#version 450 core
layout(local_size_x = 1) in;
layout(set = 0, binding = 0, std430) readonly buffer Depth {
    float data[];
} depth;
layout(set = 0, binding = 1, std430) writeonly buffer Range {
    float data[];
} range_buffer;
layout(push_constant) uniform Parameters { uint count; } p;
void main() {
    if (p.count == 0) {
        range_buffer.data[0] = 0.0;
        range_buffer.data[1] = 0.0;
        return;
    }
    float minimum = depth.data[0];
    float maximum = depth.data[0];
    for (uint i = 1; i < p.count; ++i) {
        minimum = min(minimum, depth.data[i]);
        maximum = max(maximum, depth.data[i]);
    }
    range_buffer.data[0] = minimum;
    range_buffer.data[1] = maximum;
}
