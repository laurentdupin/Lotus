#version 450 core
layout(local_size_x = 256) in;
layout(set = 0, binding = 0, std430) writeonly buffer Initial {
    float data[];
} initial_noise;
layout(set = 0, binding = 1, std430) writeonly buffer Posterior {
    float data[];
} posterior_noise;
layout(push_constant) uniform Parameters {
    uint count;
    uint seed_low;
    uint seed_high;
} p;
uint hash32(uint value) {
    value ^= value >> 16;
    value *= 0x7feb352du;
    value ^= value >> 15;
    value *= 0x846ca68bu;
    return value ^ (value >> 16);
}
float uniform_open(uint value) {
    return (float(hash32(value) >> 8) + 0.5) * (1.0 / 16777216.0);
}
float normal(uint index, uint stream) {
    uint key = index ^ p.seed_low ^ hash32(p.seed_high + stream);
    float u1 = uniform_open(key);
    float u2 = uniform_open(key ^ 0x9e3779b9u);
    return sqrt(-2.0 * log(u1)) * cos(6.28318530718 * u2);
}
void main() {
    uint i = gl_GlobalInvocationID.x;
    if (i >= p.count) return;
    initial_noise.data[i] = normal(i, 0x243f6a88u);
    posterior_noise.data[i] = normal(i, 0x85a308d3u);
}
