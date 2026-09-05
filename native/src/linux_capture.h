#pragma once
#if defined(__linux__) && !defined(__ANDROID__)
#include <inferbridge/linux_capture_vulkan.h>
struct lotus_context;
ibr_linux_capture_capabilities
lotus_linux_capture_capabilities(lotus_context *);
void lotus_infer_linux_capture(
    lotus_context *, const inferbridge::linux_capture::LinuxDmaBufImage &,
    uint64_t, float *);
#endif
