#ifndef LOTUS_NATIVE_H
#define LOTUS_NATIVE_H

#include <stdint.h>

#if defined(_WIN32)
#  if defined(LOTUS_NATIVE_BUILD)
#    define LOTUS_API __declspec(dllexport)
#  else
#    define LOTUS_API __declspec(dllimport)
#  endif
#else
#  define LOTUS_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define LOTUS_NATIVE_ABI_VERSION 1u

typedef struct lotus_context lotus_context;

enum {
    LOTUS_OK = 0,
    LOTUS_INVALID_ARGUMENT = 1,
    LOTUS_MODEL_ERROR = 2,
    LOTUS_RUNTIME_ERROR = 3
};

LOTUS_API uint32_t lotus_native_abi_version(void);
LOTUS_API const char* lotus_last_error(void);

LOTUS_API int lotus_create(
    const char* snapshot_root_utf8,
    const char* empty_prompt_cache_utf8,
    lotus_context** output);
LOTUS_API void lotus_destroy(lotus_context* context);

/*
 * RGB is interleaved float32 in [0,1]. Depth contains width*height float32
 * values in [0,1]. Noise arrays are NCHW and contain
 * 4*floor(width/8)*floor(height/8) values each.
 */
LOTUS_API int lotus_infer_rgb_f32_with_noise(
    lotus_context* context,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    const float* initial_noise,
    const float* posterior_noise,
    float* depth);

LOTUS_API int lotus_infer_rgb_f32(
    lotus_context* context,
    const float* rgb,
    uint32_t width,
    uint32_t height,
    uint64_t seed,
    float* depth);

#ifdef __cplusplus
}
#endif

#endif
