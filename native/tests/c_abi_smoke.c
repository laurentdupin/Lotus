#include "lotus_native.h"

int main(void) {
    lotus_context* context = 0;
    uint32_t width = 0;
    uint32_t height = 0;
    if (lotus_native_abi_version() != LOTUS_NATIVE_ABI_VERSION) {
        return 1;
    }
    if (lotus_create(0, 0, &context) != LOTUS_INVALID_ARGUMENT) {
        return 2;
    }
    if (lotus_inferbridge_image_shape(53, 41, &width, &height) != LOTUS_OK ||
        width != 344 || height != 264) {
        return 3;
    }
    if (lotus_inferbridge_image_shape(0, 41, &width, &height) !=
        LOTUS_INVALID_ARGUMENT) {
        return 4;
    }
    lotus_destroy(context);
    return 0;
}
