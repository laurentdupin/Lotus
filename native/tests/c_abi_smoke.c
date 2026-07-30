#include "lotus_native.h"

int main(void) {
    lotus_context* context = 0;
    if (lotus_native_abi_version() != LOTUS_NATIVE_ABI_VERSION) {
        return 1;
    }
    if (lotus_create(0, 0, &context) != LOTUS_INVALID_ARGUMENT) {
        return 2;
    }
    lotus_destroy(context);
    return 0;
}
