#pragma once
#ifdef __cplusplus
#include <cstdint>
#else
#include <stdint.h>
#endif

#ifdef __cplusplus
extern "C"
{
#endif

    // C API wrapper around C++ `NvOFWrapper` to allow calling from C sources.

    // Create an NvOF context for given width/height. Returns opaque handle or NULL on failure.
    void *nvof_create(int width, int height);

    // Destroy handle returned by `nvof_create`.
    void nvof_destroy(void *handle);

    // Compute optical flow between two 8-bit grayscale frames `frame0` and `frame1`.
    // `pitch0` and `pitch1` are bytes per row for the input frames.
    // `vx_out` and `vy_out` must point to preallocated int16_t arrays of size width*height.
    // Returns 0 on success, non-zero on failure.
    int nvof_compute(void *handle, const unsigned char *frame0, int pitch0, const unsigned char *frame1, int pitch1, int16_t *vx_out, int16_t *vy_out);

#ifdef __cplusplus
}
#endif
