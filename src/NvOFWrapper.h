#pragma once

// CUDA headers are only required when the NvOF SDK is enabled. Provide
// a lightweight fallback so this header can be included even when the
// SDK / CUDA headers are not installed on the build machine.
#ifdef NV_OF_SDK
#include <cuda_runtime.h>
#endif

// Lightweight wrapper header for optional NVIDIA Optical Flow support.
// The implementation is guarded by `NV_OF_SDK` so the project can still
// compile without the SDK; define `nv_of_sdk` meson option pointing to the
// SDK root to enable it.

class NvOFWrapper
{
public:
    NvOFWrapper(int width, int height);
    ~NvOFWrapper();

    bool Init();
    void Execute(const void *input0, const void *input1, void *flowOutput);

private:
    void *ofHandle = nullptr;
    int width = 0;
    int height = 0;
#ifdef NV_OF_SDK
    cudaStream_t stream = 0;
#else
    /* stream is unused when CUDA/NvOF is not available; keep a generic
       pointer to avoid leaking CUDA types in the public header. */
    void *stream = nullptr;
#endif
    bool nvof_available = false;
};
