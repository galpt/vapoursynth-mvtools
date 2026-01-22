#pragma once

#include <cuda_runtime.h>

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
    cudaStream_t stream = 0;
    bool nvof_available = false;
};
