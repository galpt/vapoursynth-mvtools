#include "NvOFWrapper.h"
#include <iostream>
#include <cstring>

#ifdef NV_OF_SDK
#include <cuda.h>
#include <nvOpticalFlowCuda.h>
#include <cuda_runtime.h>
#endif

NvOFWrapper::NvOFWrapper(int w, int h) : width(w), height(h) {}

NvOFWrapper::~NvOFWrapper()
{
#ifdef NV_OF_SDK
    if (stream)
        cudaStreamDestroy(stream);

    if (nvof_available && ofHandle)
    {
        nvOpticalFlowDestroy(ofHandle);
        ofHandle = nullptr;
    }
#else
    /* nothing to destroy when CUDA/NvOF not present */
    (void)stream;
    (void)ofHandle;
#endif
}

bool NvOFWrapper::Init()
{
#ifdef NV_OF_SDK
    cudaError_t err = cudaStreamCreate(&stream);
    if (err != cudaSuccess)
        return false;

    int dev = 0;
    err = cudaGetDevice(&dev);
    if (err != cudaSuccess)
    {
        std::cerr << "NvOFWrapper: no CUDA device available\n";
        return false;
    }

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    std::cerr << "NvOFWrapper: device: " << prop.name << "\n";

    NV_OF_CUDA_API_CREATE_PARAMS createParams = {};
    createParams.width = width;
    createParams.height = height;
    createParams.cudaDevice = dev;
    createParams.cudaContext = nullptr;

    NvOFStatus status = nvOpticalFlowCreate(&createParams, &ofHandle);
    if (status != NV_OF_SUCCESS)
    {
        std::cerr << "NvOFWrapper: nvOpticalFlowCreate failed\n";
        nvof_available = false;
        return true; // allow fallback behavior
    }

    nvof_available = true;
    return true;
#else
    /* No CUDA available on this build — provide a safe fallback so the
       wrapper can be constructed and Init() succeeds. The runtime path
       will return zero flow. */
    stream = nullptr;
    nvof_available = false;
    return true;
#endif
}

void NvOFWrapper::Execute(const void *input0, const void *input1, void *flowOutput)
{
    size_t flow_bytes = (size_t)width * (size_t)height * 4;

#ifdef NV_OF_SDK
    if (!nvof_available)
    {
        cudaMemsetAsync(flowOutput, 0, flow_bytes, stream);
        cudaStreamSynchronize(stream);
        return;
    }

    NV_OF_EXECUTE_INPUT_PARAMS inParams = {};
    inParams.inputFrame = (CUdeviceptr)input0;
    inParams.referenceFrame = (CUdeviceptr)input1;

    NV_OF_EXECUTE_OUTPUT_PARAMS outParams = {};
    outParams.outputFlowVector = (CUdeviceptr)flowOutput;

    NvOFStatus status = nvOpticalFlowExecute(ofHandle, &inParams, &outParams);
    if (status != NV_OF_SUCCESS)
    {
        cudaMemsetAsync(flowOutput, 0, flow_bytes, stream);
    }
    cudaStreamSynchronize(stream);
#else
    /* Fallback when CUDA/NvOF not available: zero the flow buffer on the
       host. This keeps behavior consistent (zero flow) without requiring
       CUDA headers/libraries at build time. */
    std::memset(flowOutput, 0, flow_bytes);
    (void)input0;
    (void)input1;
#endif
}
