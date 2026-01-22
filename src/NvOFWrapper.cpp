#include "NvOFWrapper.h"
#include <cuda.h>
#include <iostream>

#ifdef NV_OF_SDK
#include <nvOpticalFlowCuda.h>
#endif

NvOFWrapper::NvOFWrapper(int w, int h) : width(w), height(h) {}

NvOFWrapper::~NvOFWrapper()
{
    if (stream)
        cudaStreamDestroy(stream);

#ifdef NV_OF_SDK
    if (nvof_available && ofHandle)
    {
        nvOpticalFlowDestroy(ofHandle);
        ofHandle = nullptr;
    }
#endif
}

bool NvOFWrapper::Init()
{
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

#ifdef NV_OF_SDK
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
#else
    nvof_available = false;
#endif

    return true;
}

void NvOFWrapper::Execute(const void *input0, const void *input1, void *flowOutput)
{
    if (!nvof_available)
    {
        size_t flow_bytes = (size_t)width * (size_t)height * 4;
        cudaMemsetAsync(flowOutput, 0, flow_bytes, stream);
        cudaStreamSynchronize(stream);
        return;
    }

#ifdef NV_OF_SDK
    NV_OF_EXECUTE_INPUT_PARAMS inParams = {};
    inParams.inputFrame = (CUdeviceptr)input0;
    inParams.referenceFrame = (CUdeviceptr)input1;

    NV_OF_EXECUTE_OUTPUT_PARAMS outParams = {};
    outParams.outputFlowVector = (CUdeviceptr)flowOutput;

    NvOFStatus status = nvOpticalFlowExecute(ofHandle, &inParams, &outParams);
    if (status != NV_OF_SUCCESS)
    {
        size_t flow_bytes = (size_t)width * (size_t)height * 4;
        cudaMemsetAsync(flowOutput, 0, flow_bytes, stream);
    }
    cudaStreamSynchronize(stream);
#else
    size_t flow_bytes = (size_t)width * (size_t)height * 4;
    cudaMemsetAsync(flowOutput, 0, flow_bytes, stream);
    cudaStreamSynchronize(stream);
#endif
}
