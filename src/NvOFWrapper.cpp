#include "NvOFWrapper.h"
#include <iostream>
#include <cstring>

#if defined(NV_OF_SDK) || defined(NV_OF_SDK_DYNAMIC)
#include <cuda.h>
#if defined(__has_include)
#if __has_include(<nvOpticalFlowCuda.h>)
#include <nvOpticalFlowCuda.h>
#elif __has_include(<nvOpticalFlow.h>)
#include <nvOpticalFlow.h>
#else
#error "NvOF headers not found: nvOpticalFlowCuda.h or nvOpticalFlow.h"
#endif
#else
#include <nvOpticalFlowCuda.h>
#endif
#include <cuda_runtime.h>
#endif

#if defined(NV_OF_SDK_DYNAMIC)
#if defined(_WIN32)
#include <windows.h>
#else
#include <dlfcn.h>
#endif
#endif

NvOFWrapper::NvOFWrapper(int w, int h) : width(w), height(h) {}

NvOFWrapper::~NvOFWrapper()
{
#if defined(NV_OF_SDK) || defined(NV_OF_SDK_DYNAMIC)
    if (stream)
        cudaStreamDestroy(stream);

    if (nvof_available && ofHandle)
    {
        // Call destroy either via import or via function pointer
#if defined(NV_OF_SDK)
        nvOpticalFlowDestroy(ofHandle);
#else
        if (pf_nvOpticalFlowDestroy)
            pf_nvOpticalFlowDestroy(ofHandle);
#endif
        ofHandle = nullptr;
    }

#if defined(NV_OF_SDK_DYNAMIC)
    if (nvof_module)
    {
#if defined(_WIN32)
        FreeLibrary((HMODULE)nvof_module);
#else
        dlclose(nvof_module);
#endif
        nvof_module = nullptr;
    }
#endif
#else
    /* nothing to destroy when CUDA/NvOF not present */
    (void)stream;
    (void)ofHandle;
#endif
}

bool NvOFWrapper::Init()
{
#if defined(NV_OF_SDK) || defined(NV_OF_SDK_DYNAMIC)
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

#if defined(NV_OF_SDK_DYNAMIC)
    // Attempt to load the NvOpticalFlow runtime DLL and resolve symbols
    if (!LoadNvOF())
    {
        std::cerr << "NvOFWrapper: could not load nvopticalflow runtime; falling back\n";
        nvof_available = false;
        return true;
    }
#endif

    NV_OF_CUDA_API_CREATE_PARAMS createParams = {};
    createParams.width = width;
    createParams.height = height;
    createParams.cudaDevice = dev;
    createParams.cudaContext = nullptr;

#if defined(NV_OF_SDK)
    NvOFStatus status = nvOpticalFlowCreate(&createParams, &ofHandle);
#else
    NvOFStatus status = pf_nvOpticalFlowCreate(&createParams, &ofHandle);
#endif
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

#if defined(NV_OF_SDK) || defined(NV_OF_SDK_DYNAMIC)
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

#if defined(NV_OF_SDK)
    NvOFStatus status = nvOpticalFlowExecute(ofHandle, &inParams, &outParams);
#else
    NvOFStatus status = pf_nvOpticalFlowExecute(ofHandle, &inParams, &outParams);
#endif
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

#if defined(NV_OF_SDK_DYNAMIC)
// Dynamic loader support ----------------------------------------------------
static void *nvof_module = nullptr;
static NvOFStatus (*pf_nvOpticalFlowCreate)(const NV_OF_CUDA_API_CREATE_PARAMS *, void **) = nullptr;
static void (*pf_nvOpticalFlowDestroy)(void *) = nullptr;
static NvOFStatus (*pf_nvOpticalFlowExecute)(void *, const NV_OF_EXECUTE_INPUT_PARAMS *, NV_OF_EXECUTE_OUTPUT_PARAMS *) = nullptr;

static bool LoadNvOF()
{
#if defined(_WIN32)
    const char *dllnames[] = {"nvopticalflow.dll", "nvopticalflow64.dll", nullptr};
    for (const char **p = dllnames; *p; ++p)
    {
        HMODULE h = LoadLibraryA(*p);
        if (h)
        {
            nvof_module = (void *)h;
            pf_nvOpticalFlowCreate = (decltype(pf_nvOpticalFlowCreate))GetProcAddress(h, "nvOpticalFlowCreate");
            pf_nvOpticalFlowDestroy = (decltype(pf_nvOpticalFlowDestroy))GetProcAddress(h, "nvOpticalFlowDestroy");
            pf_nvOpticalFlowExecute = (decltype(pf_nvOpticalFlowExecute))GetProcAddress(h, "nvOpticalFlowExecute");
            break;
        }
    }
#else
    const char *dllnames[] = {"libnvopticalflow.so", "libnvopticalflow.so.1", nullptr};
    for (const char **p = dllnames; *p; ++p)
    {
        void *h = dlopen(*p, RTLD_NOW);
        if (h)
        {
            nvof_module = h;
            pf_nvOpticalFlowCreate = (decltype(pf_nvOpticalFlowCreate))dlsym(h, "nvOpticalFlowCreate");
            pf_nvOpticalFlowDestroy = (decltype(pf_nvOpticalFlowDestroy))dlsym(h, "nvOpticalFlowDestroy");
            pf_nvOpticalFlowExecute = (decltype(pf_nvOpticalFlowExecute))dlsym(h, "nvOpticalFlowExecute");
            break;
        }
    }
#endif
    if (!nvof_module || !pf_nvOpticalFlowCreate || !pf_nvOpticalFlowDestroy || !pf_nvOpticalFlowExecute)
    {
        if (nvof_module)
        {
#if defined(_WIN32)
            FreeLibrary((HMODULE)nvof_module);
#else
            dlclose(nvof_module);
#endif
            nvof_module = nullptr;
        }
        pf_nvOpticalFlowCreate = nullptr;
        pf_nvOpticalFlowDestroy = nullptr;
        pf_nvOpticalFlowExecute = nullptr;
        return false;
    }
    return true;
}
#endif
