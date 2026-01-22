#include "NvOF_capi.h"
#include "NvOFWrapper.h"
#include <cstdlib>
#include <cstring>
#ifdef NV_OF_SDK
#include <cuda_runtime.h>
#endif

struct NvOFContext
{
    NvOFWrapper *wrapper;
    int width;
    int height;
};

void *nvof_create(int width, int height)
{
    NvOFContext *ctx = (NvOFContext *)malloc(sizeof(NvOFContext));
    if (!ctx)
        return NULL;
    ctx->wrapper = new NvOFWrapper(width, height);
    ctx->width = width;
    ctx->height = height;
    if (!ctx->wrapper)
    {
        free(ctx);
        return NULL;
    }
    if (!ctx->wrapper->Init())
    {
        // allow creation even if NvOF init failed; wrapper will fall back
        // but we still return the context so caller can proceed.
    }
    return ctx;
}

void nvof_destroy(void *handle)
{
    if (!handle)
        return;
    NvOFContext *ctx = (NvOFContext *)handle;
    if (ctx->wrapper)
        delete ctx->wrapper;
    free(ctx);
}

int nvof_compute(void *handle, const unsigned char *frame0, int pitch0, const unsigned char *frame1, int pitch1, int16_t *vx_out, int16_t *vy_out)
{
    if (!handle || !frame0 || !frame1 || !vx_out || !vy_out)
        return -1;
    NvOFContext *ctx = (NvOFContext *)handle;

#ifdef NV_OF_SDK
    int w = ctx->width;
    int h = ctx->height;
    size_t flow_bytes = (size_t)w * (size_t)h * 4; // 2 x int16 per pixel

    unsigned char *d0 = nullptr, *d1 = nullptr;
    void *d_flow = nullptr;
    size_t dev_pitch0 = 0, dev_pitch1 = 0;

    if (cudaMallocPitch((void **)&d0, &dev_pitch0, (size_t)w, (size_t)h) != cudaSuccess)
        goto fail;
    if (cudaMallocPitch((void **)&d1, &dev_pitch1, (size_t)w, (size_t)h) != cudaSuccess)
        goto fail_free_d0;
    if (cudaMalloc(&d_flow, flow_bytes) != cudaSuccess)
        goto fail_free_d1;

    // Copy host frames into pitched device memory
    if (cudaMemcpy2D(d0, dev_pitch0, frame0, (size_t)pitch0, (size_t)w, (size_t)h, cudaMemcpyHostToDevice) != cudaSuccess)
        goto fail_free_all;
    if (cudaMemcpy2D(d1, dev_pitch1, frame1, (size_t)pitch1, (size_t)w, (size_t)h, cudaMemcpyHostToDevice) != cudaSuccess)
        goto fail_free_all;

    // Execute NvOF
    ctx->wrapper->Execute(d0, d1, d_flow);

    // Download flow buffer
    unsigned char *host_flow = (unsigned char *)malloc(flow_bytes);
    if (!host_flow)
        goto fail_free_all;
    if (cudaMemcpy(host_flow, d_flow, flow_bytes, cudaMemcpyDeviceToHost) != cudaSuccess)
    {
        free(host_flow);
        goto fail_free_all;
    }

    // Interpret flow as int16 vx, int16 vy per pixel
    int16_t *flow_shorts = (int16_t *)host_flow;
    for (int i = 0; i < w * h; ++i)
    {
        vx_out[i] = flow_shorts[i * 2 + 0];
        vy_out[i] = flow_shorts[i * 2 + 1];
    }

    free(host_flow);
    cudaFree(d_flow);
    cudaFree(d1);
    cudaFree(d0);
    return 0;

fail_free_all:
    if (d_flow)
        cudaFree(d_flow);
fail_free_d1:
    if (d1)
        cudaFree(d1);
fail_free_d0:
    if (d0)
        cudaFree(d0);
fail:
    return -2;
#else
    (void)ctx;
    (void)frame0;
    (void)pitch0;
    (void)frame1;
    (void)pitch1;
    (void)vx_out;
    (void)vy_out;
    return -2;
#endif
}
