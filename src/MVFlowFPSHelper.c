
#include <VapourSynth4.h>
#include <VSHelper4.h>

#include "MaskFun.h"
#include "SimpleResize.h"

#include "MVFlowFPSHelper.h"
#include "NvOF_capi.h"

const VSFrame *VS_CC mvflowfpshelperGetFrame(int n, int activationReason, void *instanceData, void **frameData, VSFrameContext *frameCtx, VSCore *core, const VSAPI *vsapi)
{
    (void)frameData;

    MVFlowFPSHelperData *d = (MVFlowFPSHelperData *)instanceData;

    if (activationReason == arInitial)
    {
        if (d->use_nvof && d->super_node)
        {
            vsapi->requestFrameFilter(n, d->super_node, frameCtx);
            vsapi->requestFrameFilter(n + 1, d->super_node, frameCtx);
        }
        else
        {
            vsapi->requestFrameFilter(n, d->vectors, frameCtx);
        }
    }
    else if (activationReason == arAllFramesReady)
    {
        const VSFrame *src = NULL;
        if (d->use_nvof && d->super_node)
        {
            src = vsapi->getFrameFilter(n, d->super_node, frameCtx);
        }
        else
        {
            src = vsapi->getFrameFilter(n, d->vectors, frameCtx);
        }

        if (!d->use_nvof || !d->super_node)
        {
            FakeGroupOfPlanes fgop;

            fgopInit(&fgop, &d->vectors_data);

            const VSMap *mvprops = vsapi->getFramePropertiesRO(src);
            fgopUpdate(&fgop, (const uint8_t *)vsapi->mapGetData(mvprops, prop_MVTools_vectors, 0, NULL));

            int isUsable = fgopIsUsable(&fgop, d->thscd1, d->thscd2);

            if (isUsable)
            {
                VSFrame *dst = vsapi->copyFrame(src, core);
                vsapi->freeFrame(src);

                VSMap *props = vsapi->getFramePropertiesRW(dst);

                const int xRatioUV = d->vectors_data.xRatioUV;
                const int yRatioUV = d->vectors_data.yRatioUV;
                const int nBlkX = d->vectors_data.nBlkX;
                const int nBlkY = d->vectors_data.nBlkY;
                const int nHeightP = d->nHeightP;
                const int nHeightPUV = d->nHeightPUV;
                const int VPitchY = d->VPitchY;
                const int VPitchUV = d->VPitchUV;
                const int nBlkXP = d->nBlkXP;
                const int nBlkYP = d->nBlkYP;
                SimpleResize *upsizer = &d->upsizer;
                SimpleResize *upsizerUV = &d->upsizerUV;

                int full_size_y = nHeightP * VPitchY * sizeof(int16_t);
                int small_size = nBlkXP * nBlkYP * sizeof(int16_t);

                int16_t *VXFullY = (int16_t *)malloc(full_size_y);
                int16_t *VYFullY = (int16_t *)malloc(full_size_y);
                int16_t *VXSmallY = (int16_t *)malloc(small_size);
                int16_t *VYSmallY = (int16_t *)malloc(small_size);

                // make  vector vx and vy small masks
                MakeVectorSmallMasks(&fgop, nBlkX, nBlkY, VXSmallY, nBlkXP, VYSmallY, nBlkXP);

                CheckAndPadSmallY(VXSmallY, VYSmallY, nBlkXP, nBlkYP, nBlkX, nBlkY);

                upsizer->simpleResize_int16_t(upsizer, VXFullY, VPitchY, VXSmallY, nBlkXP, 1);
                upsizer->simpleResize_int16_t(upsizer, VYFullY, VPitchY, VYSmallY, nBlkXP, 0);

                vsapi->mapSetData(props, prop_VXFullY, (const char *)VXFullY, full_size_y, dtBinary, maReplace);
                vsapi->mapSetData(props, prop_VYFullY, (const char *)VYFullY, full_size_y, dtBinary, maReplace);

                free(VXFullY);
                free(VYFullY);

                if (d->supervi->format.colorFamily != cfGray)
                {
                    int full_size_uv = nHeightPUV * VPitchUV * sizeof(int16_t);

                    int16_t *VXFullUV = (int16_t *)malloc(full_size_uv);
                    int16_t *VYFullUV = (int16_t *)malloc(full_size_uv);
                    int16_t *VXSmallUV = (int16_t *)malloc(small_size);
                    int16_t *VYSmallUV = (int16_t *)malloc(small_size);

                    VectorSmallMaskYToHalfUV(VXSmallY, nBlkXP, nBlkYP, VXSmallUV, xRatioUV);
                    VectorSmallMaskYToHalfUV(VYSmallY, nBlkXP, nBlkYP, VYSmallUV, yRatioUV);

                    upsizerUV->simpleResize_int16_t(upsizerUV, VXFullUV, VPitchUV, VXSmallUV, nBlkXP, 1);
                    upsizerUV->simpleResize_int16_t(upsizerUV, VYFullUV, VPitchUV, VYSmallUV, nBlkXP, 0);

                    free(VXSmallUV);
                    free(VYSmallUV);

                    vsapi->mapSetData(props, prop_VXFullUV, (const char *)VXFullUV, full_size_uv, dtBinary, maReplace);
                    vsapi->mapSetData(props, prop_VYFullUV, (const char *)VYFullUV, full_size_uv, dtBinary, maReplace);

                    free(VXFullUV);
                    free(VYFullUV);
                }

                free(VXSmallY);
                free(VYSmallY);

                fgopDeinit(&fgop);

                return dst;
            }
            else
            { // poor estimation
                fgopDeinit(&fgop);

                return src;
            }
        }
        else
        {
            // NvOF path: compute flow between super frames n and n+1
            const VSFrame *src0 = src;
            const VSFrame *src1 = vsapi->getFrameFilter(n + 1, d->super_node, frameCtx);

            if (!src0 || !src1)
            {
                if (src0)
                    vsapi->freeFrame(src0);
                if (src1)
                    vsapi->freeFrame(src1);
                return NULL;
            }

            int sw = d->supervi->width;
            int sh = d->supervi->height;

            const uint8_t *ptr0 = vsapi->getReadPtr(src0, 0);
            const uint8_t *ptr1 = vsapi->getReadPtr(src1, 0);
            int stride0 = vsapi->getStride(src0, 0);
            int stride1 = vsapi->getStride(src1, 0);

            int total = sw * sh;
            int16_t *vx_full = (int16_t *)malloc(sizeof(int16_t) * total);
            int16_t *vy_full = (int16_t *)malloc(sizeof(int16_t) * total);
            int rc = nvof_compute(d->nvof_handle, ptr0, stride0, ptr1, stride1, vx_full, vy_full);

            VSFrame *dst = vsapi->copyFrame(src0, core);
            vsapi->freeFrame(src0);
            vsapi->freeFrame(src1);

            VSMap *props = vsapi->getFramePropertiesRW(dst);

            const int nHeightP = d->nHeightP;
            const int VPitchY = d->VPitchY;
            int full_size_y = nHeightP * VPitchY * sizeof(int16_t);
            int16_t *VXFullY = (int16_t *)malloc(full_size_y);
            int16_t *VYFullY = (int16_t *)malloc(full_size_y);

            if (rc == 0)
            {
                // simple nearest-neighbor scale from super resolution to VPitchY x nHeightP
                double sx = (double)sw / (double)VPitchY;
                double sy = (double)sh / (double)nHeightP;
                for (int y = 0; y < nHeightP; ++y)
                {
                    int syi = (int)(y * sy);
                    if (syi >= sh)
                        syi = sh - 1;
                    for (int x = 0; x < VPitchY; ++x)
                    {
                        int sxi = (int)(x * sx);
                        if (sxi >= sw)
                            sxi = sw - 1;
                        int idx_src = syi * sw + sxi;
                        int idx_dst = y * VPitchY + x;
                        VXFullY[idx_dst] = vx_full[idx_src];
                        VYFullY[idx_dst] = vy_full[idx_src];
                    }
                }
            }
            else
            {
                // fallback to zeros
                memset(VXFullY, 0, full_size_y);
                memset(VYFullY, 0, full_size_y);
            }

            vsapi->mapSetData(props, prop_VXFullY, (const char *)VXFullY, full_size_y, dtBinary, maReplace);
            vsapi->mapSetData(props, prop_VYFullY, (const char *)VYFullY, full_size_y, dtBinary, maReplace);

            free(VXFullY);
            free(VYFullY);
            free(vx_full);
            free(vy_full);

            return dst;
        }
    }

    return NULL;
}

void VS_CC mvflowfpshelperFree(void *instanceData, VSCore *core, const VSAPI *vsapi)
{
    (void)core;

    MVFlowFPSHelperData *d = (MVFlowFPSHelperData *)instanceData;

    vsapi->freeNode(d->vectors);

    free(d);
}
