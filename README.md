# vapoursynth-mvtools

## What I added

- Optional NVIDIA Optical Flow (NvOF) support (runtime guarded).
- Meson option: `nv_of_sdk` to enable/include the NvOF SDK at build time.
- New wrapper and C API: `src/NvOFWrapper.{h,cpp}` and `src/NvOF_capi.{h,cpp}`.
- `mv.FlowFPS` / helper updated with an optional `use_nvof` parameter to compute flow via NvOF.
- CI: added `vapoursynth-mvtools/.github/workflows/release-windows.yml` to build and publish a zip to GitHub Releases.

> [!NOTE]
> - Builds without the NvOF SDK still succeed; to enable hardware flow the SDK and CUDA must be available on the build/runtime machine or CI runner.
> - The NvOF implementation is compiled only when `-DNV_OF_SDK` is set; local SDK headers/libraries may be required to match exact SDK symbols.

## Description

MVTools is a set of filters for motion estimation and compensation.

This is a port of version 2.5.11.20 of the Avisynth plugin.

Some changes from version 2.5.11.9 of the SVP fork have been incorporated as well (http://www.svp-team.com/wiki/Download).

The filter DepanEstimate was ported from the Avisynth plugin DepanEstimate, version 1.10.

The filters DepanCompensate and DepanStabilise were ported from the Avisynth plugin Depan, version 1.13.1.


## Differences

* All:
  * Free multithreading, courtesy of VapourSynth.
  * Parameters are all lowercase now.
  * YUY2 is not supported.
  * Grayscale, 4:2:0, 4:2:2, 4:4:0, and 4:4:4 are supported, except for DepanCompensate and DepanStabilise, which don't support 4:4:0.
  * Up to 16 bits per sample are supported.
  * The audio is definitely not killed.
  * No "planar" parameter.
  * "isse" parameter renamed to "opt".

* Analyse:
  * No "temporal" parameter, as it's sort of incompatible with multithreading.
  * No "outfile" parameter.
  * No "sadx264" parameter. If opt is True, the best functions imported from x264 will be selected automatically. Otherwise, only C functions will be used.
  * New parameters "fields" and "tff".
  * The optimised SAD, SATD, and SSD functions from x264 have been updated to the latest versions (as of September 2014).
  * Block sizes of 64x32, 64x64, 128x64, and 128x128 are supported.
  * The "dct" parameter can be 5..10 even with blocks larger than 16x16.

* Recalculate: Same as Analyse.

* Compensate:
  * No "recursion" parameter. It was dodgy.
  * New parameter "tff".

* Flow: New parameter "tff".

* SCDetection:
  * No "ysc" parameter. The input frames are returned unchanged, with the `_SceneChangePrev` or `_SceneChangeNext` property attached.
  * No "isse" parameter. It wasn't used.

* DepanAnalyse:
  * Formerly "MDepan".
  * New parameters "fields" and "tff".
  * No "log", "range", "isse" parameters.

* DepanEstimate:
  * New parameters "fields" and "tff".
  * No "range", "log", "debug", "extlog" parameters.

* DepanCompensate:
  * Formerly "DePan".
  * No "inputlog" parameter.

* DepanStabilise:
  * Formerly "DePanStabilize".
  * No "inputlog" parameter.
  * Methods -1 and 2 unavailable.

## Usage

Examples:

```py
mv.Super(clip clip[, int hpad=16, int vpad=16, int pel=2, int levels=0, bint chroma=True, int sharp=2, int rfilter=2, clip pelclip=None, bint opt=True])

mv.Analyse(clip super[, int blksize=8, int blksizev=blksize, int levels=0, int search=4, int searchparam=2, int pelsearch=0, bint isb=False, int lambda, bint chroma=True, int delta=1, bint truemotion=True, int lsad, int plevel, int global, int pnew, int pzero=pnew, int pglobal=0, int overlap=0, int overlapv=overlap, bint divide=False, int badsad=10000, int badrange=24, bint opt=True, bint meander=True, bint trymany=False, bint fields=False, bint tff, int search_coarse=3, int dct=0])
```

...and others; see original `readme.rst` for full usage examples.

## Compilation

FFTW3 configured for 32 bit floats is required ("fftw3f").

```sh
meson setup build
ninja -C build
```

## License

GPL 2, same as the Avisynth plugins.
