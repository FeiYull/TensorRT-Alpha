// =============================================================================
//  sample_preprocess_compare
// -----------------------------------------------------------------------------
//  What this sample proves
//      Running the 5 preprocessing operators one after another and running the
//      single fused kernel must produce BIT-IDENTICAL results.
//
//      Path A (5 separate kernels, one launch each)
//          1 resize          u8  BGR HWC  srcW x srcH   -> newW x newH
//          2 copyMakeBorder  u8  BGR HWC  newW x newH   -> outW x outH (pad 114)
//          3 cvtColor        u8  BGR HWC  outW x outH   -> RGB HWC
//          4 hwc2chw         u8  RGB HWC  outW x outH   -> RGB CHW
//          5 normalize       u8  RGB CHW  outW x outH   -> f32 RGB CHW (/255)
//      Path B (fused, one launch)
//          preprocessFusedVecDevice: u8 BGR HWC -> f32 RGB CHW
//
//      Both paths then get copied back to the host and compared element by
//      element. "Identical" means bit-identical (== on float), NOT "within a
//      tolerance".
//
//  IMPORTANT -- the geometry must be letterbox, not a plain stretch
//      The 5-kernel chain above is wired as a LETTERBOX:
//          resize              srcW x srcH -> newW x newH   (uniform scale)
//          copyMakeBorder      pad to outW x outH            (floor/ceil split)
//      where (newW, newH, padLeft, padTop, ...) come from the shared
//      computeLetterBox() that the fused kernel itself uses. That is what makes
//      the two paths comparable.
//
//      A chain built as "stretch resize to 640x640 + fixed 16/16/8/8 padding" is
//      a DIFFERENT transform and will legitimately differ from the fused kernel.
//
//  Usage
//      sample_preprocess_compare.exe [--img PATH] [--size W H] [--pad V]
//                                    [--divisor F] [--iters N] [--selfcheck]
//          --img PATH     input image            (default data/bus.jpg)
//          --size W H     network input size     (default 640 640)
//          --pad V        u8-domain border fill  (default 114)
//          --divisor F    normalize divisor      (default 255, i.e. /255)
//          --iters N      repeat the launches N times before copying back
//                         (default 1; useful when profiling with ncu/nsys)
//          --selfcheck    fault injection: flip one source byte, re-run only the
//                         fused path and require the comparator to catch it.
//                         Guards against a comparator that always reports "equal".
//
//  Exit code: 0 = all elements identical, 1 = differences found, 2 = setup error.
// =============================================================================
#include "trt_alpha/kernels/ops/copy_make_border.hpp"
#include "trt_alpha/kernels/ops/cvt_color.hpp"
#include "trt_alpha/kernels/ops/hwc2chw.hpp"
#include "trt_alpha/kernels/ops/normalize.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec.hpp"
#include "trt_alpha/kernels/ops/resize.hpp"

#include <cuda_runtime.h>
#include <opencv2/opencv.hpp>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

using namespace trt_alpha::kernels::ops;

namespace {

void cudaMust(cudaError_t e, const char* what)
{
    if (e != cudaSuccess)
    {
        std::fprintf(stderr, "[CUDA ERROR] %s : %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}

//! Tightly pack a cv::Mat row by row (imread's Mat may have row padding, so a
//! plain memcpy is not always valid).
std::vector<unsigned char> packMat(const cv::Mat& m)
{
    std::vector<unsigned char> out((std::size_t)m.total() * m.elemSize());
    if (out.empty()) return out;
    if (m.isContinuous())
    {
        std::memcpy(out.data(), m.data, out.size());
    }
    else
    {
        const std::size_t rowBytes = (std::size_t)m.cols * m.elemSize();
        for (int y = 0; y < m.rows; ++y)
            std::memcpy(out.data() + (std::size_t)y * rowBytes, m.ptr(y), rowBytes);
    }
    return out;
}

struct CompareResult
{
    long long nDiff    = 0;
    double    maxAbs   = 0.0;
    long long firstIdx = -1;
    float     firstA   = 0.0f, firstB = 0.0f;
    int       firstC   = 0, firstY = 0, firstX = 0;
};

//! Element-by-element comparison of two f32 RGB CHW buffers.
//! "Equal" is strict float equality (NaN == NaN is also treated as equal).
CompareResult compareF32(const std::vector<float>& a, const std::vector<float>& b,
                         int outW, int outH)
{
    CompareResult r;
    const long long plane = (long long)outW * outH;
    const long long n     = (long long)std::min(a.size(), b.size());

    for (long long i = 0; i < n; ++i)
    {
        const float x = a[(std::size_t)i];
        const float y = b[(std::size_t)i];
        const bool  same = (x == y) || (std::isnan(x) && std::isnan(y));
        if (same) continue;

        ++r.nDiff;
        const double d = std::fabs((double)x - (double)y);
        if (d > r.maxAbs) r.maxAbs = d;

        if (r.firstIdx < 0)
        {
            r.firstIdx = i;
            r.firstC   = (int)(i / plane);
            const long long rem = i % plane;
            r.firstY   = (int)(rem / outW);
            r.firstX   = (int)(rem % outW);
            r.firstA   = x;
            r.firstB   = y;
        }
    }
    return r;
}

}  // namespace

int main(int argc, char** argv)
{
    std::string imgPath = "data/bus.jpg";
    int         netW    = 640, netH = 640;
    double      padValue = 114.0;
    float       divisor  = 255.0f;
    int         iters    = 1;
    bool        selfcheck = false;

    for (int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        if      (!std::strcmp(a, "--img")    && i + 1 < argc) imgPath  = argv[++i];
        else if (!std::strcmp(a, "--size")   && i + 2 < argc) { netW = std::atoi(argv[++i]); netH = std::atoi(argv[++i]); }
        else if (!std::strcmp(a, "--pad")    && i + 1 < argc) padValue = std::atof(argv[++i]);
        else if (!std::strcmp(a, "--divisor")&& i + 1 < argc) divisor  = (float)std::atof(argv[++i]);
        else if (!std::strcmp(a, "--iters")  && i + 1 < argc) iters    = std::atoi(argv[++i]);
        else if (!std::strcmp(a, "--selfcheck"))              selfcheck = true;
        else { std::fprintf(stderr, "[error] unknown argument: %s\n", a); return 2; }
    }
    if (netW <= 0 || netH <= 0 || iters <= 0 || divisor == 0.0f)
    {
        std::fprintf(stderr, "[error] bad --size / --iters / --divisor\n");
        return 2;
    }

    // ------------------------------------------------------------------ input
    cv::Mat img = cv::imread(imgPath, cv::IMREAD_COLOR);
    if (img.empty())
    {
        std::fprintf(stderr, "[error] cannot read image: %s\n", imgPath.c_str());
        return 2;
    }
    if (img.channels() != 3)
    {
        std::fprintf(stderr, "[error] expected a 3-channel image, got %d channels\n", img.channels());
        return 2;
    }

    const int srcW = img.cols;
    const int srcH = img.rows;

    // ---------------------------------------------------- letterbox geometry
    // The SAME function the fused kernel uses internally -- one single source of
    // truth, so the two paths cannot disagree about the geometry.
    const LetterBoxInfo geo = computeLetterBox(srcW, srcH, netW, netH);
    if (geo.newW < 1 || geo.newH < 1 || geo.newW > netW || geo.newH > netH)
    {
        std::fprintf(stderr, "[error] degenerate letterbox geometry for %dx%d -> %dx%d\n",
                     srcW, srcH, netW, netH);
        return 2;
    }

    std::printf("============ sample_preprocess_compare ============\n");
    std::printf("  image            : %s\n", imgPath.c_str());
    std::printf("  source           : %d x %d  (BGR HWC, u8)\n", srcW, srcH);
    std::printf("  network input    : %d x %d\n", netW, netH);
    std::printf("  letterbox ratio  : %.9g\n", geo.ratio);
    std::printf("  resized (newW x newH) : %d x %d\n", geo.newW, geo.newH);
    std::printf("  padding L/R/T/B  : %d/%d/%d/%d   (fill %g, divisor %g)\n",
                geo.padLeft, geo.padRight, geo.padTop, geo.padBottom,
                padValue, (double)divisor);
    std::printf("  repeat launches  : %d\n", iters);
    std::printf("  judgement        : bit-exact (== on float, NOT a tolerance)\n\n");

    const std::size_t srcElems  = (std::size_t)srcW * srcH * 3;
    const std::size_t newElems  = (std::size_t)geo.newW * geo.newH * 3;
    const std::size_t outElems  = (std::size_t)netW * netH * 3;

    // ------------------------------------------------------------ host input
    std::vector<unsigned char> hSrc = packMat(img);
    if (hSrc.size() != srcElems)
    {
        std::fprintf(stderr, "[error] packed size %zu != %zu\n", hSrc.size(), srcElems);
        return 2;
    }

    // ------------------------------------------------------- device buffers
    unsigned char* dSrc     = nullptr;   // srcW x srcH x 3     u8  BGR HWC
    unsigned char* dResized = nullptr;   // newW x newH x 3     u8  BGR HWC
    unsigned char* dPadded  = nullptr;   // netW x netH x 3     u8  BGR HWC
    unsigned char* dRgb     = nullptr;   // netW x netH x 3     u8  RGB HWC
    unsigned char* dChw     = nullptr;   // 3 x netH x netW     u8  RGB CHW
    float*         dNormA   = nullptr;   // 3 x netH x netW     f32 RGB CHW  (path A)
    float*         dNormB   = nullptr;   // 3 x netH x netW     f32 RGB CHW  (path B)

    cudaMust(cudaMalloc(&dSrc,     srcElems),                        "cudaMalloc dSrc");
    cudaMust(cudaMalloc(&dResized, newElems),                        "cudaMalloc dResized");
    cudaMust(cudaMalloc(&dPadded,  outElems),                        "cudaMalloc dPadded");
    cudaMust(cudaMalloc(&dRgb,     outElems),                        "cudaMalloc dRgb");
    cudaMust(cudaMalloc(&dChw,     outElems),                        "cudaMalloc dChw");
    cudaMust(cudaMalloc(&dNormA,   outElems * sizeof(float)),        "cudaMalloc dNormA");
    cudaMust(cudaMalloc(&dNormB,   outElems * sizeof(float)),        "cudaMalloc dNormB");

    // ------------------------------------------------------------ build plans
    // All plans are built once, outside the loop: coefficients are precomputed and
    // uploaded here, so the launches below neither allocate nor copy.
    CvResizePlan<unsigned char, unsigned char> resizePlan(srcW, srcH, geo.newW, geo.newH);
    CvCopyMakeBorderPlan<unsigned char, unsigned char> borderPlan(
        geo.newW, geo.newH, geo.padTop, geo.padBottom, geo.padLeft, geo.padRight);
    CvCvtColorPlan<unsigned char, unsigned char> cvtPlan(netW, netH, ColorCode::BGR2RGB);
    CvHwcToChwPlan<unsigned char, unsigned char> chwPlan(netW, netH, 3);
    CvNormalizePlan<unsigned char, float>        normPlan =
        CvNormalizePlan<unsigned char, float>::byDivisor(divisor);
    CvPreprocessVecPlan vecPlan = CvPreprocessVecPlan::byDivisor(
        srcW, srcH, netW, netH, padValue, divisor);

    if (!resizePlan.ok()) { std::fprintf(stderr, "[plan] CvResizePlan : %s\n", cudaGetErrorString(resizePlan.lastError())); return 2; }
    if (!borderPlan.ok()) { std::fprintf(stderr, "[plan] CvCopyMakeBorderPlan : %s\n", cudaGetErrorString(borderPlan.lastError())); return 2; }
    if (!cvtPlan.ok())    { std::fprintf(stderr, "[plan] CvCvtColorPlan : %s\n", cudaGetErrorString(cvtPlan.lastError())); return 2; }
    if (!chwPlan.ok())    { std::fprintf(stderr, "[plan] CvHwcToChwPlan : %s\n", cudaGetErrorString(chwPlan.lastError())); return 2; }
    if (!normPlan.ok())   { std::fprintf(stderr, "[plan] CvNormalizePlan : %s\n", cudaGetErrorString(normPlan.lastError())); return 2; }
    if (!vecPlan.ok())    { std::fprintf(stderr, "[plan] CvPreprocessVecPlan : %s\n", cudaGetErrorString(vecPlan.lastError())); return 2; }

    const double padScalar[3] = { padValue, padValue, padValue };

    // ------------------------------------------------------------ run both paths
    cudaMust(cudaMemcpy(dSrc, hSrc.data(), srcElems, cudaMemcpyHostToDevice), "H2D src");

    for (int it = 0; it < iters; ++it)
    {
        // ---- path A: five separate kernels, in order ----
        resizePlan.launch(nullptr, 1, dSrc, dResized, 3);
        borderPlan.launch(nullptr, 1, dResized, dPadded, 3, BorderType::Constant, padScalar);
        cvtPlan.launch(nullptr, 1, dPadded, dRgb, 3);
        chwPlan.launch(nullptr, 1, dRgb, dChw);
        normPlan.launch(nullptr, 1, dChw, dNormA, 3, netW * netH);

        // ---- path B: one fused kernel ----
        vecPlan.launch(nullptr, 1, dSrc, dNormB);
    }

    cudaMust(cudaGetLastError(), "kernel launch");
    cudaMust(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

    // Report per-step launch errors as well (each plan records its own).
    const cudaError_t eA[] = { resizePlan.lastError(), borderPlan.lastError(),
                               cvtPlan.lastError(), chwPlan.lastError(),
                               normPlan.lastError() };
    const char* nameA[] = { "resize", "copyMakeBorder", "cvtColor", "hwc2chw", "normalize" };
    for (int i = 0; i < 5; ++i)
    {
        if (eA[i] != cudaSuccess)
        {
            std::fprintf(stderr, "[launch] %s : %s\n", nameA[i], cudaGetErrorString(eA[i]));
            return 2;
        }
    }
    if (vecPlan.lastError() != cudaSuccess)
    {
        std::fprintf(stderr, "[launch] preprocess_fused_vec : %s\n",
                     cudaGetErrorString(vecPlan.lastError()));
        return 2;
    }

    // ------------------------------------------------------------ copy results back
    std::vector<float> hNormA(outElems), hNormB(outElems);
    cudaMust(cudaMemcpy(hNormA.data(), dNormA, outElems * sizeof(float), cudaMemcpyDeviceToHost), "D2H path A");
    cudaMust(cudaMemcpy(hNormB.data(), dNormB, outElems * sizeof(float), cudaMemcpyDeviceToHost), "D2H path B");

    // ----------------------------------------------------------------- compare
    const CompareResult cmp = compareF32(hNormA, hNormB, netW, netH);

    std::printf("---- result ----\n");
    std::printf("  5-kernel chain  : %zu elements\n", hNormA.size());
    std::printf("  fused kernel    : %zu elements\n", hNormB.size());
    std::printf("  differing       : %lld\n", cmp.nDiff);
    std::printf("  max |a - b|     : %g\n", cmp.maxAbs);
    if (cmp.firstIdx >= 0)
    {
        std::printf("  first mismatch  : #%lld [c=%d y=%d x=%d]  chain=%.9g  fused=%.9g\n",
                    cmp.firstIdx, cmp.firstC, cmp.firstY, cmp.firstX,
                    (double)cmp.firstA, (double)cmp.firstB);
    }

    bool ok = (cmp.nDiff == 0) && (hNormA.size() == hNormB.size());

    // ------------------------------------------------- optional fault injection
    bool selfOk = true;
    if (selfcheck)
    {
        std::printf("\n---- self-check (fault injection) ----\n");
        unsigned char* dBad = nullptr;
        cudaMust(cudaMalloc(&dBad, srcElems), "cudaMalloc dBad");
        cudaMust(cudaMemcpy(dBad, dSrc, srcElems, cudaMemcpyDeviceToDevice), "D2D copy for dBad");

        // Corrupt a full-width band of source rows, ^0xFF on every byte.
        //
        // Why not a single byte or a single bit? Two properties of the algorithm
        // (not bugs) make a fixed-position fault unreliable:
        //
        //   1) Quantization. The fixed-point chain ends with "(a + b + 2) >> 2". For
        //      a typical position a 1-LSB change moves the intermediate by only 1..2
        //      units before that final >> 2, so it is regularly absorbed and yields an
        //      identical output pixel.
        //
        //   2) Downscaling does not sample every source pixel. With a half-pixel
        //      convention the sampled rows/columns form a subset: for ratio 0.2963
        //      (scale 3.375) the sampled rows are 537, 538, 541, 542, 547, ... --
        //      row 540 lies in a gap. A fault placed there legitimately produces no
        //      output change at all.
        //
        // A band of ceil(1/ratio) + 2 rows always contains at least one sampled row
        // (consecutive sampled rows are at most ceil(1/ratio) apart), and it covers
        // every column, so the fault is guaranteed to reach the output for any
        // geometry. The full-byte flip makes the delta large enough to survive the
        // quantization of step 1).
        const int  bandRows   = (int)std::ceil(1.0 / geo.ratio) + 2;
        const std::size_t rowBytes = (std::size_t)srcW * 3;
        const std::size_t bandBytes = std::min((std::size_t)bandRows * rowBytes, srcElems);
        std::size_t flipAt = (srcElems / 2 / rowBytes) * rowBytes;   // start of a row mid-image
        if (flipAt + bandBytes > srcElems) flipAt = srcElems - bandBytes;
        flipAt = (flipAt / rowBytes) * rowBytes;

        std::vector<unsigned char> band(bandBytes, 0);
        cudaMust(cudaMemcpy(band.data(), dSrc + flipAt, bandBytes, cudaMemcpyDeviceToHost),
                 "read source band");
        const unsigned char before = band[0];
        for (std::size_t i = 0; i < bandBytes; ++i) band[i] = (unsigned char)(band[i] ^ 0xFF);
        cudaMust(cudaMemcpy(dBad + flipAt, band.data(), bandBytes, cudaMemcpyHostToDevice),
                 "write flipped band");

        float* dNormBad = nullptr;
        cudaMust(cudaMalloc(&dNormBad, outElems * sizeof(float)), "cudaMalloc dNormBad");
        vecPlan.launch(nullptr, 1, dBad, dNormBad);
        cudaMust(cudaDeviceSynchronize(), "sync after fault injection");

        std::vector<float> hNormBad(outElems);
        cudaMust(cudaMemcpy(hNormBad.data(), dNormBad, outElems * sizeof(float), cudaMemcpyDeviceToHost),
                 "D2H fault injection");

        const CompareResult bad = compareF32(hNormA, hNormBad, netW, netH);
        selfOk = (bad.nDiff > 0);
        std::printf("  corrupted %zu bytes (%d rows) starting at #%zu (first: %u -> %u)\n",
                    bandBytes, bandRows, flipAt, (unsigned)before, (unsigned)band[0]);
        std::printf("  comparator found %lld differing elements  -> %s\n",
                    bad.nDiff, selfOk ? "CATCH OK" : "NOT CAUGHT (comparator is broken!)");
        if (bad.nDiff > 0)
        {
            std::printf("  first mismatch  : [c=%d y=%d x=%d]  chain=%.9g  fused=%.9g\n",
                        bad.firstC, bad.firstY, bad.firstX,
                        (double)bad.firstA, (double)bad.firstB);
        }

        cudaFree(dNormBad);
        cudaFree(dBad);
    }

    // ------------------------------------------------------------------- verdict
    std::printf("\n============ verdict ============\n");
    if (ok && selfOk)
    {
        std::printf("  The 5-kernel chain and the fused kernel are BIT-IDENTICAL (%lld elements compared).\n",
                    (long long)hNormA.size());
    }
    else if (!ok)
    {
        std::printf("  MISMATCH: %lld of %zu elements differ (max %g).\n",
                    cmp.nDiff, hNormA.size(), cmp.maxAbs);
    }
    else
    {
        std::printf("  The two paths agree, but the fault-injection self-check did not fire.\n");
    }

    cudaFree(dSrc);
    cudaFree(dResized);
    cudaFree(dPadded);
    cudaFree(dRgb);
    cudaFree(dChw);
    cudaFree(dNormA);
    cudaFree(dNormB);

    return (ok && selfOk) ? 0 : 1;
}
