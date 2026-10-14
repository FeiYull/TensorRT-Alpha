// =============================================================================
//  sample_preprocess_voc_compare
// -----------------------------------------------------------------------------
//  What this sample proves
//      The C++ letterbox preprocessing (kernels/ops) and the Python reference
//      (ultralytics LetterBox with auto=False, i.e. "rect=False") agree
//      BIT-IDENTICALLY on the VOC dataset -- both in geometry and in pixels.
//
//  Why this exists
//      The old Python probe used LetterBox(auto=True) (= rect). That only pads
//      up to a multiple of the stride, so its output size follows the image
//      aspect ratio (640x448, 640x384, ...) and cannot feed a static engine.
//      This sample instead validates against auto=False, which pads all the way
//      to 640x640 -- the exact geometry C++ uses.
//
//  Input (produced by E:\Code\ultralytics\sry_dump_voc_rect_off.py)
//      index.tsv         one row per image
//      <stem>.src.bin    u8  BGR HWC  H0 x W0 x 3     cv2.imread decode result
//      <stem>.lb.bin     u8  BGR HWC  640 x 640 x 3   after letterbox
//      <stem>.norm.bin   f32 RGB CHW  3 x 640 x 640   after /255  (model input)
//
//  Checks per image
//      1 geometry  : computeLetterBox(H0,W0,640,640)  vs  index.tsv columns
//      2 letterbox : 5-kernel chain u8 output         vs  <stem>.lb.bin
//      3 normalize : fused kernel f32 output          vs  <stem>.norm.bin
//
//  Usage
//      sample_preprocess_voc_compare.exe [--index PATH] [--dir PATH]
//                                        [--size W H] [--limit N]
//          --index PATH  index.tsv   (default D:/Data/bin_preprocess/voc_rect_off/index.tsv)
//          --dir   PATH  bin dir     (default = the index's directory)
//          --size  W H   net input   (default 640 640)
//          --limit N     only the first N rows (0 = all)
//
//  Exit code: 0 = every image bit-identical, 1 = differences found, 2 = setup error.
// =============================================================================
#include "trt_alpha/kernels/ops/copy_make_border.hpp"
#include "trt_alpha/kernels/ops/cvt_color.hpp"
#include "trt_alpha/kernels/ops/hwc2chw.hpp"
#include "trt_alpha/kernels/ops/normalize.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/resize.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <memory>
#include <sstream>
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

bool readFile(const std::string& path, std::vector<unsigned char>& out)
{
    std::ifstream f(path, std::ios::binary);
    if (!f) return false;
    f.seekg(0, std::ios::end);
    const std::streamoff n = f.tellg();
    f.seekg(0, std::ios::beg);
    if (n < 0) return false;
    out.resize((std::size_t)n);
    if (n > 0) f.read(reinterpret_cast<char*>(out.data()), n);
    return (bool)f || f.eof();
}

struct Row
{
    std::string stem;
    int    H0 = 0, W0 = 0;
    int    newW = 0, newH = 0;
    int    padL = 0, padR = 0, padT = 0, padB = 0;
    std::string ratio;
};

std::vector<Row> readIndex(const std::string& path)
{
    std::vector<Row> rows;
    std::ifstream f(path);
    if (!f) return rows;
    std::string line;
    while (std::getline(f, line))
    {
        if (line.empty() || line[0] == '#') continue;
        std::istringstream ss(line);
        Row r;
        std::string tok;
        // image H0 W0 newW newH padL padR padT padB ratio ...
        std::getline(ss, r.stem, '\t');
        std::getline(ss, tok, '\t'); r.H0   = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.W0   = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.newW = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.newH = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.padL = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.padR = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.padT = std::atoi(tok.c_str());
        std::getline(ss, tok, '\t'); r.padB = std::atoi(tok.c_str());
        std::getline(ss, r.ratio, '\t');
        if (!r.stem.empty() && r.H0 > 0 && r.W0 > 0) rows.push_back(r);
    }
    return rows;
}

std::string dirOf(const std::string& p)
{
    const std::size_t k = p.find_last_of("/\\");
    return (k == std::string::npos) ? std::string(".") : p.substr(0, k);
}

}  // namespace

int main(int argc, char** argv)
{
    std::string indexPath = "D:/Data/bin_preprocess/voc_rect_off/index.tsv";
    std::string binDir;
    int netW = 640, netH = 640;
    int limit = 0;

    for (int i = 1; i < argc; ++i)
    {
        const char* a = argv[i];
        if      (!std::strcmp(a, "--index") && i + 1 < argc) indexPath = argv[++i];
        else if (!std::strcmp(a, "--dir")   && i + 1 < argc) binDir    = argv[++i];
        else if (!std::strcmp(a, "--size")  && i + 2 < argc) { netW = std::atoi(argv[++i]); netH = std::atoi(argv[++i]); }
        else if (!std::strcmp(a, "--limit") && i + 1 < argc) limit    = std::atoi(argv[++i]);
        else { std::fprintf(stderr, "[error] unknown argument: %s\n", a); return 2; }
    }
    if (binDir.empty()) binDir = dirOf(indexPath);

    std::vector<Row> rows = readIndex(indexPath);
    if (rows.empty())
    {
        std::fprintf(stderr, "[error] cannot read index: %s\n", indexPath.c_str());
        return 2;
    }
    if (limit > 0 && limit < (int)rows.size()) rows.resize((std::size_t)limit);

    std::printf("============ sample_preprocess_voc_compare ============\n");
    std::printf("  index           : %s\n", indexPath.c_str());
    std::printf("  bin dir         : %s\n", binDir.c_str());
    std::printf("  network input   : %d x %d\n", netW, netH);
    std::printf("  images          : %zu\n", rows.size());
    std::printf("  judgement       : bit-exact (== on float/u8, NOT a tolerance)\n\n");

    const std::size_t lbElems   = (std::size_t)netW * netH * 3;
    const std::size_t normBytes = lbElems * sizeof(float);

    // Reused device buffers (sized for the largest source we may meet).
    unsigned char* dSrc     = nullptr;
    unsigned char* dResized = nullptr;
    unsigned char* dPadded  = nullptr;
    unsigned char* dRgb     = nullptr;
    unsigned char* dChw     = nullptr;
    float*         dNorm    = nullptr;

    std::size_t capSrc = 0;
    std::size_t capNew = 0;

    // Plans are cached and rebuilt only when the source geometry changes --
    // this is the same pattern the real pipeline uses.
    std::unique_ptr<CvResizePlan<unsigned char, unsigned char>>         resizePlan;
    std::unique_ptr<CvCopyMakeBorderPlan<unsigned char, unsigned char>> borderPlan;
    std::unique_ptr<CvCvtColorPlan<unsigned char, unsigned char>>       cvtPlan;
    std::unique_ptr<CvHwcToChwPlan<unsigned char, unsigned char>>       chwPlan;
    CvNormalizePlan<unsigned char, float>                               normPlan =
        CvNormalizePlan<unsigned char, float>::byDivisor(255.0f);
    std::unique_ptr<CvPreprocessVecPlan>                                vecPlan;

    int curSrcW = -1, curSrcH = -1;

    int nPass = 0, nFail = 0, nSkip = 0;
    long long totalElems = 0;

    std::vector<unsigned char> hSrc, hLb, hNormRaw;
    std::vector<float>         hNormA, hNormCpp;

    for (std::size_t ri = 0; ri < rows.size(); ++ri)
    {
        const Row& r = rows[ri];
        const std::string base = binDir + "/" + r.stem;

        if (!readFile(base + ".src.bin", hSrc) ||
            !readFile(base + ".lb.bin", hLb) ||
            !readFile(base + ".norm.bin", hNormRaw))
        {
            std::printf("[%4zu/%zu] %-20s [SKIP] missing bin\n", ri + 1, rows.size(), r.stem.c_str());
            ++nSkip;
            continue;
        }

        const std::size_t srcElems = (std::size_t)r.W0 * r.H0 * 3;
        if (hSrc.size() != srcElems || hLb.size() != lbElems || hNormRaw.size() != normBytes)
        {
            std::printf("[%4zu/%zu] %-20s [SKIP] bad bin size (src %zu/%zu, lb %zu/%zu, norm %zu/%zu)\n",
                        ri + 1, rows.size(), r.stem.c_str(),
                        hSrc.size(), srcElems, hLb.size(), lbElems, hNormRaw.size(), normBytes);
            ++nSkip;
            continue;
        }

        // -------------------------------------------------------- 1. geometry
        const LetterBoxInfo geo = computeLetterBox(r.W0, r.H0, netW, netH);
        const bool geoOk = (geo.newW == r.newW) && (geo.newH == r.newH) &&
                           (geo.padLeft == r.padL) && (geo.padRight == r.padR) &&
                           (geo.padTop == r.padT) && (geo.padBottom == r.padB);

        // --------------------------------------------------- (re)build plans
        if (r.W0 != curSrcW || r.H0 != curSrcH)
        {
            resizePlan.reset(new CvResizePlan<unsigned char, unsigned char>(r.W0, r.H0, geo.newW, geo.newH));
            borderPlan.reset(new CvCopyMakeBorderPlan<unsigned char, unsigned char>(
                geo.newW, geo.newH, geo.padTop, geo.padBottom, geo.padLeft, geo.padRight));
            cvtPlan.reset(new CvCvtColorPlan<unsigned char, unsigned char>(netW, netH, ColorCode::BGR2RGB));
            chwPlan.reset(new CvHwcToChwPlan<unsigned char, unsigned char>(netW, netH, 3));
            vecPlan.reset(new CvPreprocessVecPlan(
                CvPreprocessVecPlan::byDivisor(r.W0, r.H0, netW, netH, 114.0, 255.0f)));
            if (!resizePlan->ok() || !borderPlan->ok() || !cvtPlan->ok() ||
                !chwPlan->ok() || !vecPlan->ok())
            {
                std::fprintf(stderr, "[error] plan build failed at %s (%dx%d)\n",
                             r.stem.c_str(), r.W0, r.H0);
                return 2;
            }
            curSrcW = r.W0;
            curSrcH = r.H0;
        }

        // -------------------------------------------------------- device mem
        if (srcElems > capSrc)
        {
            if (dSrc) cudaFree(dSrc);
            cudaMust(cudaMalloc(&dSrc, srcElems), "cudaMalloc dSrc");
            capSrc = srcElems;
        }
        const std::size_t newElems = (std::size_t)geo.newW * geo.newH * 3;
        if (newElems > capNew)
        {
            if (dResized) cudaFree(dResized);
            cudaMust(cudaMalloc(&dResized, newElems), "cudaMalloc dResized");
            capNew = newElems;
        }
        if (!dPadded) { cudaMust(cudaMalloc(&dPadded, lbElems), "cudaMalloc dPadded"); }
        if (!dRgb)    { cudaMust(cudaMalloc(&dRgb,    lbElems), "cudaMalloc dRgb");    }
        if (!dChw)    { cudaMust(cudaMalloc(&dChw,    lbElems), "cudaMalloc dChw");    }
        if (!dNorm)   { cudaMust(cudaMalloc(&dNorm, normBytes), "cudaMalloc dNorm");   }

        cudaMust(cudaMemcpy(dSrc, hSrc.data(), srcElems, cudaMemcpyHostToDevice), "H2D src");

        // -------------------------------------------- 2. u8 letterbox (5 chain)
        const double padScalar[3] = { 114.0, 114.0, 114.0 };
        resizePlan->launch(nullptr, 1, dSrc, dResized, 3);
        borderPlan->launch(nullptr, 1, dResized, dPadded, 3, BorderType::Constant, padScalar);
        cvtPlan->launch(nullptr, 1, dPadded, dRgb, 3);
        chwPlan->launch(nullptr, 1, dRgb, dChw);

        // ------------------------------------------- 3. fused normalize (f32)
        vecPlan->launch(nullptr, 1, dSrc, dNorm);

        cudaMust(cudaGetLastError(), "kernel launch");
        cudaMust(cudaDeviceSynchronize(), "cudaDeviceSynchronize");

        // ---------------------------------------------------------- read back
        std::vector<unsigned char> hLbCpp(lbElems);
        cudaMust(cudaMemcpy(hLbCpp.data(), dPadded, lbElems, cudaMemcpyDeviceToHost), "D2H lb");
        hNormCpp.resize(lbElems);
        cudaMust(cudaMemcpy(hNormCpp.data(), dNorm, normBytes, cudaMemcpyDeviceToHost), "D2H norm");

        // ------------------------------------------------------------ compare
        long long lbDiff = 0, nmDiff = 0;
        long long firstLb = -1, firstNm = -1;
        for (std::size_t i = 0; i < lbElems; ++i)
        {
            if (hLbCpp[i] != hLb[i]) { ++lbDiff; if (firstLb < 0) firstLb = (long long)i; }
        }
        const float* ref = reinterpret_cast<const float*>(hNormRaw.data());
        double maxAbs = 0.0;
        for (std::size_t i = 0; i < lbElems; ++i)
        {
            const float x = hNormCpp[i], y = ref[i];
            const bool same = (x == y) || (std::isnan(x) && std::isnan(y));
            if (!same)
            {
                ++nmDiff;
                if (firstNm < 0) firstNm = (long long)i;
                const double d = std::fabs((double)x - (double)y);
                if (d > maxAbs) maxAbs = d;
            }
        }

        totalElems += (long long)lbElems;
        const bool ok = geoOk && (lbDiff == 0) && (nmDiff == 0);
        if (ok) ++nPass; else ++nFail;

        std::printf("[%4zu/%zu] %-20s %4dx%-4d geo=%s lb=%s norm=%s\n",
                    ri + 1, rows.size(), r.stem.c_str(), r.W0, r.H0,
                    geoOk ? "OK" : "BAD", lbDiff ? "DIFF" : "OK", nmDiff ? "DIFF" : "OK");
        if (!geoOk)
        {
            std::printf("         geometry mismatch: cpp new=%dx%d pad=%d/%d/%d/%d  "
                        "py new=%dx%d pad=%d/%d/%d/%d\n",
                        geo.newW, geo.newH, geo.padLeft, geo.padRight, geo.padTop, geo.padBottom,
                        r.newW, r.newH, r.padL, r.padR, r.padT, r.padB);
        }
        if (firstLb >= 0)
        {
            const long long c = firstLb / (long long)(netW * netH);
            const long long rem = firstLb % (long long)(netW * netH);
            std::printf("         first u8  mismatch #%lld [c=%lld y=%lld x=%lld] cpp=%u py=%u\n",
                        firstLb, c, rem / netW, rem % netW,
                        (unsigned)hLbCpp[(std::size_t)firstLb], (unsigned)hLb[(std::size_t)firstLb]);
        }
        if (firstNm >= 0)
        {
            const long long plane = (long long)netW * netH;
            const long long c = firstNm / plane;
            const long long rem = firstNm % plane;
            std::printf("         first f32 mismatch #%lld [c=%lld y=%lld x=%lld] cpp=%.9g py=%.9g  max|d|=%g\n",
                        firstNm, c, rem / netW, rem % netW,
                        (double)hNormCpp[(std::size_t)firstNm], (double)ref[(std::size_t)firstNm], maxAbs);
        }
    }

    if (dSrc)     cudaFree(dSrc);
    if (dResized) cudaFree(dResized);
    if (dPadded)  cudaFree(dPadded);
    if (dRgb)     cudaFree(dRgb);
    if (dChw)     cudaFree(dChw);
    if (dNorm)    cudaFree(dNorm);

    std::printf("\n============ summary ============\n");
    std::printf("  compared   : %d images  (%lld elements each check, u8+f32)\n", nPass + nFail, (long long)lbElems);
    std::printf("  PASS       : %d\n", nPass);
    std::printf("  FAIL       : %d\n", nFail);
    std::printf("  SKIP       : %d\n", nSkip);
    std::printf("  total f32 elements compared : %lld\n", totalElems);
    std::printf("  verdict    : %s\n",
                (nFail == 0 && nPass > 0)
                    ? "cpp preprocessing is BIT-IDENTICAL to the Python reference (rect=False)"
                    : "MISMATCH -- see per-image lines above");

    return (nFail == 0 && nPass > 0) ? 0 : 1;
}
