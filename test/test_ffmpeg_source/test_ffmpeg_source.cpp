// =============================================================================
//  test_ffmpeg_source
// -----------------------------------------------------------------------------
//  Checks that the two decode backends really are interchangeable.
//
//  What it does (all against ONE file, so timing differences are decode only):
//    1. OpenCV       -- the baseline every existing command line uses
//    2. FFmpeg+none  -- libavcodec on the CPU + libswscale
//    3. FFmpeg+nvdec -- NVDEC + the nv12ToBgr kernel
//  For each: read frames to EOF (batch = 1), then compare
//    * resolution agreement
//    * frame-count agreement
//    * BGR pixel difference between (2) and (3) -- ENFORCED. Both honour the
//      stream's colour metadata, so the only expected difference is the chroma
//      upsampling: libswscale interpolates, the nv12ToBgr kernel duplicates.
//    * BGR pixel difference between (1) and (2) -- reported only. OpenCV goes
//      through sws_scale(), which ignores AVFrame::colorspace and converts
//      everything as BT.601; the FFmpeg backends follow the VUI, so on a BT.709
//      stream the two legitimately differ (measured ~6.8 LSB mean).
//    * decode throughput (ms/frame), printed for the record
//
//  Usage:
//      test_ffmpeg_source [video] [--dump <dir>]
//  Default video: data/people.mp4 (resolved against the project root).
//  --dump writes the first frame of each path as P6 PPM for offline diffing.
// =============================================================================
#include "trt_alpha/core/buffer_view.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/datasource/ffmpeg_source.hpp"
#include "trt_alpha/datasource/i_data_source.hpp"
#include "trt_alpha/datasource/opencv_source.hpp"
#include "trt_alpha/datasource/source_config.hpp"
#include "trt_alpha/datasource/source_factory.hpp"

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>

#if defined(_WIN32)
#define WIN32_LEAN_AND_MEAN
#include <windows.h>
#endif
#include <memory>
#include <string>
#include <vector>

namespace {

//! Process CPU time in milliseconds (kernel + user, summed over all threads).
//!
//! std::clock() cannot be used on Windows: MSVC returns wall-clock time for it,
//! which would make the CPU column identical to the wall-clock column and hide
//! exactly the effect this column exists to show.
double processCpuMs()
{
#if defined(_WIN32)
    FILETIME create, exit, kernel, user;
    if (GetProcessTimes(GetCurrentProcess(), &create, &exit, &kernel, &user) == 0)
    {
        return 0.0;
    }
    ULARGE_INTEGER k, u;
    k.LowPart = kernel.dwLowDateTime;
    k.HighPart = kernel.dwHighDateTime;
    u.LowPart = user.dwLowDateTime;
    u.HighPart = user.dwHighDateTime;
    return static_cast<double>(k.QuadPart + u.QuadPart) / 10000.0;   // 100 ns ticks
#else
    return 1000.0 * static_cast<double>(std::clock()) / static_cast<double>(CLOCKS_PER_SEC);
#endif
}

// Read the stream to EOF so that the decoder's flush path is covered, with a cap
// so a broken source cannot hang the run.
constexpr int kMaxFrames = 1000;

struct Run
{
    std::string label;
    int frames = 0;
    int width = 0;
    int height = 0;
    double msPerFrame = 0.0;      //!< wall clock (frames/s is its reciprocal)
    double cpuMsPerFrame = 0.0;   //!< process CPU time, i.e. the cost to the CPU

    //! First decoded frame, tightly packed BGR (kept for the pixel comparison).
    std::vector<std::uint8_t> first;
};

bool readAll(const std::string& label,
             const trt_alpha::datasource::SourceConfig& cfg,
             Run& out,
             bool keepFirst)
{
    using Clock = std::chrono::steady_clock;

    std::unique_ptr<trt_alpha::datasource::IDataSource> src;
    try
    {
        src = trt_alpha::datasource::makeSource(cfg);
    }
    catch (const std::exception& e)
    {
        std::printf("[FAIL] %-22s cannot open: %s\n", label.c_str(), e.what());
        return false;
    }

    out.label = label;

    trt_alpha::core::Batch batch;
    // Timing starts after the first frame: opening the cuvid context, allocating
    // the pinned staging buffer and loading the kernel module happen once and
    // would otherwise dominate a short clip.
    auto   t0 = Clock::now();
    double cpu0 = processCpuMs();
    while (out.frames < kMaxFrames)
    {
        if (!src->next(batch))
        {
            break;
        }
        if (batch.validCount <= 0 || batch.views.empty())
        {
            break;
        }

        // The source may fill the host copy asynchronously; this test is the
        // consumer, so it has to wait exactly like a real one does before reading
        // the pixels. The wait is part of the measured cost on purpose -- it is
        // what the pipeline pays too.
        if (batch.hostReady)
        {
            batch.hostReady->wait();
        }

        const trt_alpha::core::BufferView& v = batch.views[0];
        if (v.width <= 0 || v.height <= 0 || v.stride < v.width * 3)
        {
            std::printf("[FAIL] %-22s bogus view %dx%d stride=%d\n", label.c_str(),
                        v.width, v.height, v.stride);
            return false;
        }
        if (out.width == 0)
        {
            out.width = v.width;
            out.height = v.height;
            if (keepFirst)
            {
                out.first.resize(static_cast<std::size_t>(v.width) * 3u *
                                 static_cast<std::size_t>(v.height));
                for (int y = 0; y < v.height; ++y)
                {
                    std::memcpy(out.first.data() + static_cast<std::size_t>(y) * v.width * 3u,
                                v.data + static_cast<std::size_t>(y) * v.stride,
                                static_cast<std::size_t>(v.width) * 3u);
                }
            }
        }
        else if (v.width != out.width || v.height != out.height)
        {
            std::printf("[FAIL] %-22s resolution changed mid-stream (%dx%d -> %dx%d)\n",
                        label.c_str(), out.width, out.height, v.width, v.height);
            return false;
        }

        if (v.space != trt_alpha::core::MemorySpace::Host)
        {
            std::printf("[FAIL] %-22s frame is not on the Host\n", label.c_str());
            return false;
        }
        if (v.format != trt_alpha::core::PixelFormat::BGR8)
        {
            std::printf("[FAIL] %-22s frame is not BGR8\n", label.c_str());
            return false;
        }

        ++out.frames;
        if (out.frames == 1)
        {
            // First frame done (one-time setup paid): start measuring.
            t0 = Clock::now();
            cpu0 = processCpuMs();
        }
    }
    const auto t1 = Clock::now();
    const double cpu1 = processCpuMs();

    const double ms =
        std::chrono::duration_cast<std::chrono::duration<double, std::milli>>(t1 - t0).count();

    // The first frame is excluded from the averages (see the timer reset above).
    const int timed = (out.frames > 1) ? (out.frames - 1) : 1;
    out.msPerFrame = ms / static_cast<double>(timed);

    // CPU time / frame -- the number that says how much CPU a backend eats. This
    // is where hardware decode pays off even when wall clock does not.
    out.cpuMsPerFrame = (cpu1 - cpu0) / static_cast<double>(timed);
    return out.frames > 0;
}

// P6 PPM of a tightly packed BGR frame -- no image library needed, and every
// analysis tool (Python/PIL/ImageMagick) reads it.
void writePpm(const std::string& path, const std::vector<std::uint8_t>& bgr,
              int width, int height)
{
    if (bgr.empty() || width <= 0 || height <= 0)
    {
        return;
    }
    std::FILE* f = std::fopen(path.c_str(), "wb");
    if (f == nullptr)
    {
        std::printf("[WARN] cannot write %s\n", path.c_str());
        return;
    }
    std::fprintf(f, "P6\n%d %d\n255\n", width, height);
    for (std::size_t i = 0; i + 2 < bgr.size(); i += 3)
    {
        const std::uint8_t rgb[3] = {bgr[i + 2], bgr[i + 1], bgr[i]};
        std::fwrite(rgb, 1, 3, f);
    }
    std::fclose(f);
    std::printf("wrote %s\n", path.c_str());
}

// -----------------------------------------------------------------------------
//  Pixel comparison between two runs of the same file.
//
//  `enforce` is false for pairs that are expected to differ for a principled
//  reason (see the call sites); the numbers are still printed so a regression
//  shows up in the log.
// -----------------------------------------------------------------------------
void comparePixels(const char* what,
                   const std::vector<std::uint8_t>& a,
                   const std::vector<std::uint8_t>& b,
                   bool enforce,
                   double maxMean,
                   double maxPctOver8,
                   int& failures)
{
    if (a.empty() || a.size() != b.size())
    {
        std::printf("[FAIL] %s: frame sizes differ (%zu vs %zu)\n", what, a.size(), b.size());
        ++failures;
        return;
    }

    double sum = 0.0;
    int    maxDiff = 0;
    int    over8 = 0;
    const std::size_t n = a.size();
    for (std::size_t i = 0; i < n; ++i)
    {
        const int d = std::abs(static_cast<int>(a[i]) - static_cast<int>(b[i]));
        sum += static_cast<double>(d);
        if (d > maxDiff) { maxDiff = d; }
        if (d > 8)       { ++over8; }
    }
    const double mean = sum / static_cast<double>(n);
    const double pctOver8 = 100.0 * static_cast<double>(over8) / static_cast<double>(n);

    std::printf("pixel diff %-26s mean %7.3f  max %3d  >8 LSB %6.3f%%  %s\n", what,
                mean, maxDiff, pctOver8, enforce ? "(enforced)" : "(report only)");

    if (enforce && (mean > maxMean || pctOver8 > maxPctOver8))
    {
        std::printf("[FAIL] %s: mean %.3f (limit %.3f) / >8 LSB %.3f%% (limit %.3f%%)\n",
                    what, mean, maxMean, pctOver8, maxPctOver8);
        ++failures;
    }
}

}  // namespace

int main(int argc, char** argv)
{
    using namespace trt_alpha;

    std::string video = "data/people.mp4";
    std::string dumpDir;
    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--dump" && i + 1 < argc)
        {
            dumpDir = argv[++i];
        }
        else if (!a.empty() && a[0] != '-')
        {
            video = a;
        }
    }

    if (!datasource::ffmpegBackendAvailable())
    {
        // A build without FFmpeg must still fail loudly when asked for it --
        // silently handing back an OpenCVSource would make a benchmark compare
        // the wrong two things.
        datasource::SourceConfig cfg;
        cfg.type = datasource::SourceType::Video;
        cfg.path = video;
        cfg.decodeBackend = datasource::DecodeBackend::FFmpeg;
        try
        {
            auto s = datasource::makeSource(cfg);
            std::printf("[FAIL] makeSource returned a source although FFmpeg is not built in\n");
            return 1;
        }
        catch (const std::exception& e)
        {
            std::printf("[PASS] ffmpeg requested without support -> %s\n", e.what());
        }

        std::printf("[SKIP] this build has no FFmpeg support (TRT_ALPHA_WITH_FFMPEG=OFF)\n");
        return 77;   // CTest: SKIP_RETURN_CODE
    }

    datasource::SourceConfig base;
    base.type = datasource::SourceType::Video;
    base.path = video;
    base.batchSize = 1;

    datasource::SourceConfig cv = base;
    cv.decodeBackend = datasource::DecodeBackend::OpenCV;

    datasource::SourceConfig ffSw = base;
    ffSw.decodeBackend = datasource::DecodeBackend::FFmpeg;
    ffSw.hwAccel = datasource::HwAccel::None;

    datasource::SourceConfig ffHw = base;
    ffHw.decodeBackend = datasource::DecodeBackend::FFmpeg;
    ffHw.hwAccel = datasource::HwAccel::Nvdec;

    Run rCv, rSw, rHw;
    if (!readAll("opencv", cv, rCv, true))              { return 1; }
    if (!readAll("ffmpeg (software)", ffSw, rSw, true)) { return 1; }

    // NVDEC may legitimately be unavailable (no GPU, no cuvid decoder); that is
    // not a failure of the source, so report and keep the software result.
    const bool haveHw = readAll("ffmpeg (nvdec)", ffHw, rHw, true);

    std::printf("\n%-22s %8s %12s %11s %10s\n", "backend", "frames", "resolution",
                "ms/frame", "cpu ms/fr");
    auto row = [](const Run& r)
    {
        std::printf("%-22s %8d %12s %11.3f %10.3f\n", r.label.c_str(), r.frames,
                    (std::to_string(r.width) + "x" + std::to_string(r.height)).c_str(),
                    r.msPerFrame, r.cpuMsPerFrame);
    };
    row(rCv);
    row(rSw);
    if (haveHw)
    {
        row(rHw);
    }

    int failures = 0;

    // ---- resolution / frame count agreement --------------------------------
    if (rSw.width != rCv.width || rSw.height != rCv.height)
    {
        std::printf("[FAIL] resolution differs: opencv %dx%d vs ffmpeg %dx%d\n",
                    rCv.width, rCv.height, rSw.width, rSw.height);
        ++failures;
    }
    if (rSw.frames != rCv.frames)
    {
        std::printf("[FAIL] frame count differs: opencv %d vs ffmpeg %d\n",
                    rCv.frames, rSw.frames);
        ++failures;
    }
    if (haveHw && (rHw.frames != rSw.frames || rHw.width != rSw.width ||
                   rHw.height != rSw.height))
    {
        std::printf("[FAIL] nvdec disagrees with software: %d frames %dx%d vs %d frames %dx%d\n",
                    rHw.frames, rHw.width, rHw.height, rSw.frames, rSw.width, rSw.height);
        ++failures;
    }

    // ---- pixel differences -------------------------------------------------
    //  ffmpeg-sw vs ffmpeg-nvdec: both honour the stream's VUI, so the only
    //  expected difference is the chroma upsampling (libswscale bilinear vs the
    //  kernel's nearest-neighbour). Enforced -- this is the real proof that the
    //  two decode paths agree, not just that they produce the same frame count.
    //
    //  opencv vs ffmpeg-sw: report only. OpenCV goes through sws_scale(), which
    //  ignores AVFrame::colorspace and converts everything as BT.601; the FFmpeg
    //  backends follow the VUI. On a BT.709 stream that is ~6.8 LSB by design,
    //  in OpenCV's favour of being wrong, not ours.
    std::printf("\n");
    comparePixels("opencv vs ffmpeg-sw", rCv.first, rSw.first,
                  false, 0.0, 0.0, failures);

    if (haveHw)
    {
        comparePixels("ffmpeg-sw vs ffmpeg-nvdec", rSw.first, rHw.first,
                      true, 2.0, 1.0, failures);
    }

    // ---- optional frame dump (for offline analysis) ------------------------
    if (!dumpDir.empty())
    {
        std::printf("\n");
        writePpm(dumpDir + "/cv.ppm", rCv.first, rCv.width, rCv.height);
        writePpm(dumpDir + "/sw.ppm", rSw.first, rSw.width, rSw.height);
        if (haveHw)
        {
            writePpm(dumpDir + "/hw.ppm", rHw.first, rHw.width, rHw.height);
        }
    }

    // ---- factory routing ---------------------------------------------------
    {
        datasource::SourceConfig img;
        img.type = datasource::SourceType::Image;
        img.path = "data/bus.jpg";
        img.batchSize = 1;
        img.decodeBackend = datasource::DecodeBackend::FFmpeg;   // must be ignored
        try
        {
            auto s = datasource::makeSource(img);
            trt_alpha::core::Batch b;
            if (s == nullptr || !s->next(b) || b.validCount != 1)
            {
                std::printf("[FAIL] image source did not produce exactly one frame\n");
                ++failures;
            }
        }
        catch (const std::exception& e)
        {
            std::printf("[FAIL] image source threw: %s\n", e.what());
            ++failures;
        }
    }

    if (failures == 0)
    {
        std::printf("\nALL PASS\n");
        return 0;
    }
    std::printf("\n%d CHECK(S) FAILED\n", failures);
    return 1;
}
