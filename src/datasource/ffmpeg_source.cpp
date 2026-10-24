// =============================================================================
//  trt_alpha :: datasource :: ffmpeg_source（实现）
// -----------------------------------------------------------------------------
//  Video files and network streams through libavformat + libavcodec.
//
//  Two decode paths, one output format:
//
//      HwAccel::Nvdec                     HwAccel::None
//      ------------------------------     -----------------------------
//      h264_cuvid / hevc_cuvid / ...      avcodec's own decoder (CPU)
//      AVFrame = AV_PIX_FMT_CUDA (NV12)   AVFrame = yuv420p / nv12 / ...
//      nv12ToBgr kernel (GPU)             sws_scale (libswscale, CPU)
//      one D2H of the BGR frame           rows land in the batch buffer
//
//  Both end in tightly packed BGR8 host frames, so preprocess / infer /
//  postprocess / render are untouched by the switch. That is the whole point:
//  decode is the only thing that changes.
//
//  Threading: one IDataSource is driven by one source thread; next() blocks on
//  demux/decode. requestStop() is the only cross-thread call.
// =============================================================================
#include "trt_alpha/datasource/ffmpeg_source.hpp"

#if defined(TRT_ALPHA_WITH_FFMPEG) && TRT_ALPHA_WITH_FFMPEG

extern "C" {
#include <libavcodec/avcodec.h>
#include <libavformat/avformat.h>
#include <libavutil/error.h>
#include <libavutil/hwcontext.h>
#include <libavutil/imgutils.h>
#include <libavutil/opt.h>
#include <libavutil/pixdesc.h>
#include <libswscale/swscale.h>
}

#include "trt_alpha/core/gpu_event.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/paths.hpp"
#include "trt_alpha/kernels/ops/nv12_to_bgr.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha::datasource {
namespace {

namespace fs = std::filesystem;

//! "<stem>_<6 digits>" naming, identical to what OpenCVSource produces so
//! --save output looks the same whichever backend decoded it.
std::string stemOf(const std::string& pathOrUrl)
{
    std::string s = pathOrUrl.substr(0, pathOrUrl.find_first_of("?#"));
    while (!s.empty() && s.back() == '/')
    {
        s.pop_back();
    }
    const std::size_t slash = s.find_last_of("/\\");
    std::string name = (slash == std::string::npos) ? s : s.substr(slash + 1);
    const std::size_t dot = name.find_last_of('.');
    if (dot != std::string::npos && dot != 0)
    {
        name.resize(dot);
    }
    if (name.find_first_not_of('.') == std::string::npos)
    {
        name.clear();
    }
    return name.empty() ? std::string("stream") : name;
}

std::string frameSuffix(std::uint64_t index)
{
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%06llu", static_cast<unsigned long long>(index));
    return std::string(buf);
}

std::string avErr2Str(int err)
{
    char buf[AV_ERROR_MAX_STRING_SIZE] = {0};
    if (av_strerror(err, buf, sizeof(buf)) != 0)
    {
        return "error " + std::to_string(err);
    }
    return std::string(buf);
}

//! NVDEC decoder for a codec id, or nullptr when there is none.
const char* cuvidDecoderName(AVCodecID id) noexcept
{
    switch (id)
    {
    case AV_CODEC_ID_H264:       return "h264_cuvid";
    case AV_CODEC_ID_HEVC:       return "hevc_cuvid";
    case AV_CODEC_ID_AV1:        return "av1_cuvid";
    case AV_CODEC_ID_VP9:        return "vp9_cuvid";
    case AV_CODEC_ID_VP8:        return "vp8_cuvid";
    case AV_CODEC_ID_MPEG4:      return "mpeg4_cuvid";
    case AV_CODEC_ID_MPEG2VIDEO: return "mpeg2_cuvid";
    case AV_CODEC_ID_MPEG1VIDEO: return "mpeg1_cuvid";
    case AV_CODEC_ID_VC1:        return "vc1_cuvid";
    case AV_CODEC_ID_MJPEG:      return "mjpeg_cuvid";
    default:                     return nullptr;
    }
}

//! Map the stream's colour metadata onto the kernel's enumerations.
//!
//! This is derived ONCE from AVCodecParameters (i.e. the container / VUI) instead
//! of per frame, on purpose: the software and the cuvid decoders do not always
//! attach the same metadata to the frames they emit, and then the two paths would
//! silently convert with different matrices (measured: 1.6 LSB mean drift on an
//! MJPEG clip). The stream itself is the only answer both paths can agree on.
//!
//! Getting this wrong shifts every colour by a few percent, so the stream wins
//! over a guess.
kernels::ops::YuvMatrix matrixOf(int colorspace, int width, int height)
{
    switch (colorspace)
    {
    case AVCOL_SPC_BT709:
        return kernels::ops::YuvMatrix::BT709;
    case AVCOL_SPC_BT470BG:
    case AVCOL_SPC_SMPTE170M:
        return kernels::ops::YuvMatrix::BT601;
    case AVCOL_SPC_BT2020_NCL:
    case AVCOL_SPC_BT2020_CL:
        return kernels::ops::YuvMatrix::BT2020;
    default:
        // Unspecified: HD content is BT.709 in practice, SD is BT.601.
        return (height >= 720 || width >= 1280) ? kernels::ops::YuvMatrix::BT709
                                                : kernels::ops::YuvMatrix::BT601;
    }
}

kernels::ops::YuvRange rangeOf(int colorRange)
{
    return (colorRange == AVCOL_RANGE_JPEG) ? kernels::ops::YuvRange::Full
                                            : kernels::ops::YuvRange::Limited;
}

//! The same matrix, as the libswscale selector that sws_getCoefficients() wants.
int swsCsp(kernels::ops::YuvMatrix m)
{
    switch (m)
    {
    case kernels::ops::YuvMatrix::BT601:  return SWS_CS_ITU601;
    case kernels::ops::YuvMatrix::BT2020: return SWS_CS_BT2020;
    case kernels::ops::YuvMatrix::BT709:
    default:                              return SWS_CS_ITU709;
    }
}

}  // namespace

// =============================================================================
//  Impl
// =============================================================================
struct FFmpegSource::Impl
{
    explicit Impl(const SourceConfig& c) : cfg(c) {}

    SourceConfig cfg;

    //! Path actually handed to libavformat: an absolute UTF-8 path for local
    //! files (libavformat's Windows file layer expects UTF-8), the URL as-is
    //! otherwise.
    std::string mediaPath;

    AVFormatContext* fmt = nullptr;   //!< demuxer
    AVCodecContext*  dec = nullptr;   //!< decoder (software or cuvid)
    AVBufferRef*     hwDev = nullptr; //!< CUDA device for NVDEC
    SwsContext*      sws = nullptr;   //!< software path only
    AVFrame*         frame = nullptr; //!< reused decode target
    AVPacket*        pkt = nullptr;

    int  streamIndex = -1;
    bool useNvdec = false;
    bool draining = false;   //!< a flush packet was already sent
    bool eof = false;

    int W = 0;               //!< decoded width  (fixed after the first frame)
    int H = 0;               //!< decoded height

    //! Colour conversion parameters, taken once from the stream (see matrixOf).
    kernels::ops::YuvMatrix colorMatrix = kernels::ops::YuvMatrix::BT709;
    kernels::ops::YuvRange  colorRange  = kernels::ops::YuvRange::Limited;

    //! Marks the point where the nv12ToBgr kernel has finished reading the cuvid
    //! surface. Those surfaces are recycled by avcodec_receive_frame(), so the
    //! kernel must be done before the next receive -- this is the only thing this
    //! source still waits for. The host copy that follows is NOT waited on.
    //!
    //! The wait is deliberately NOT next to the record: see decodeNext().
    core::GpuEvent kernelDone;
    bool kernelPending = false;

    std::uint64_t nextFrameIndex = 0;
    std::string   stem;              //!< "<video stem>" for --save naming
    bool          bannerLogged = false;
    std::atomic<bool> stop{false};

    // ---- lifetime ----
    void open();
    void close() noexcept;
    void seekToStart();

    // ---- decode ----
    bool decodeNext();                       //!< false = end of stream
    void toBgrInto(std::uint8_t* dst, int dstStride, std::uint8_t* devSlot,
                   bool copyToHost);

    // ---- helpers ----
    void logBanner() const;
};

void FFmpegSource::Impl::logBanner() const
{
    TRT_LOG_INFO("FFmpegSource: " << cfg.path << " -> " << W << "x" << H << ", "
                 << (useNvdec ? "NVDEC (cuVID)" : "libavcodec software")
                 << ", batch=" << cfg.batchSize);
}

void FFmpegSource::Impl::open()
{
    if (cfg.type != SourceType::Video)
    {
        throw std::runtime_error(
            "FFmpegSource: only video files and streams are supported (the OpenCV "
            "backend owns images, image directories and cameras)");
    }
    if (cfg.path.empty())
    {
        throw std::runtime_error("FFmpegSource: empty source path");
    }

    avformat_network_init();

    // Local files: resolve against the project root and fail with the same
    // readable "where did we look" message the other sources produce. libavformat
    // on Windows converts UTF-8 to wide chars internally, so hand it UTF-8.
    if (core::Paths::isUrl(cfg.path))
    {
        mediaPath = cfg.path;
    }
    else
    {
        const fs::path p = core::Paths::requireFile(cfg.path, "video");
        mediaPath = p.u8string();
    }
    stem = stemOf(mediaPath);

    AVDictionary* opts = nullptr;
    // Network streams: cap the blocking time so an unreachable camera fails fast
    // instead of hanging the source thread forever.
    if (cfg.timeoutMs > 0 && core::Paths::isUrl(cfg.path))
    {
        av_dict_set(&opts, "rw_timeout",
                    std::to_string(static_cast<long long>(cfg.timeoutMs) * 1000).c_str(), 0);
        // TCP by default: UDP RTSP loses whole GOPs on a lossy link.
        av_dict_set(&opts, "rtsp_transport", "tcp", 0);
    }

    int ret = avformat_open_input(&fmt, mediaPath.c_str(), nullptr, &opts);
    av_dict_free(&opts);
    if (ret < 0)
    {
        throw std::runtime_error("FFmpegSource: cannot open '" + mediaPath + "': " +
                                 avErr2Str(ret));
    }

    ret = avformat_find_stream_info(fmt, nullptr);
    if (ret < 0)
    {
        throw std::runtime_error("FFmpegSource: no stream info for '" + mediaPath +
                                 "': " + avErr2Str(ret));
    }

    const AVCodec* codec = nullptr;
    streamIndex = av_find_best_stream(fmt, AVMEDIA_TYPE_VIDEO, -1, -1, &codec, 0);
    if (streamIndex < 0 || codec == nullptr)
    {
        throw std::runtime_error("FFmpegSource: no video stream in '" + mediaPath + "'");
    }

    AVStream* st = fmt->streams[streamIndex];
    const AVCodecID codecId = st->codecpar->codec_id;

    // Colour metadata is stream-level, so both decode paths convert identically
    // regardless of what either decoder puts on its frames.
    colorMatrix = matrixOf(st->codecpar->color_space,
                           st->codecpar->width, st->codecpar->height);
    colorRange  = rangeOf(st->codecpar->color_range);

    // ---- pick the decoder -------------------------------------------------
    bool wantNvdec = (cfg.hwAccel == HwAccel::Nvdec || cfg.hwAccel == HwAccel::Auto);
    std::string nvdecWhy;

    if (wantNvdec)
    {
        const char* name = cuvidDecoderName(codecId);
        if (name == nullptr)
        {
            nvdecWhy = "no NVDEC decoder for this codec";
        }
        else
        {
            ret = av_hwdevice_ctx_create(&hwDev, AV_HWDEVICE_TYPE_CUDA, nullptr, nullptr, 0);
            if (ret < 0)
            {
                // The CUDA driver may be missing or busy; keep the message short,
                // the full text would be a 200-character driver dump.
                nvdecWhy = "CUDA device unavailable (" + avErr2Str(ret) + ")";
            }
            else
            {
                const AVCodec* hwCodec = avcodec_find_decoder_by_name(name);
                if (hwCodec == nullptr)
                {
                    nvdecWhy = std::string("decoder '") + name + "' not present in this FFmpeg build";
                }
                else
                {
                    codec = hwCodec;
                    useNvdec = true;
                }
            }
        }
    }

    if (!useNvdec)
    {
        if (cfg.hwAccel == HwAccel::Nvdec)
        {
            av_buffer_unref(&hwDev);
            throw std::runtime_error("FFmpegSource: NVDEC requested but unavailable: " + nvdecWhy);
        }
        if (wantNvdec)
        {
            TRT_LOG_WARN("FFmpegSource: NVDEC unavailable (" << nvdecWhy
                         << "), falling back to software decode");
        }
        av_buffer_unref(&hwDev);
    }

    // ---- open the codec ---------------------------------------------------
    dec = avcodec_alloc_context3(codec);
    if (dec == nullptr)
    {
        throw std::runtime_error("FFmpegSource: avcodec_alloc_context3 failed");
    }
    ret = avcodec_parameters_to_context(dec, st->codecpar);
    if (ret < 0)
    {
        throw std::runtime_error("FFmpegSource: cannot copy codec parameters: " + avErr2Str(ret));
    }

    dec->pkt_timebase = st->time_base;

    AVDictionary* decOpts = nullptr;
    if (useNvdec)
    {
        dec->hw_device_ctx = av_buffer_ref(hwDev);
        // A tiny surface pool stalls the decoder (it blocks waiting for a free
        // surface); the default 20 is fine but 16-24 measured better than 8.
        av_dict_set(&decOpts, "surfaces", "24", 0);
    }
    else
    {
        // Software decode: let FFmpeg pick the thread count, it scales well and
        // this path is only used when there is no GPU decoder.
        dec->thread_count = 0;
    }

    ret = avcodec_open2(dec, codec, &decOpts);
    av_dict_free(&decOpts);
    if (ret < 0)
    {
        throw std::runtime_error(std::string("FFmpegSource: cannot open decoder '") +
                                 codec->name + "': " + avErr2Str(ret));
    }

    frame = av_frame_alloc();
    pkt = av_packet_alloc();
    if (frame == nullptr || pkt == nullptr)
    {
        throw std::runtime_error("FFmpegSource: frame/packet allocation failed");
    }
}

void FFmpegSource::Impl::close() noexcept
{
    // A kernel may still be reading the last decoded surface; freeing the decoder
    // first would hand that surface back to the driver underneath it.
    if (kernelPending)
    {
        kernelDone.wait();
        kernelPending = false;
    }
    if (sws != nullptr)      { sws_freeContext(sws); sws = nullptr; }
    if (pkt != nullptr)      { av_packet_free(&pkt); }
    if (frame != nullptr)    { av_frame_free(&frame); }
    if (dec != nullptr)      { avcodec_free_context(&dec); }
    av_buffer_unref(&hwDev);
    if (fmt != nullptr)      { avformat_close_input(&fmt); }
}

void FFmpegSource::Impl::seekToStart()
{
    const int ret = av_seek_frame(fmt, streamIndex, 0, AVSEEK_FLAG_BACKWARD);
    if (ret < 0)
    {
        eof = true;
        return;
    }
    avcodec_flush_buffers(dec);
    draining = false;
    eof = false;
}

bool FFmpegSource::Impl::decodeNext()
{
    // cuvid recycles surfaces inside avcodec_receive_frame(), so the previous
    // frame's nv12ToBgr kernel has to be finished by the time we get here -- but
    // not a single instruction earlier. Waiting at this end instead of right
    // after the kernel launch puts the whole rest of the batch in between (a few
    // frames of bookkeeping, the result push, sometimes a wait for queue space),
    // which is more than enough for a ~25 us kernel to land. Measured effect:
    // the 79 us/frame event wait drops to near zero.
    if (kernelPending)
    {
        kernelDone.wait();
        kernelPending = false;
    }

    while (!stop.load(std::memory_order_relaxed))
    {
        const int ret = avcodec_receive_frame(dec, frame);
        if (ret == 0)
        {
            if (W == 0)
            {
                W = frame->width;
                H = frame->height;
            }
            return true;
        }
        if (ret == AVERROR_EOF)
        {
            return false;
        }
        if (ret != AVERROR(EAGAIN))
        {
            throw std::runtime_error("FFmpegSource: decode failed: " + avErr2Str(ret));
        }

        // Need more input.
        int rd = av_read_frame(fmt, pkt);
        if (rd == AVERROR_EOF)
        {
            if (draining)
            {
                return false;   // flush already requested, nothing left
            }
            avcodec_send_packet(dec, nullptr);   // enter flush mode
            draining = true;
            continue;
        }
        if (rd < 0)
        {
            throw std::runtime_error("FFmpegSource: read failed: " + avErr2Str(rd));
        }

        if (pkt->stream_index == streamIndex)
        {
            const int sr = avcodec_send_packet(dec, pkt);
            av_packet_unref(pkt);
            if (sr < 0 && sr != AVERROR(EAGAIN))
            {
                throw std::runtime_error("FFmpegSource: send_packet failed: " + avErr2Str(sr));
            }
        }
        else
        {
            av_packet_unref(pkt);
        }
    }
    return false;
}

void FFmpegSource::Impl::toBgrInto(std::uint8_t* dst, int dstStride,
                                   std::uint8_t* devSlot, bool copyToHost)
{
    if (!useNvdec)
    {
        // Software path: the frame is in host memory already, let sws do it.
        sws = sws_getCachedContext(sws, frame->width, frame->height,
                                   static_cast<AVPixelFormat>(frame->format),
                                   frame->width, frame->height, AV_PIX_FMT_BGR24,
                                   SWS_BILINEAR, nullptr, nullptr, nullptr);
        if (sws == nullptr)
        {
            throw std::runtime_error("FFmpegSource: sws_getCachedContext failed");
        }

        // sws_scale() is the legacy entry point: it ignores AVFrame::colorspace
        // and converts everything as BT.601. On a BT.709 stream that is a 6.8 LSB
        // mean drift against the NVDEC path (which reads the VUI), i.e. the two
        // backends would disagree on colour. Push the stream's own matrix/range
        // down instead so both paths agree.
        const int* coeffs = sws_getCoefficients(swsCsp(colorMatrix));
        const int srcRange = (colorRange == kernels::ops::YuvRange::Full) ? 1 : 0;  // 0 = tv
        const int dstRange = 1;                                                     // RGB is full
        if (coeffs == nullptr ||
            sws_setColorspaceDetails(sws, coeffs, srcRange, coeffs, dstRange,
                                     0, 1 << 16, 1 << 16) < 0)
        {
            throw std::runtime_error("FFmpegSource: sws_setColorspaceDetails failed");
        }
        std::uint8_t* outData[4] = {dst, nullptr, nullptr, nullptr};
        int outLines[4] = {dstStride, 0, 0, 0};
        const int rows = sws_scale(sws, frame->data, frame->linesize, 0, frame->height,
                                   outData, outLines);
        if (rows != frame->height)
        {
            throw std::runtime_error("FFmpegSource: sws_scale produced " +
                                     std::to_string(rows) + " rows, expected " +
                                     std::to_string(frame->height));
        }
        return;
    }

    // Hardware path: frame->data[0]/[1] are DEVICE pointers (Y plane, interleaved
    // UV plane). Convert into this batch's device slot and stop there -- those
    // pixels are what the pre-process kernel reads, so nothing has to travel back
    // to the host for inference.
    if (devSlot == nullptr)
    {
        throw std::runtime_error("FFmpegSource: NVDEC path without a device slot");
    }

    cudaError_t e = kernels::ops::nv12ToBgr(
        0, frame->data[0], frame->data[1],
        frame->linesize[0], frame->linesize[1],
        devSlot, W * 3, W, H, colorMatrix, colorRange);
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("FFmpegSource: nv12ToBgr failed: ") +
                                 cudaGetErrorString(e));
    }

    // cuvid hands its surfaces back to the decoder pool, and the next
    // avcodec_receive_frame() may reuse the one the kernel is still reading.
    // Only the record happens here; the wait is in decodeNext(), because that is
    // the last moment that is still safe and it is far enough away for the
    // kernel to have finished long before we get there.
    kernelDone.record();
    kernelPending = true;

    if (!copyToHost)
    {
        // Nobody reads a host frame (--show and --save are both off): the batch
        // is consumed straight out of device memory, so the 6.2 MB copy simply
        // does not happen. On a 1080p frame that is 0.475 ms of 2.14 ms.
        return;
    }

    // The renderer still draws into a host BGR image, so one copy down is
    // unavoidable. It is queued and left running: the batch carries a completion
    // mark and whoever reads the pixels waits there, which is a few ms later.
    e = cudaMemcpy2DAsync(dst, static_cast<std::size_t>(dstStride),
                          devSlot, static_cast<std::size_t>(W) * 3u,
                          static_cast<std::size_t>(W) * 3u,
                          static_cast<std::size_t>(H),
                          cudaMemcpyDeviceToHost, nullptr);
    if (e != cudaSuccess)
    {
        throw std::runtime_error(std::string("FFmpegSource: D2H copy failed: ") +
                                 cudaGetErrorString(e));
    }
}
// =============================================================================
//  FFmpegSource
// =============================================================================
FFmpegSource::FFmpegSource(const SourceConfig& cfg)
    : m_impl(std::make_unique<Impl>(cfg))
{
    m_impl->open();
    TRT_LOG_DEBUG("FFmpegSource: opened, waiting for the first decoded frame");
}

FFmpegSource::~FFmpegSource()
{
    if (m_impl)
    {
        m_impl->close();
    }
}

void FFmpegSource::requestStop()
{
    m_impl->stop.store(true);
}

const char* FFmpegSource::typeName() const noexcept
{
    return m_impl->useNvdec ? "ffmpeg-nvdec" : "ffmpeg";
}

bool FFmpegSource::next(core::Batch& out)
{
    Impl& s = *m_impl;

    if (s.stop.load(std::memory_order_relaxed))
    {
        return false;
    }

    const int want = (s.cfg.batchSize > 0) ? s.cfg.batchSize : 1;

    out = core::Batch{};
    out.sourceId = s.cfg.sourceId;
    out.firstFrameIndex = s.nextFrameIndex;

    // Does anybody actually want the pixels in RAM? Only the renderer does. When
    // it is absent the NVDEC path can keep everything in device memory; the
    // software path cannot (sws_scale writes to the host), so it keeps its copy
    // regardless -- and that copy is what the model then uploads, as before.
    const bool hostCopyNeeded = s.cfg.needHostCopy || !s.useNvdec;

    for (int i = 0; i < want; ++i)
    {
        if (s.stop.load(std::memory_order_relaxed))
        {
            break;
        }

        bool got = s.decodeNext();
        if (!got && i == 0 && s.cfg.loop && !core::Paths::isUrl(s.cfg.path))
        {
            s.seekToStart();
            got = s.decodeNext();
        }
        if (!got)
        {
            if (i == 0)
            {
                s.eof = true;
                return false;
            }
            break;   // short batch: the tail slots stay zeroed
        }

        // Frame 0 owns the geometry; a later frame that disagrees is a source
        // bug (variable-resolution stream), not something to paper over.
        if (i == 0)
        {
            const std::size_t oneFrame =
                static_cast<std::size_t>(s.W) * static_cast<std::size_t>(s.H) * 3u;

            if (hostCopyNeeded)
            {
                // NVDEC fills this host copy with cudaMemcpy2DAsync, and only
                // page-locked memory makes that call return immediately: against
                // pageable memory the driver copies synchronously into its own
                // staging buffer, which cost the decode thread 793 us per 1080p
                // frame (measured) -- the "async" copy was not async at all. The
                // software path writes with sws_scale and never touches CUDA, so
                // it keeps plain pageable memory.
                out.buffer = s.useNvdec
                                 ? core::Buffer::createPinned(s.W * want, s.H, 3,
                                                              core::DataType::UInt8)
                                 : core::Buffer::createHost(s.W * want, s.H, 3,
                                                            core::DataType::UInt8);
            }

            if (s.useNvdec)
            {
                // The BGR result stays in device memory and the model reads it
                // there, so the per-frame host-to-device upload disappears.
                out.deviceBuffer = core::Buffer::createDevice(s.W * want, s.H, 3,
                                                              core::DataType::UInt8);
                if (hostCopyNeeded)
                {
                    out.hostReady = std::make_shared<core::GpuEvent>();
                }
            }

            // `views` describes whichever block holds the pixels this batch will
            // be read from. With a host copy that is the host block and `space`
            // says Host; without one it is the device block and `space` says
            // Device -- the layout is the same either way, so the frame count and
            // geometry stay discoverable.
            out.views.clear();
            out.views.reserve(static_cast<std::size_t>(want));
            const std::uint8_t* base =
                (out.buffer != nullptr) ? out.buffer->data() : out.deviceBuffer->data();
            for (int k = 0; k < want; ++k)
            {
                core::BufferView v;
                v.data = base + static_cast<std::size_t>(k) * oneFrame;
                v.width = s.W;
                v.height = s.H;
                v.stride = s.W * 3;
                v.channels = 3;
                v.dtype = core::DataType::UInt8;
                v.space = (out.buffer != nullptr) ? core::MemorySpace::Host
                                                  : core::MemorySpace::Device;
                v.format = core::PixelFormat::BGR8;
                out.views.push_back(v);
            }
        }
        else if (s.frame->width != s.W || s.frame->height != s.H)
        {
            throw std::runtime_error(
                "FFmpegSource: frame " + std::to_string(s.nextFrameIndex) + " is " +
                std::to_string(s.frame->width) + "x" + std::to_string(s.frame->height) +
                " but this batch is " + std::to_string(s.W) + "x" + std::to_string(s.H) +
                " (from frame 0); frames within one batch must share the same resolution");
        }

        const core::BufferView& v = out.views[static_cast<std::size_t>(i)];
        std::uint8_t* dst = nullptr;
        if (out.buffer != nullptr)
        {
            dst = out.buffer->mutableData() +
                  static_cast<std::size_t>(i) * static_cast<std::size_t>(v.stride) *
                      static_cast<std::size_t>(v.height);
        }
        std::uint8_t* devSlot = nullptr;
        if (out.deviceBuffer != nullptr)
        {
            const std::size_t oneFrame = static_cast<std::size_t>(s.W) *
                                        static_cast<std::size_t>(s.H) * 3u;
            devSlot = out.deviceBuffer->mutableData() +
                      static_cast<std::size_t>(i) * oneFrame;
        }
        s.toBgrInto(dst, v.stride, devSlot, hostCopyNeeded);

        out.frameNames.push_back(s.stem + "_" + frameSuffix(s.nextFrameIndex));
        out.validCount = i + 1;
        ++s.nextFrameIndex;
    }

    if (out.validCount == 0)
    {
        return false;
    }

    // A short batch leaves its unused slots untouched. Pre-process reads every
    // slot of a static batch, so the tail has to be deterministic on whichever
    // side it lives. Only the last batch of a stream is short, so this costs one
    // memset per run -- and it is the only place that zeroes anything: zeroing
    // the whole block at allocation time used to cost a 12 MB memset per batch,
    // because a reused pool block arrives dirty.
    if (out.validCount < want && out.validCount > 0)
    {
        const std::size_t oneFrame = static_cast<std::size_t>(s.W) *
                                     static_cast<std::size_t>(s.H) * 3u;
        const std::size_t filled = oneFrame * static_cast<std::size_t>(out.validCount);

        if (out.buffer != nullptr)
        {
            const std::size_t tail = out.buffer->byteSize() - filled;
            if (tail > 0)
            {
                std::memset(out.buffer->mutableData() + filled, 0, tail);
            }
        }
        if (out.deviceBuffer != nullptr)
        {
            const std::size_t tail = out.deviceBuffer->byteSize() - filled;
            if (tail > 0)
            {
                const cudaError_t e = cudaMemsetAsync(
                    out.deviceBuffer->mutableData() + filled, 0, tail, nullptr);
                if (e != cudaSuccess)
                {
                    throw std::runtime_error(
                        std::string("FFmpegSource: device tail memset failed: ") +
                        cudaGetErrorString(e));
                }
            }
        }
    }

    // Everything the host copy needs has been queued by now; mark the point so a
    // reader can block on the actual copy instead of this thread doing it.
    if (out.hostReady)
    {
        out.hostReady->record();
    }

    if (!s.bannerLogged && s.W != 0)
    {
        s.bannerLogged = true;
        s.logBanner();
    }
    return true;
}

}  // namespace trt_alpha::datasource

#else  // ---------------------------------------------------------------- stub --

#include "trt_alpha/core/logger.hpp"

#include <stdexcept>

namespace trt_alpha::datasource {

struct FFmpegSource::Impl
{
    SourceConfig cfg;
};

FFmpegSource::FFmpegSource(const SourceConfig& cfg)
    : m_impl(std::make_unique<Impl>())
{
    m_impl->cfg = cfg;
    throw std::runtime_error(
        "FFmpegSource: this build was configured without FFmpeg support "
        "(TRT_ALPHA_WITH_FFMPEG=OFF); use --decode opencv, or rebuild with "
        "TRT_ALPHA_WITH_FFMPEG=ON and FFMPEG_ROOT pointing at a shared FFmpeg build");
}

FFmpegSource::~FFmpegSource() = default;

void FFmpegSource::requestStop() {}

const char* FFmpegSource::typeName() const noexcept { return "ffmpeg-disabled"; }

bool FFmpegSource::next(core::Batch&) { return false; }

}  // namespace trt_alpha::datasource

#endif
