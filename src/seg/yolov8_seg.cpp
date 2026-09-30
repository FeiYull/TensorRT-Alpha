// =============================================================================
//  trt_alpha :: det :: YoloV8Seg（实现）
// =============================================================================
#include "yolov8_seg.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <opencv2/imgproc.hpp>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <string>
#include <vector>

namespace trt_alpha::seg {
namespace {

void buildLetterboxAffine(int srcW, int srcH, int dstW, int dstH,
                          kernels::AffineMat& dst2src)
{
    const float a = static_cast<float>(dstH) / static_cast<float>(srcH);
    const float b = static_cast<float>(dstW) / static_cast<float>(srcW);
    const float scale = std::min(a, b);
    const float tx = (-scale * srcW + dstW + scale - 1.f) * 0.5f;
    const float ty = (-scale * srcH + dstH + scale - 1.f) * 0.5f;
    const float invScale = 1.f / scale;
    dst2src.v0 = invScale;  dst2src.v1 = 0.f;  dst2src.v2 = -tx * invScale;
    dst2src.v3 = 0.f;       dst2src.v4 = invScale; dst2src.v5 = -ty * invScale;
}

}  // namespace

const std::string& YoloV8Seg::name() const noexcept
{
    static const std::string kName = "yolov8-seg";
    return kName;
}

void YoloV8Seg::loadConfig(const core::ModelConfig& cfg)
{
    m_cfg = cfg;
    m_numClass = cfg.getInt("num_class", 80);
    if (m_numClass <= 0)
    {
        throw std::runtime_error("yolov8-seg: num_class must be > 0");
    }
    m_numMaskCoeffs = cfg.getInt("num_mask_coeffs", 32);
    m_maskProtoH = cfg.getInt("mask_proto_h", 160);
    m_maskProtoW = cfg.getInt("mask_proto_w", 160);
    m_confThreshold = cfg.getFloat("conf_thresh", 0.25f);
    m_iouThreshold = cfg.getFloat("iou_thresh", 0.7f);
    m_topK = cfg.getInt("top_k", 300);
    m_normScale = cfg.getFloat("norm_scale", 255.f);
    m_padValue = cfg.getFloat("pad_value", 114.f);

    TRT_LOG_INFO("YoloV8Seg: config num_class=" << m_numClass
                 << " mask_coeffs=" << m_numMaskCoeffs
                 << " proto=" << m_maskProtoW << "x" << m_maskProtoH
                 << " conf=" << m_confThreshold
                 << " iou=" << m_iouThreshold
                 << " top_k=" << m_topK);
}

void YoloV8Seg::discoverEngineIo()
{
    for (const auto& t : m_engine->ioTensors())
    {
        if (t.isInput)                     { m_inputName = t.name;   continue; }
        if (t.name == "output0")           { m_output0Name = t.name; continue; }
        if (t.name == "output1")           { m_output1Name = t.name; continue; }
    }
    if (m_inputName.empty() || m_output0Name.empty() || m_output1Name.empty())
    {
        throw std::runtime_error("yolov8-seg: missing expected I/O tensors");
    }

    m_engine->setInputShape(m_inputName, nvinfer1::Dims4(m_cfg.batchSize, 3,
                                                          m_cfg.dstH, m_cfg.dstW));

    // output0: [B, 116, 8400]
    const nvinfer1::Dims out0 = m_engine->contextShape(m_output0Name);
    if (out0.nbDims != 3)
    {
        throw std::runtime_error("yolov8-seg: expect output0 as [B, 4+nc+32, anchors]");
    }
    m_srcRow  = static_cast<int>(out0.d[1]);
    m_anchors = static_cast<int>(out0.d[2]);
    if (m_srcRow != 4 + m_numClass + m_numMaskCoeffs)
    {
        throw std::runtime_error(
            "yolov8-seg: output0 channel = " + std::to_string(m_srcRow) +
            " but 4+num_class+mask_coeffs = " +
            std::to_string(4 + m_numClass + m_numMaskCoeffs));
    }

    // output1: [B, 32, 160, 160]
    const nvinfer1::Dims out1 = m_engine->contextShape(m_output1Name);
    if (out1.nbDims != 4)
    {
        throw std::runtime_error("yolov8-seg: expect output1 as [B, 32, H, W]");
    }
    const int protoH = static_cast<int>(out1.d[2]);
    const int protoW = static_cast<int>(out1.d[3]);
    const int protoC = static_cast<int>(out1.d[1]);
    if (protoC != m_numMaskCoeffs || protoH != m_maskProtoH || protoW != m_maskProtoW)
    {
        TRT_LOG_WARN("yolov8-seg: output1 shape [" << protoC << ", " << protoH
                     << ", " << protoW << "] != config ["
                     << m_numMaskCoeffs << ", " << m_maskProtoH << ", "
                     << m_maskProtoW << "], using engine shape");
        m_numMaskCoeffs = protoC;
        m_maskProtoH = protoH;
        m_maskProtoW = protoW;
    }
}

void YoloV8Seg::allocateBuffers()
{
    const int B = m_cfg.batchSize;
    const int H = m_cfg.dstH;
    const int W = m_cfg.dstW;
    const std::size_t dstArea = std::size_t(H) * W;
    const std::size_t oneImageF32 = 3 * dstArea * sizeof(float);

    auto logBox = [&](const char* nm, int batch, int ch, int h, int w,
                      core::DataType dt, std::size_t bytes, core::MemorySpace sp)
    {
        core::detail::AllocInfo info;
        info.name = nm; info.batch = batch; info.channels = ch;
        info.height = h; info.width = w; info.dtype = dt;
        info.bytes = bytes; info.space = sp;
        core::detail::logAllocBox(info);
    };

    m_inputSrc.allocate(std::size_t(B) * 3 * W * H);
    logBox("yolov8_seg.input_src", B, 3, H, W, core::DataType::UInt8,
           m_inputSrc.bytes(), core::MemorySpace::Device);
    m_resizeOut.allocate(std::size_t(B) * oneImageF32);
    logBox("yolov8_seg.resize_out", B, 3, H, W, core::DataType::Float32,
           m_resizeOut.bytes(), core::MemorySpace::Device);
    m_inputNchw.allocate(std::size_t(B) * oneImageF32);
    logBox("yolov8_seg.input_nchw", B, 3, H, W, core::DataType::Float32,
           m_inputNchw.bytes(), core::MemorySpace::Device);

    m_outputSrc.allocate(std::size_t(B) * m_srcRow * m_anchors * sizeof(float));
    logBox("yolov8_seg.output0", B, m_srcRow, 1, m_anchors,
           core::DataType::Float32, m_outputSrc.bytes(), core::MemorySpace::Device);

    m_outputTransposed.allocate(std::size_t(B) * m_srcRow * m_anchors * sizeof(float));
    logBox("yolov8_seg.output0_transposed", B, m_anchors, 1, m_srcRow,
           core::DataType::Float32, m_outputTransposed.bytes(),
           core::MemorySpace::Device);

    m_outputSeg.allocate(std::size_t(B) * m_numMaskCoeffs * m_maskProtoH * m_maskProtoW * sizeof(float));
    logBox("yolov8_seg.output1", B, m_numMaskCoeffs, m_maskProtoH, m_maskProtoW,
           core::DataType::Float32, m_outputSeg.bytes(), core::MemorySpace::Device);

    m_objectsRow = 7 + m_numMaskCoeffs;
    m_objectsPerImage = 1 + m_objectsRow * m_topK;
    m_objects.allocate(std::size_t(B) * m_objectsPerImage * sizeof(float));
    logBox("yolov8_seg.objects", B, m_objectsPerImage, 1, 1,
           core::DataType::Float32, m_objects.bytes(), core::MemorySpace::Device);
    m_objectsHost.allocate(std::size_t(B) * m_objectsPerImage * sizeof(float));

    m_outputSegHost.allocate(std::size_t(B) * m_numMaskCoeffs * m_maskProtoH * m_maskProtoW * sizeof(float));

    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->setTensorAddress(m_inputName.c_str(), m_inputNchw.data()) ||
        !ctx->setTensorAddress(m_output0Name.c_str(), m_outputSrc.data()) ||
        !ctx->setTensorAddress(m_output1Name.c_str(), m_outputSeg.data()))
    {
        throw std::runtime_error("yolov8-seg: setTensorAddress failed");
    }
}

void YoloV8Seg::init(const core::ModelConfig& cfg)
{
    loadConfig(cfg);
    m_engine = std::make_unique<core::TrtEngine>(cfg.engine);
    discoverEngineIo();
    allocateBuffers();
    m_segmentations.assign(static_cast<std::size_t>(m_cfg.batchSize), {});
    TRT_LOG_INFO("YoloV8Seg: initialized (batch=" << m_cfg.batchSize
                 << ", anchors=" << m_anchors
                 << ", srcRow=" << m_srcRow << ")");
}

void YoloV8Seg::setBatch(const core::Batch& batch)
{
    if (batch.views.empty())
    {
        throw std::runtime_error("yolov8-seg: empty batch");
    }
    m_batch = static_cast<int>(batch.views.size());
    m_srcH = batch.views[0].height;
    m_srcW = batch.views[0].width;

    buildLetterboxAffine(m_srcW, m_srcH, m_cfg.dstW, m_cfg.dstH, m_dst2src);

    const std::size_t oneImage = 3 * std::size_t(m_srcH) * m_srcW;
    const std::size_t total = oneImage * batch.views.size();
    if (batch.buffer == nullptr || batch.buffer->data() == nullptr)
    {
        throw std::runtime_error("yolov8-seg: batch.buffer is null");
    }
    if (total > m_inputSrc.bytes())
    {
        m_inputSrc.allocate(total);
    }
    cudaMemcpyAsync(m_inputSrc.data(), batch.buffer->data(), total,
                    cudaMemcpyHostToDevice, m_stream.get());
    m_stream.synchronize();
}

void YoloV8Seg::preprocess()
{
    kernels::resizeLetterbox(m_stream.get(), m_batch,
                             static_cast<const std::uint8_t*>(m_inputSrc.data()),
                             m_srcW, m_srcH,
                             m_resizeOut.asFloat(),
                             m_cfg.dstW, m_cfg.dstH,
                             m_padValue, m_dst2src);

    kernels::bgrToNchwNormalized(m_stream.get(), m_batch,
                                 m_resizeOut.asFloat(),
                                 m_inputNchw.asFloat(),
                                 m_cfg.dstW, m_cfg.dstH,
                                 m_normScale, m_normMean, m_normStd,
                                 /*swapRB=*/true);
}

void YoloV8Seg::infer()
{
    nvinfer1::IExecutionContext* ctx = m_engine->context();
    if (!ctx->enqueueV3(m_stream.get()))
    {
        throw std::runtime_error("yolov8-seg: enqueueV3 failed");
    }
}

void YoloV8Seg::postprocess()
{
    m_segmentations.assign(static_cast<std::size_t>(m_batch), {});

    kernels::YoloDecodeParams p;
    p.batch = m_batch;
    p.numClasses = m_numClass;
    p.topK = m_topK;
    p.confThreshold = m_confThreshold;
    p.iouThreshold = m_iouThreshold;

    cudaMemsetAsync(m_objects.data(), 0,
                    std::size_t(m_objectsPerImage) * m_cfg.batchSize * sizeof(float),
                    m_stream.get());

    // 1. transpose output0: [B, 116, 8400] -> [B, 8400, 116]
    kernels::transposeAnchors(m_stream.get(), m_batch,
                              m_outputSrc.asFloat(), m_srcRow, m_anchors,
                              m_outputTransposed.asFloat());

    // 2. decode（带 mask 系数）
    kernels::decodeYoloV8SegHead(m_stream.get(), p,
                                 m_outputTransposed.asFloat(), m_anchors,
                                 m_numMaskCoeffs, m_objects.asFloat());

    // 3. NMS（只读前 7 个字段，mask 系数不参与）
    kernels::nmsFast(m_stream.get(), p, m_objects.asFloat(), m_objectsRow);

    // 4. D2H
    cudaMemcpyAsync(m_objectsHost.data(), m_objects.data(),
                    std::size_t(m_objectsPerImage) * m_batch * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    cudaMemcpyAsync(m_outputSegHost.data(), m_outputSeg.data(),
                    std::size_t(m_numMaskCoeffs) * m_maskProtoH * m_maskProtoW
                        * m_batch * sizeof(float),
                    cudaMemcpyDeviceToHost, m_stream.get());
    m_stream.synchronize();

    // 5. CPU 后处理 mask
    const float* objHost = m_objectsHost.asFloat();
    const float* protoHost = m_outputSegHost.asFloat();

    for (int b = 0; b < m_batch; ++b)
    {
        const float* objRow = objHost + std::size_t(b) * m_objectsPerImage;
        const int count = std::clamp(static_cast<int>(objRow[0]), 0, m_topK);
        const float* protoB = protoHost + std::size_t(b) * m_numMaskCoeffs
                              * m_maskProtoH * m_maskProtoW;

        for (int i = 0; i < count; ++i)
        {
            const float* o = objRow + 1 + i * m_objectsRow;
            if (o[6] < 0.5f) { continue; }

            // 框（网络输入像素坐标）-> 原图像素坐标
            const float x0 = m_dst2src.v0 * o[0] + m_dst2src.v1 * o[1] + m_dst2src.v2;
            const float y0 = m_dst2src.v3 * o[0] + m_dst2src.v4 * o[1] + m_dst2src.v5;
            const float x1 = m_dst2src.v0 * o[2] + m_dst2src.v1 * o[3] + m_dst2src.v2;
            const float y1 = m_dst2src.v3 * o[2] + m_dst2src.v4 * o[3] + m_dst2src.v5;

            // mask 系数
            const float* coeff = o + 7;

            // 5.1 mask = sigmoid(sum_k(coeff[k] * proto[k]))
            // proto 布局: [32, 160, 160]，连续
            cv::Mat mask160(m_maskProtoH, m_maskProtoW, CV_32F);
            for (int k = 0; k < m_numMaskCoeffs; ++k)
            {
                const float* protoK = protoB + std::size_t(k) * m_maskProtoH * m_maskProtoW;
                if (k == 0)
                {
                    for (int y = 0; y < m_maskProtoH; ++y)
                    {
                        const float* srcRow = protoK + std::size_t(y) * m_maskProtoW;
                        float* dstRow = mask160.ptr<float>(y);
                        for (int x = 0; x < m_maskProtoW; ++x)
                        {
                            dstRow[x] = coeff[k] * srcRow[x];
                        }
                    }
                }
                else
                {
                    for (int y = 0; y < m_maskProtoH; ++y)
                    {
                        const float* srcRow = protoK + std::size_t(y) * m_maskProtoW;
                        float* dstRow = mask160.ptr<float>(y);
                        for (int x = 0; x < m_maskProtoW; ++x)
                        {
                            dstRow[x] += coeff[k] * srcRow[x];
                        }
                    }
                }
            }
            // sigmoid
            cv::exp(-mask160, mask160);
            mask160 = 1.f / (1.f + mask160);

            // 5.2 mask 裁剪到 bbox（在 160×160 空间）
            const float scale160 = static_cast<float>(m_maskProtoW) / m_cfg.dstW;
            const int mx0 = std::clamp(static_cast<int>(std::lround(o[0] * scale160)), 0, m_maskProtoW);
            const int my0 = std::clamp(static_cast<int>(std::lround(o[1] * scale160)), 0, m_maskProtoH);
            const int mx1 = std::clamp(static_cast<int>(std::lround(o[2] * scale160)), 0, m_maskProtoW);
            const int my1 = std::clamp(static_cast<int>(std::lround(o[3] * scale160)), 0, m_maskProtoH);
            const int mw = mx1 - mx0;
            const int mh = my1 - my0;
            if (mw <= 0 || mh <= 0) { continue; }

            // 5.3 原图 bbox（裁剪到图像范围）
            const int bx0 = std::clamp(static_cast<int>(std::lround(x0)), 0, m_srcW);
            const int by0 = std::clamp(static_cast<int>(std::lround(y0)), 0, m_srcH);
            const int bx1 = std::clamp(static_cast<int>(std::lround(x1)), 0, m_srcW);
            const int by1 = std::clamp(static_cast<int>(std::lround(y1)), 0, m_srcH);
            const int bw = bx1 - bx0;
            const int bh = by1 - by0;
            if (bw <= 0 || bh <= 0) { continue; }

            // 5.4 resize mask 到 bbox 大小 + 二值化
            cv::Mat maskRoi = mask160(cv::Rect(mx0, my0, mw, mh));
            cv::Mat maskResized;
            cv::resize(maskRoi, maskResized, cv::Size(bw, bh), 0, 0, cv::INTER_LINEAR);
            cv::Mat maskBin = maskResized > 0.5f;

            // 5.5 创建 Buffer（uint8，CV_8UC1），填 0/255
            auto maskBuf = core::Buffer::createHost(bw, bh, 1, core::DataType::UInt8);
            std::uint8_t* maskData = maskBuf->mutableData();
            for (int y = 0; y < bh; ++y)
            {
                const uchar* src = maskBin.ptr<uchar>(y);
                std::uint8_t* dst = maskData + std::size_t(y) * bw;
                for (int x = 0; x < bw; ++x)
                {
                    dst[x] = (src[x] != 0) ? 255 : 0;
                }
            }

            // 5.6 组装 Segmentation
            seg::Segmentation s;
            s.box.left = x0;
            s.box.top = y0;
            s.box.right = x1;
            s.box.bottom = y1;
            s.box.confidence = o[4];
            s.box.label = static_cast<int>(o[5]);
            s.mask = maskBuf->view();
            s.maskOwner = maskBuf;
            m_segmentations[std::size_t(b)].push_back(std::move(s));
        }
    }
}

void YoloV8Seg::reset()
{
    for (auto& v : m_segmentations) { v.clear(); }
    m_batch = 0;
}

}  // namespace trt_alpha::det

TRT_ALPHA_REGISTER_MODEL("yolov8-seg", trt_alpha::seg::YoloV8Seg);