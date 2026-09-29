// =============================================================================
//  trt_alpha :: kernels :: yolo_decode（实现）
// -----------------------------------------------------------------------------
//  decode / nms / transpose 移植自 TensorRT-Alpha（实测正确）；
//  改动：参数化 CUDA 流、objects 行宽参数化（为 seg 扩展让路）、统一命名。
// =============================================================================
#include "trt_alpha/kernels/yolo_decode.cuh"

#include <stdexcept>
#include <string>

namespace trt_alpha::kernels {
namespace {

constexpr int kBlockSize = 8;

void checkCuda(cudaError_t err, const char* op)
{
    if (err != cudaSuccess)
    {
        throw std::runtime_error(std::string("kernels::") + op + " failed: " +
                                 cudaGetErrorString(err));
    }
}

}  // namespace

namespace detail {

__global__ void decodeYuNetKernel(
    int batchSize, int numCandidates, int topK, float confThreshold,
    int srcImgW, int srcImgH,
    const float* __restrict__ loc,  int locRow,
    const float* __restrict__ conf, int confRow,
    const float* __restrict__ iou,  int iouRow,
    const float* __restrict__ priorBoxes,
    const float* __restrict__ variances,
    float* __restrict__ dst, int dstRow)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // candidate 序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= numCandidates || dy >= batchSize)
    {
        return;
    }

    const int locArea  = numCandidates * locRow;
    const int confArea = numCandidates * confRow;
    const int iouArea  = numCandidates * iouRow;
    const int dstArea  = 1 + dstRow * topK;

    float* pitem_conf = const_cast<float*>(conf) + dy * confArea + dx * confRow;
    float* pitem_iou  = const_cast<float*>(iou)  + dy * iouArea  + dx * iouRow;

    // clamp iou 到 [0, 1]
    if (pitem_iou[0] < 0.f) { pitem_iou[0] = 0.f; }
    if (pitem_iou[0] > 1.f) { pitem_iou[0] = 1.f; }

    // softmax(conf) 第 2 类（人脸）概率
    const float e0 = expf(pitem_conf[0]);
    const float e1 = expf(pitem_conf[1]);
    const float exp_sum = e0 + e1;
    pitem_conf[1] = e1 / exp_sum;

    const float score = sqrtf(pitem_conf[1] * pitem_iou[0]);
    if (score <= confThreshold)
    {
        return;
    }

    const int index = atomicAdd(dst + dy * dstArea, 1);
    if (index >= topK)
    {
        return;
    }

    const float* pitem_loc = loc + dy * locArea + dx * locRow;
    // 直接改 pitem_loc —— 但它是 const float*，要转成可写
    // 为了不改输入，用临时变量
    float locBuf[14];
    for (int i = 0; i < locRow; ++i) { locBuf[i] = pitem_loc[i]; }

    // bbox 解码（严格照 legacy）
    const float pb0 = priorBoxes[4 * dx + 0];
    const float pb1 = priorBoxes[4 * dx + 1];
    const float pb2 = priorBoxes[4 * dx + 2];
    const float pb3 = priorBoxes[4 * dx + 3];
    const float v0 = variances[0];
    const float v1 = variances[1];

    locBuf[0] = pb0 + locBuf[0] * v0 * pb2;
    locBuf[1] = pb1 + locBuf[1] * v0 * pb3;
    locBuf[2] = pb2 * expf(locBuf[2] * v1);
    locBuf[3] = pb3 * expf(locBuf[3] * v1);

    locBuf[0] -= locBuf[2] / 2.f;
    locBuf[1] -= locBuf[3] / 2.f;
    locBuf[2] += locBuf[0];
    locBuf[3] += locBuf[1];

    locBuf[0] *= srcImgW;
    locBuf[1] *= srcImgH;
    locBuf[2] *= srcImgW;
    locBuf[3] *= srcImgH;

    // 5 个关键点：同样反归一化到原图
    locBuf[4]  = (pb0 + locBuf[4]  * v0 * pb2) * srcImgW;
    locBuf[6]  = (pb0 + locBuf[6]  * v0 * pb2) * srcImgW;
    locBuf[8]  = (pb0 + locBuf[8]  * v0 * pb2) * srcImgW;
    locBuf[10] = (pb0 + locBuf[10] * v0 * pb2) * srcImgW;
    locBuf[12] = (pb0 + locBuf[12] * v0 * pb2) * srcImgW;

    locBuf[5]  = (pb1 + locBuf[5]  * v0 * pb3) * srcImgH;
    locBuf[7]  = (pb1 + locBuf[7]  * v0 * pb3) * srcImgH;
    locBuf[9]  = (pb1 + locBuf[9]  * v0 * pb3) * srcImgH;
    locBuf[11] = (pb1 + locBuf[11] * v0 * pb3) * srcImgH;
    locBuf[13] = (pb1 + locBuf[13] * v0 * pb3) * srcImgH;

    // 写输出（17 个 float）：[left top right bottom conf label keep] + 5 对点
    float* pitem_dst = dst + dy * dstArea + 1 + index * dstRow;
    pitem_dst[0] = locBuf[0];   // left
    pitem_dst[1] = locBuf[1];   // top
    pitem_dst[2] = locBuf[2];   // right
    pitem_dst[3] = locBuf[3];   // bottom
    pitem_dst[4] = score;       // confidence
    pitem_dst[5] = 1.f;         // label（YuNet 单类，人脸）
    pitem_dst[6] = 1.f;         // keep
    pitem_dst[7]  = locBuf[4];  // 点 1 x
    pitem_dst[8]  = locBuf[5];  // 点 1 y
    pitem_dst[9]  = locBuf[6];
    pitem_dst[10] = locBuf[7];
    pitem_dst[11] = locBuf[8];
    pitem_dst[12] = locBuf[9];
    pitem_dst[13] = locBuf[10];
    pitem_dst[14] = locBuf[11];
    pitem_dst[15] = locBuf[12];
    pitem_dst[16] = locBuf[13];
}

__global__ void transposeKernel(int batchSize, const float* __restrict__ src,
                                int srcRow, int anchors, float* __restrict__ dst)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // anchor 序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
    const int srcArea = anchors * srcRow;
    const float* srcCol = src + dy * srcArea + dx;
    float* dstRow = dst + dy * srcArea + dx * srcRow;
    for (int i = 0; i < srcRow; ++i)
    {
        dstRow[i] = srcCol[i * anchors];
    }
}

__global__ void decodeHeadKernel(int batchSize, int numClasses, int topK,
                                 float confThresh, const float* __restrict__ src,
                                 int srcRow, int anchors,
                                 float* __restrict__ dst, int dstRow,
                                 int numMaskCoeffs)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
    const int srcArea = anchors * srcRow;
    const int dstArea = 1 + dstRow * topK;

    const float* item = src + dy * srcArea + dx * srcRow;
    const float* clsScore = item + 4;
    float confidence = clsScore[0];
    int label = 0;
    for (int i = 1; i < numClasses; ++i)
    {
        if (clsScore[i] > confidence)
        {
            confidence = clsScore[i];
            label = i;
        }
    }
    if (confidence < confThresh)
    {
        return;
    }

    const int index = atomicAdd(dst + dy * dstArea, 1);
    if (index >= topK)
    {
        return;
    }

    const float cx = item[0];
    const float cy = item[1];
    const float w = item[2];
    const float h = item[3];
    float* out = dst + dy * dstArea + 1 + index * dstRow;
    out[0] = cx - w * 0.5f;
    out[1] = cy - h * 0.5f;
    out[2] = cx + w * 0.5f;
    out[3] = cy + h * 0.5f;
    out[4] = confidence;
    out[5] = static_cast<float>(label);
    out[6] = 1.f;
    for (int i = 0; i < numMaskCoeffs; ++i)
    {
        out[7 + i] = item[4 + numClasses + i];
    }
}

__device__ inline float boxIou(float al, float at, float ar, float ab,
                               float bl, float bt, float br, float bb)
{
    const float cl = max(al, bl);
    const float ct = max(at, bt);
    const float cr = min(ar, br);
    const float cb = min(ab, bb);
    const float cArea = max(cr - cl, 0.0f) * max(cb - ct, 0.0f);
    if (cArea == 0.0f)
    {
        return 0.0f;
    }
    const float aArea = max(0.0f, ar - al) * max(0.0f, ab - at);
    const float bArea = max(0.0f, br - bl) * max(0.0f, bb - bt);
    return cArea / (aArea + bArea - cArea);
}

__global__ void nmsFastKernel(int topK, int batchSize, float iouThresh,
                              float* __restrict__ src, int srcRow)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dy >= batchSize)
    {
        return;
    }
    const int srcArea = 1 + srcRow * topK;
    const int count = min(static_cast<int>(src[dy * srcArea]), topK);
    if (dx >= count)
    {
        return;
    }
    float* current = src + dy * srcArea + 1 + dx * srcRow;
    for (int i = 0; i < count; ++i)
    {
        const float* item = src + dy * srcArea + 1 + i * srcRow;
        if (i == dx || item[5] != current[5])
        {
            continue;
        }
        if (item[4] >= current[4])
        {
            if (item[4] == current[4] && i < dx)
            {
                continue;
            }
            if (boxIou(current[0], current[1], current[2], current[3],
                       item[0], item[1], item[2], item[3]) > iouThresh)
            {
                current[6] = 0.f;
                return;
            }
        }
    }
}



__global__ void decodeV4HeadKernel(int batchSize, int numClasses, int topK,
                                   float confThresh, const float* __restrict__ src,
                                   int anchors, int dstW, int dstH,
                                   float* __restrict__ dst, int dstRow)
{
    (void)dstW;
    (void)dstH;   // 不用：归一化坐标原样输出，转像素在 postprocess 做

    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // anchor 序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
    // src 布局：[B, anchors, 1, 4+nc] -> 每行 4+nc 个 float
    const int srcRow = 4 + numClasses;
    const int srcArea = anchors * srcRow;
    const int dstArea = 1 + dstRow * topK;

    const float* item = src + dy * srcArea + dx * srcRow;
    const float* clsScore = item + 4;
    float confidence = clsScore[0];
    int label = 0;
    for (int i = 1; i < numClasses; ++i)
    {
        if (clsScore[i] > confidence)
        {
            confidence = clsScore[i];
            label = i;
        }
    }
    if (confidence < confThresh)
    {
        return;
    }

    const int index = atomicAdd(dst + dy * dstArea, 1);
    if (index >= topK)
    {
        return;
    }

    // YOLOv4: ONNX 输出的 4 个值 = (left, top, right, bottom)，归一化 0~1。
    // 原样存进 objects（不转 xyxy、不乘 dstW/dstH），
    // 像素转换 + 仿射逆变换在 postprocess 做（和 legacy 一致）。
    float* out = dst + dy * dstArea + 1 + index * dstRow;
    out[0] = item[0];   // left
    out[1] = item[1];   // top
    out[2] = item[2];   // right
    out[3] = item[3];   // bottom
    out[4] = confidence;
    out[5] = static_cast<float>(label);
    out[6] = 1.f;
}


__global__ void decodeV5HeadKernel(int batchSize, int numClasses, int topK,
                                   float confThresh, const float* __restrict__ src,
                                   int srcRow, int anchors,
                                   float* __restrict__ dst, int dstRow)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // anchor 序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
    const int srcArea = anchors * srcRow;
    const int dstArea = 1 + dstRow * topK;

    const float* item = src + dy * srcArea + dx * srcRow;
    const float objectness = item[4];
    if (objectness < confThresh)
    {
        return;
    }
    const float* clsScore = item + 5;
    float confidence = clsScore[0];
    int label = 0;
    for (int i = 1; i < numClasses; ++i)
    {
        if (clsScore[i] > confidence)
        {
            confidence = clsScore[i];
            label = i;
        }
    }
    confidence *= objectness;
    if (confidence < confThresh)
    {
        return;
    }

    const int index = atomicAdd(dst + dy * dstArea, 1);
    if (index >= topK)
    {
        return;
    }

    const float cx = item[0];
    const float cy = item[1];
    const float w = item[2];
    const float h = item[3];
    float* out = dst + dy * dstArea + 1 + index * dstRow;
    out[0] = cx - w * 0.5f;
    out[1] = cy - h * 0.5f;
    out[2] = cx + w * 0.5f;
    out[3] = cy + h * 0.5f;
    out[4] = confidence;
    out[5] = static_cast<float>(label);
    out[6] = 1.f;
}

__global__ void decodeNasHeadKernel(int batchSize, int numClasses, int topK,
                                    float confThresh, const float* __restrict__ src,
                                    int srcRow, int anchors,
                                    float* __restrict__ dst, int dstRow)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // anchor 序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
    const int srcArea = anchors * srcRow;
    const int dstArea = 1 + dstRow * topK;

    const float* item = src + dy * srcArea + dx * srcRow;
    const float* clsScore = item + 4;
    float confidence = clsScore[0];
    int label = 0;
    for (int i = 1; i < numClasses; ++i)
    {
        if (clsScore[i] > confidence)
        {
            confidence = clsScore[i];
            label = i;
        }
    }
    if (confidence < confThresh)
    {
        return;
    }

    const int index = atomicAdd(dst + dy * dstArea, 1);
    if (index >= topK)
    {
        return;
    }

    // YOLO-NAS: item[0..3] 直接是 left/top/right/bottom
    float* out = dst + dy * dstArea + 1 + index * dstRow;
    out[0] = item[0];
    out[1] = item[1];
    out[2] = item[2];
    out[3] = item[3];
    out[4] = confidence;
    out[5] = static_cast<float>(label);
    out[6] = 1.f;
}

__global__ void decodeSegHeadKernel(int batchSize, int numClasses, int topK,
                                    float confThresh, const float* __restrict__ src,
                                    int srcRow, int anchors,
                                    int numMaskCoeffs,
                                    float* __restrict__ dst, int dstRow)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // anchor 序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
    const int srcArea = anchors * srcRow;
    const int dstArea = 1 + dstRow * topK;

    const float* item = src + dy * srcArea + dx * srcRow;
    const float* clsScore = item + 4;
    float confidence = clsScore[0];
    int label = 0;
    for (int i = 1; i < numClasses; ++i)
    {
        if (clsScore[i] > confidence)
        {
            confidence = clsScore[i];
            label = i;
        }
    }
    if (confidence < confThresh)
    {
        return;
    }

    const int index = atomicAdd(dst + dy * dstArea, 1);
    if (index >= topK)
    {
        return;
    }

    const float cx = item[0];
    const float cy = item[1];
    const float w  = item[2];
    const float h  = item[3];
    float* out = dst + dy * dstArea + 1 + index * dstRow;
    out[0] = cx - w * 0.5f;
    out[1] = cy - h * 0.5f;
    out[2] = cx + w * 0.5f;
    out[3] = cy + h * 0.5f;
    out[4] = confidence;
    out[5] = static_cast<float>(label);
    out[6] = 1.f;

    // 后 numMaskCoeffs 个：mask 系数
    for (int i = 0; i < numMaskCoeffs; ++i)
    {
        out[7 + i] = item[4 + numClasses + i];
    }
}

}  // namespace detail

void decodeYuNetHead(cudaStream_t stream,
                     const float* loc, const float* conf, const float* iou,
                     int batch, int numCandidates,
                     int srcImgW, int srcImgH,
                     float confThreshold, int topK,
                     const float* priorBoxes,
                     const float* variances,
                     float* objects)
{
    const int dstRow = 17;   // 7 + 10（5 个关键点）
    const int locRow  = 14;
    const int confRow = 2;
    const int iouRow  = 1;

    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((numCandidates + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);

    detail::decodeYuNetKernel<<<grid, block, 0, stream>>>(
        batch, numCandidates, topK, confThreshold,
        srcImgW, srcImgH,
        loc, locRow, conf, confRow, iou, iouRow,
        priorBoxes, variances, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYuNetHead launch");
}

void decodeYoloNasHead(cudaStream_t stream, const YoloDecodeParams& p,
                       const float* src, int anchors, float* objects)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((anchors + kBlockSize - 1) / kBlockSize,
                    (p.batch + kBlockSize - 1) / kBlockSize);
    const int srcRow = 4 + p.numClasses;
    const int dstRow = kObjectWidth;

    detail::decodeNasHeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, srcRow, anchors, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloNasHead launch");
}

void transposeAnchors(cudaStream_t stream, int batch,
                      const float* src, int srcRow, int anchors, float* dst)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((anchors + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::transposeKernel<<<grid, block, 0, stream>>>(batch, src, srcRow, anchors, dst);
    checkCuda(cudaGetLastError(), "transposeAnchors launch");
}

void decodeYoloV8Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects)
{
    decodeYoloV8SegHead(stream, p, src, anchors, 0, objects);
}

void decodeYoloV8SegHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numMaskCoeffs, float* objects)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((anchors + kBlockSize - 1) / kBlockSize,
                    (p.batch + kBlockSize - 1) / kBlockSize);
    const int dstRow = kObjectWidth + numMaskCoeffs;
    const int srcRow = 4 + p.numClasses + numMaskCoeffs;

    detail::decodeSegHeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, srcRow, anchors, numMaskCoeffs, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloV8SegHead launch");
}

void decodeYoloV4Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors,
                      int dstW, int dstH,
                      float* objects)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((anchors + kBlockSize - 1) / kBlockSize,
                    (p.batch + kBlockSize - 1) / kBlockSize);
    const int dstRow = kObjectWidth;

    detail::decodeV4HeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, anchors, dstW, dstH, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloV4Head launch");
}

void decodeYoloV5Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((anchors + kBlockSize - 1) / kBlockSize,
                    (p.batch + kBlockSize - 1) / kBlockSize);
    const int srcRow = 5 + p.numClasses;
    const int dstRow = kObjectWidth;

    detail::decodeV5HeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, srcRow, anchors, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloV5Head launch");
}

void nmsFast(cudaStream_t stream, const YoloDecodeParams& p,
             float* objects, int objectWidth)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((p.topK + kBlockSize - 1) / kBlockSize,
                    (p.batch + kBlockSize - 1) / kBlockSize);
    detail::nmsFastKernel<<<grid, block, 0, stream>>>(
        p.topK, p.batch, p.iouThreshold, objects, objectWidth);
    checkCuda(cudaGetLastError(), "nmsFast launch");
}

}  // namespace trt_alpha::kernels