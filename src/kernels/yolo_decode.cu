// =============================================================================
//  trt_alpha :: kernels :: yolo_decode（实现）
// -----------------------------------------------------------------------------
//  decode / nms / transpose 移植自 TensorRT-Alpha（实测正确）；
//  改动：参数化 CUDA 流、objects 行宽参数化（为 seg 扩展让路）、统一命名、
//        block size 走 common.cuh。
// =============================================================================
#include "trt_alpha/kernels/common.cuh"
#include "trt_alpha/kernels/yolo_decode.cuh"

#include <stdexcept>
#include <string>

namespace trt_alpha::kernels {
namespace {

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
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
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

    if (pitem_iou[0] < 0.f) { pitem_iou[0] = 0.f; }
    if (pitem_iou[0] > 1.f) { pitem_iou[0] = 1.f; }

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
    float locBuf[14];
    for (int i = 0; i < locRow; ++i) { locBuf[i] = pitem_loc[i]; }

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

    float* pitem_dst = dst + dy * dstArea + 1 + index * dstRow;
    pitem_dst[0] = locBuf[0];
    pitem_dst[1] = locBuf[1];
    pitem_dst[2] = locBuf[2];
    pitem_dst[3] = locBuf[3];
    pitem_dst[4] = score;
    pitem_dst[5] = 1.f;
    pitem_dst[6] = 1.f;
    pitem_dst[7]  = locBuf[4];
    pitem_dst[8]  = locBuf[5];
    pitem_dst[9]  = locBuf[6];
    pitem_dst[10] = locBuf[7];
    pitem_dst[11] = locBuf[8];
    pitem_dst[12] = locBuf[9];
    pitem_dst[13] = locBuf[10];
    pitem_dst[14] = locBuf[11];
    pitem_dst[15] = locBuf[12];
    pitem_dst[16] = locBuf[13];
}

__global__ void normPredKernel(int batchSize, float* __restrict__ data,
                               int area, float scale,
                               const float* __restrict__ minVals,
                               const float* __restrict__ maxVals)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= area || dy >= batchSize)
    {
        return;
    }
    const float v = data[dy * area + dx];
    data[dy * area + dx] = scale * (v - minVals[dy]) / (maxVals[dy] - minVals[dy]);
}

__global__ void transposeKernel(int batchSize, const float* __restrict__ src,
                                int srcRow, int anchors, float* __restrict__ dst)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
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
    (void)dstH;

    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= anchors || dy >= batchSize)
    {
        return;
    }
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

    float* out = dst + dy * dstArea + 1 + index * dstRow;
    out[0] = item[0];
    out[1] = item[1];
    out[2] = item[2];
    out[3] = item[3];
    out[4] = confidence;
    out[5] = static_cast<float>(label);
    out[6] = 1.f;
}

__global__ void decodeV5HeadKernel(int batchSize, int numClasses, int topK,
                                   float confThresh, const float* __restrict__ src,
                                   int srcRow, int anchors,
                                   float* __restrict__ dst, int dstRow)
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

    for (int i = 0; i < numMaskCoeffs; ++i)
    {
        out[7 + i] = item[4 + numClasses + i];
    }
}

__global__ void decodePoseHeadKernel(int batchSize, int topK,
                                     float confThresh, const float* __restrict__ src,
                                     int srcRow, int anchors,
                                     int numKpts,
                                     float* __restrict__ dst, int dstRow)
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
    const float conf = item[4];
    if (conf < confThresh)
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
    out[4] = conf;
    out[5] = 0.f;
    out[6] = 1.f;

    const float* kpt = item + 5;
    for (int i = 0; i < numKpts * 3; ++i)
    {
        out[7 + i] = kpt[i];
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
    // 行宽来自公共契约（postprocess.hpp），yunet.cpp 会拿引擎声明形状校验它。
    const int dstRow  = kYuNetObjectsRow;
    const int locRow  = kYuNetLocRow;
    const int confRow = kYuNetConfRow;
    const int iouRow  = kYuNetIouRow;

    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(numCandidates), batch);

    detail::decodeYuNetKernel<<<grid, block, 0, stream>>>(
        batch, numCandidates, topK, confThreshold,
        srcImgW, srcImgH,
        loc, locRow, conf, confRow, iou, iouRow,
        priorBoxes, variances, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYuNetHead launch");
}

void normPred(cudaStream_t stream, int batch,
              float* data, int width, int height,
              float scale,
              const float* minVals, const float* maxVals)
{
    const int area = width * height;
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(area), batch);
    detail::normPredKernel<<<grid, block, 0, stream>>>(
        batch, data, area, scale, minVals, maxVals);
    checkCuda(cudaGetLastError(), "normPred launch");
}

void decodeYoloNasHead(cudaStream_t stream, const YoloDecodeParams& p,
                       const float* src, int anchors, float* objects)
{
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(anchors), p.batch);
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
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(anchors), batch);
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
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(anchors), p.batch);
    const int dstRow = kObjectWidth + numMaskCoeffs;
    const int srcRow = 4 + p.numClasses + numMaskCoeffs;

    detail::decodeSegHeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, srcRow, anchors, numMaskCoeffs, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloV8SegHead launch");
}

void decodeYoloV8PoseHead(cudaStream_t stream, const YoloDecodeParams& p,
                          const float* src, int anchors,
                          int numKpts, float* objects)
{
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(anchors), p.batch);
    const int dstRow = kObjectWidth + numKpts * 3;
    const int srcRow = 4 + 1 + numKpts * 3;

    detail::decodePoseHeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.topK, p.confThreshold,
        src, srcRow, anchors, numKpts, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloV8PoseHead launch");
}

void decodeYoloV4Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors,
                      int dstW, int dstH,
                      float* objects)
{
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(anchors), p.batch);
    const int dstRow = kObjectWidth;

    detail::decodeV4HeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, anchors, dstW, dstH, objects, dstRow);
    checkCuda(cudaGetLastError(), "decodeYoloV4Head launch");
}

void decodeYoloV5Head(cudaStream_t stream, const YoloDecodeParams& p,
                      const float* src, int anchors, float* objects)
{
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(anchors), p.batch);
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
    const dim3 block = block2D();
    const dim3 grid = gridSize2D(static_cast<std::size_t>(p.topK), p.batch);
    detail::nmsFastKernel<<<grid, block, 0, stream>>>(
        p.topK, p.batch, p.iouThreshold, objects, objectWidth);
    checkCuda(cudaGetLastError(), "nmsFast launch");
}

}  // namespace trt_alpha::kernels