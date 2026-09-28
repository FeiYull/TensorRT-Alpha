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

// __global__ void decodeV4HeadKernel(int batchSize, int numClasses, int topK,
//                                    float confThresh, const float* __restrict__ src,
//                                    int anchors, int dstW, int dstH,
//                                    float* __restrict__ dst, int dstRow)
// {
//     const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // anchor 序号
//     const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
//     if (dx >= anchors || dy >= batchSize)
//     {
//         return;
//     }
//     // src 布局：[B, anchors, 1, 4+nc] -> 每行 4+nc 个 float
//     const int srcRow = 4 + numClasses;
//     const int srcArea = anchors * srcRow;
//     const int dstArea = 1 + dstRow * topK;

//     const float* item = src + dy * srcArea + dx * srcRow;
//     const float* clsScore = item + 4;
//     float confidence = clsScore[0];
//     int label = 0;
//     for (int i = 1; i < numClasses; ++i)
//     {
//         if (clsScore[i] > confidence)
//         {
//             confidence = clsScore[i];
//             label = i;
//         }
//     }
//     if (confidence < confThresh)
//     {
//         return;
//     }

//     const int index = atomicAdd(dst + dy * dstArea, 1);
//     if (index >= topK)
//     {
//         return;
//     }

//     // YOLOv4: cx/cy/w/h 归一化（0~1），中心点 + 宽高 -> xyxy
//     const float cx = item[0] * static_cast<float>(dstW);
//     const float cy = item[1] * static_cast<float>(dstH);
//     const float w  = item[2] * static_cast<float>(dstW);
//     const float h  = item[3] * static_cast<float>(dstH);

//     float* out = dst + dy * dstArea + 1 + index * dstRow;
//     out[0] = cx - w * 0.5f;
//     out[1] = cy - h * 0.5f;
//     out[2] = cx + w * 0.5f;
//     out[3] = cy + h * 0.5f;
//     out[4] = confidence;
//     out[5] = static_cast<float>(label);
//     out[6] = 1.f;
// }


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

}  // namespace detail

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

    detail::decodeHeadKernel<<<grid, block, 0, stream>>>(
        p.batch, p.numClasses, p.topK, p.confThreshold,
        src, srcRow, anchors, objects, dstRow, numMaskCoeffs);
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