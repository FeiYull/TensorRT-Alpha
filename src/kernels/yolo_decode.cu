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

}  // namespace detail

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