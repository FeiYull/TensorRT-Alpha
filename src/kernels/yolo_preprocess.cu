// =============================================================================
//  trt_alpha :: kernels :: yolo_preprocess（实现）
// -----------------------------------------------------------------------------
//  resize kernel 的双线性 + 越界回退逻辑移植自 TensorRT-Alpha（已实测正确）；
//  改动：参数化 CUDA 流、加边界检查、统一命名。
// =============================================================================
#include "trt_alpha/kernels/yolo_preprocess.cuh"

#include <cmath>
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



    
//! 2x3 仿射投影：输出坐标 -> 源图坐标。
__device__ inline void affineProject(const AffineMat& m, int x, int y,
                                     float* px, float* py)
{
    *px = m.v0 * x + m.v1 * y + m.v2;
    *py = m.v3 * x + m.v4 * y + m.v5;
}

__global__ void resizeLetterboxKernel(const std::uint8_t* __restrict__ src,
                                      int srcW, int srcH,
                                      float* __restrict__ dst, int dstW, int dstH,
                                      int batchSize, float padValue, AffineMat m)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // dst 像素序号
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    if (dx >= dstW * dstH || dy >= batchSize)
    {
        return;
    }
    const int dstY = dx / dstW;
    const int dstX = dx % dstW;

    float srcX = 0.f, srcY = 0.f;
    affineProject(m, dstX, dstY, &srcX, &srcY);

    float c0 = padValue, c1 = padValue, c2 = padValue;
    if (srcX >= -1.f && srcX < srcW && srcY >= -1.f && srcY < srcH)
    {
        const int yLow = static_cast<int>(floorf(srcY));
        const int xLow = static_cast<int>(floorf(srcX));
        const int yHigh = yLow + 1;
        const int xHigh = xLow + 1;
        const unsigned char pad[3] = {
            static_cast<unsigned char>(padValue),
            static_cast<unsigned char>(padValue),
            static_cast<unsigned char>(padValue)};
        const float ly = srcY - yLow;
        const float lx = srcX - xLow;
        const float w1 = (1.f - ly) * (1.f - lx);
        const float w2 = (1.f - ly) * lx;
        const float w3 = ly * (1.f - lx);
        const float w4 = ly * lx;

        const unsigned char* v1 = pad;
        const unsigned char* v2 = pad;
        const unsigned char* v3 = pad;
        const unsigned char* v4 = pad;
        const int srcVolume = 3 * srcH * srcW;
        if (yLow >= 0)
        {
            if (xLow >= 0)  { v1 = src + dy * srcVolume + (yLow * srcW + xLow) * 3; }
            if (xHigh < srcW){ v2 = src + dy * srcVolume + (yLow * srcW + xHigh) * 3; }
        }
        if (yHigh < srcH)
        {
            if (xLow >= 0)  { v3 = src + dy * srcVolume + (yHigh * srcW + xLow) * 3; }
            if (xHigh < srcW){ v4 = src + dy * srcVolume + (yHigh * srcW + xHigh) * 3; }
        }
        c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
        c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
        c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
    }

    float* out = dst + dy * (3 * dstH * dstW) + (dstY * dstW + dstX) * 3;
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
}

__global__ void resizeLetterboxF32Kernel(const float* __restrict__ src,
                                         int srcW, int srcH,
                                         float* __restrict__ dst, int dstW, int dstH,
                                         int batchSize, float padValue, AffineMat m)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx >= dstW * dstH || dy >= batchSize)
    {
        return;
    }
    const int dstY = dx / dstW;
    const int dstX = dx % dstW;

    float srcX = 0.f, srcY = 0.f;
    affineProject(m, dstX, dstY, &srcX, &srcY);

    float c0 = padValue, c1 = padValue, c2 = padValue;
    if (srcX >= -1.f && srcX < srcW && srcY >= -1.f && srcY < srcH)
    {
        const int yLow = static_cast<int>(floorf(srcY));
        const int xLow = static_cast<int>(floorf(srcX));
        const int yHigh = yLow + 1;
        const int xHigh = xLow + 1;
        const float pad[3] = { padValue, padValue, padValue };
        const float ly = srcY - yLow;
        const float lx = srcX - xLow;
        const float w1 = (1.f - ly) * (1.f - lx);
        const float w2 = (1.f - ly) * lx;
        const float w3 = ly * (1.f - lx);
        const float w4 = ly * lx;

        const float* v1 = pad;
        const float* v2 = pad;
        const float* v3 = pad;
        const float* v4 = pad;
        const int srcVolume = 3 * srcH * srcW;
        if (yLow >= 0)
        {
            if (xLow >= 0)  { v1 = src + dy * srcVolume + (yLow * srcW + xLow) * 3; }
            if (xHigh < srcW){ v2 = src + dy * srcVolume + (yLow * srcW + xHigh) * 3; }
        }
        if (yHigh < srcH)
        {
            if (xLow >= 0)  { v3 = src + dy * srcVolume + (yHigh * srcW + xLow) * 3; }
            if (xHigh < srcW){ v4 = src + dy * srcVolume + (yHigh * srcW + xHigh) * 3; }
        }
        c0 = w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0];
        c1 = w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1];
        c2 = w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2];
    }

    float* out = dst + dy * (3 * dstH * dstW) + (dstY * dstW + dstX) * 3;
    out[0] = c0;
    out[1] = c1;
    out[2] = c2;
}

__global__ void bgrToNchwNormKernel(const float* __restrict__ src,
                                    float* __restrict__ dst,
                                    int batchSize, int width, int height,
                                    float scale, float m0, float m1, float m2,
                                    float s0, float s1, float s2,
                                    int swapRB)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    const int volume = width * height * 3;
    if (dx >= volume || dy >= batchSize)
    {
        return;
    }
    const int spatial = dx / 3;
    const int chIn = dx % 3;
    // swapRB: BGR -> RGB；不 swap: 保持 BGR
    const int chOut = swapRB ? (2 - chIn) : chIn;
    const int y = spatial / width;
    const int x = spatial % width;

    const float mean = (chOut == 0) ? m0 : (chOut == 1) ? m1 : m2;
    const float stdv = (chOut == 0) ? s0 : (chOut == 1) ? s1 : s2;
    const float v = src[dy * volume + dx];

    dst[dy * volume + chOut * (width * height) + y * width + x] =
        (v / scale - mean) / stdv;
}

__global__ void copyWithPaddingKernel(int batchSize,
                                      const float* __restrict__ src,
                                      int srcWidth, int srcHeight,
                                      float* __restrict__ dst,
                                      int dstWidth, int dstHeight,
                                      float paddingValue,
                                      int padTop, int padLeft)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    const int dstArea = dstWidth * dstHeight;
    if (dx >= dstArea || dy >= batchSize)
    {
        return;
    }
    const int dstY = dx / dstWidth;
    const int dstX = dx % dstWidth;
    const int srcVolume = 3 * srcHeight * srcWidth;
    const int dstVolume = 3 * dstHeight * dstWidth;
    float* pdst = dst + dy * dstVolume + dstY * dstWidth * 3 + dstX * 3;

    if (dstY < (srcHeight + padTop) && dstY >= padTop &&
        dstX < (srcWidth + padLeft) && dstX >= padLeft)
    {
        const int sy = dstY - padTop;
        const int sx = dstX - padLeft;
        const float* psrc = src + dy * srcVolume + sy * srcWidth * 3 + sx * 3;
        pdst[0] = psrc[0];
        pdst[1] = psrc[1];
        pdst[2] = psrc[2];
    }
    else
    {
        pdst[0] = paddingValue;
        pdst[1] = paddingValue;
        pdst[2] = paddingValue;
    }
}

__global__ void bgrToRgbHwcKernel(float* __restrict__ data,
                                  int batchSize, int width, int height)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;   // 像素序号（spatial）
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;   // batch 序号
    const int spatial = width * height;
    if (dx >= spatial || dy >= batchSize)
    {
        return;
    }
    float* pixel = data + dy * spatial * 3 + dx * 3;
    // BGR <-> RGB: 交换第 0 和第 2 个通道
    const float b = pixel[0];
    const float r = pixel[2];
    pixel[0] = r;
    pixel[2] = b;
}

__global__ void hwcToChwKernel(const float* __restrict__ src,
                               float* __restrict__ dst,
                               int batchSize, int width, int height)
{
    const int dx = blockDim.x * blockIdx.x + threadIdx.x;
    const int dy = blockDim.y * blockIdx.y + threadIdx.y;
    const int imgArea = width * height;
    const int volume = imgArea * 3;
    if (dx >= volume || dy >= batchSize)
    {
        return;
    }
    const int ch = dx / imgArea;
    const int spatial = dx % imgArea;
    const int row = spatial / width;
    const int col = spatial % width;
    // dst: NCHW -> [dy][ch][row][col]
    // src: HWC  -> [dy][row][col][ch]
    const int srcIdx = dy * volume + (row * width + col) * 3 + ch;
    dst[dy * volume + dx] = src[srcIdx];
}

}  // namespace detail



void copyWithPadding(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, int padTop, int padLeft)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((dstW * dstH + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::copyWithPaddingKernel<<<grid, block, 0, stream>>>(
        batch, src, srcW, srcH, dst, dstW, dstH, padValue, padTop, padLeft);
    checkCuda(cudaGetLastError(), "copyWithPadding launch");
}

void bgrToRgbHwc(cudaStream_t stream, int batch,
                 float* data, int width, int height)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((width * height + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::bgrToRgbHwcKernel<<<grid, block, 0, stream>>>(data, batch, width, height);
    checkCuda(cudaGetLastError(), "bgrToRgbHwc launch");
}

void resizeLetterbox(cudaStream_t stream, int batch,
                     const std::uint8_t* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((dstW * dstH + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::resizeLetterboxKernel<<<grid, block, 0, stream>>>(
        src, srcW, srcH, dst, dstW, dstH, batch, padValue, dst2src);
    checkCuda(cudaGetLastError(), "resizeLetterbox launch");
}

void resizeLetterbox(cudaStream_t stream, int batch,
                     const float* src, int srcW, int srcH,
                     float* dst, int dstW, int dstH,
                     float padValue, AffineMat dst2src)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((dstW * dstH + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::resizeLetterboxF32Kernel<<<grid, block, 0, stream>>>(
        src, srcW, srcH, dst, dstW, dstH, batch, padValue, dst2src);
    checkCuda(cudaGetLastError(), "resizeLetterboxF32 launch");
}

void bgrToNchwNormalized(cudaStream_t stream, int batch,
                         const float* src, float* dst,
                         int width, int height,
                         float scale, const float mean[3], const float std_[3],
                         bool swapRB)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((width * height * 3 + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::bgrToNchwNormKernel<<<grid, block, 0, stream>>>(
        src, dst, batch, width, height, scale,
        mean[0], mean[1], mean[2], std_[0], std_[1], std_[2],
        swapRB ? 1 : 0);
    checkCuda(cudaGetLastError(), "bgrToNchwNormalized launch");
}

void hwcToChw(cudaStream_t stream, int batch,
              const float* src, float* dst,
              int width, int height)
{
    const dim3 block(kBlockSize, kBlockSize);
    const dim3 grid((width * height * 3 + kBlockSize - 1) / kBlockSize,
                    (batch + kBlockSize - 1) / kBlockSize);
    detail::hwcToChwKernel<<<grid, block, 0, stream>>>(src, dst, batch, width, height);
    checkCuda(cudaGetLastError(), "hwcToChw launch");
}

}  // namespace trt_alpha::kernels