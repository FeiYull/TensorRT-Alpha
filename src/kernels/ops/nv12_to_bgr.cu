// =============================================================================
//  trt_alpha :: kernels :: ops :: nv12_to_bgr（实现）
// -----------------------------------------------------------------------------
//  One thread per output pixel. The kernel is purely memory bound:
//      1080p -> 3.1 MB read (NV12) + 6.2 MB write (BGR) ~= 0.02 ms on a 5060 Ti
//  so there is no point in trying to be clever with shared memory: the UV plane
//  is read twice per 2x2 block, which the L2 absorbs.
//
//  Coefficient layout (standard YCbCr -> RGB):
//      R = yScale * (Y - yOffset) + cScale * 2 (1 - kr) * (V - 128)
//      B = yScale * (Y - yOffset) + cScale * 2 (1 - kb) * (U - 128)
//      G = yScale * (Y - yOffset) - cScale * (2 kb (1 - kb) / kg) * (U - 128)
//                                - cScale * (2 kr (1 - kr) / kg) * (V - 128)
//  with kg = 1 - kr - kb. Limited range folds into yScale = 255/219,
//  yOffset = 16 and cScale = 255/224; full range uses 1 / 0 / 1.
//
//  The chroma scale is easy to forget and the numbers still look plausible:
//  dropping it under-expands every colour by 224/255 = 12.2%, which on saturated
//  content means errors up to ~44 LSB (measured against libswscale).
// =============================================================================
#include "trt_alpha/kernels/ops/nv12_to_bgr.hpp"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

//! Clamp to [0,255] and round; __saturatef only clamps to [0,1], so it is not
//! usable here.
__device__ __forceinline__ unsigned char toU8(float v)
{
    return static_cast<unsigned char>(__float2uint_rn(fminf(fmaxf(v, 0.0f), 255.0f)));
}

struct YuvCoeff
{
    float yScale;
    float yOffset;
    float rv;   //!< 2 (1 - kr)
    float bu;   //!< 2 (1 - kb)
    float gu;   //!< 2 kb (1 - kb) / kg
    float gv;   //!< 2 kr (1 - kr) / kg
};

__host__ __device__ __forceinline__ YuvCoeff makeCoeff(YuvMatrix m, YuvRange r)
{
    float kr = 0.2126f;
    float kb = 0.0722f;
    if (m == YuvMatrix::BT601)
    {
        kr = 0.2990f;
        kb = 0.1140f;
    }
    else if (m == YuvMatrix::BT2020)
    {
        kr = 0.2627f;
        kb = 0.0593f;
    }

    const float kg = 1.0f - kr - kb;

    // Limited range squeezes chroma into [16, 240] as well, so the chroma
    // coefficients must be expanded by 255/224 on top of the 1/224 normalisation.
    float cScale = 1.0f;
    float yScale = 1.0f;
    float yOffset = 0.0f;
    if (r == YuvRange::Limited)
    {
        yScale  = 255.0f / 219.0f;   // == 1.16438
        yOffset = 16.0f;
        cScale  = 255.0f / 224.0f;   // == 1.13839
    }

    YuvCoeff c;
    c.yScale  = yScale;
    c.yOffset = yOffset;
    c.rv = cScale * 2.0f * (1.0f - kr);
    c.bu = cScale * 2.0f * (1.0f - kb);
    c.gu = cScale * 2.0f * kb * (1.0f - kb) / kg;
    c.gv = cScale * 2.0f * kr * (1.0f - kr) / kg;
    return c;
}

__global__ void nv12ToBgrKernel(const unsigned char* __restrict__ yPlane,
                                const unsigned char* __restrict__ uvPlane,
                                int yPitch,
                                int uvPitch,
                                unsigned char* __restrict__ bgr,
                                int bgrPitch,
                                int width,
                                int height,
                                YuvCoeff k)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x;
    const int y = blockIdx.y * blockDim.y + threadIdx.y;
    if (x >= width || y >= height)
    {
        return;
    }

    const float luma = (static_cast<float>(__ldg(&yPlane[(std::size_t)y * yPitch + x])) -
                        k.yOffset) * k.yScale;

    // 4:2:0 -> one chroma pair per 2x2 block; NV12 interleaves U then V.
    const int cx = x >> 1;
    const int cy = y >> 1;
    const unsigned char* uv =
        &uvPlane[static_cast<std::size_t>(cy) * uvPitch + static_cast<std::size_t>(cx) * 2u];
    const float cb = static_cast<float>(uv[0]) - 128.0f;
    const float cr = static_cast<float>(uv[1]) - 128.0f;

    const float r = luma + k.rv * cr;
    const float g = luma - k.gu * cb - k.gv * cr;
    const float b = luma + k.bu * cb;

    unsigned char* dst = &bgr[static_cast<std::size_t>(y) * bgrPitch +
                             static_cast<std::size_t>(x) * 3u];
    dst[0] = toU8(b);
    dst[1] = toU8(g);
    dst[2] = toU8(r);
}

}  // namespace

cudaError_t nv12ToBgr(cudaStream_t stream,
                      const std::uint8_t* y,
                      const std::uint8_t* uv,
                      int yPitch,
                      int uvPitch,
                      std::uint8_t* bgr,
                      int bgrPitch,
                      int width,
                      int height,
                      YuvMatrix matrix,
                      YuvRange range)
{
    if (y == nullptr || uv == nullptr || bgr == nullptr)
    {
        return cudaErrorInvalidValue;
    }
    if (width <= 0 || height <= 0 || (width & 1) != 0 || (height & 1) != 0)
    {
        return cudaErrorInvalidValue;
    }
    if (yPitch < width || uvPitch < width || bgrPitch < width * 3)
    {
        return cudaErrorInvalidValue;
    }

    const dim3 block(32u, 8u);
    const dim3 grid(static_cast<unsigned>((width + 31) / 32),
                    static_cast<unsigned>((height + 7) / 8));

    nv12ToBgrKernel<<<grid, block, 0, stream>>>(
        y, uv, yPitch, uvPitch, bgr, bgrPitch, width, height,
        makeCoeff(matrix, range));

    return cudaGetLastError();
}

}  // namespace trt_alpha::kernels::ops
