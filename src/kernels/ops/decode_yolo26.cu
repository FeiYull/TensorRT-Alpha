// =============================================================================
//  trt_alpha :: kernels :: ops :: decode_yolo26
// -----------------------------------------------------------------------------
//  Implementation described at the top of decode_yolo26.hpp.
//
//  One thread per anchor, batch on blockIdx.y. Everything is a single IEEE-754
//  float32 add / sub / divide-by-2, so the output is bit-identical to the Python
//  reference (torch .max(dim) + xywh2xyxy).
// =============================================================================
#include "trt_alpha/kernels/ops/decode_yolo26.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops {
namespace {

// =============================================================================
//  Per-anchor decode kernel
// -----------------------------------------------------------------------------
//  For anchor a of image n:
//      conf / cls : max + argmax over the nc class channels [4, 4+nc)
//      box        : xywh (network pixels) -> xyxy, as ultralytics xywh2xyxy
//      extra      : copied verbatim
//
//  Reads are coalesced: for a fixed channel, consecutive threads touch
//  consecutive anchors.
// =============================================================================
__global__ void decodeYolo26Kernel(const float* __restrict__ y,
                                   float* __restrict__ cand,
                                   int nc, int extra, int anchors)
{
    const int a = (int)(blockIdx.x * blockDim.x + threadIdx.x);
    if (a >= anchors || blockIdx.y < 0)
    {
        return;
    }
    const int    n   = blockIdx.y;
    const int    yC  = 4 + nc + extra;      // channels of y
    const int    cS  = 6 + extra;           // channels of cand

    const float* __restrict__ yb = y + ((size_t)n * (size_t)yC * (size_t)anchors);
    float*       __restrict__ cb = cand + ((size_t)n * (size_t)anchors * (size_t)cS);

    // ---- class reduction: max value + its (first) index ----
    // torch.max returns the first maximal index; "v > best" keeps the same rule.
    const float* __restrict__ cls_ptr = yb + (size_t)4 * (size_t)anchors;
    float best  = cls_ptr[a];
    int   bestI = 0;
    for (int c = 1; c < nc; ++c)
    {
        const float v = cls_ptr[(size_t)c * (size_t)anchors + a];
        if (v > best) { best = v; bestI = c; }
    }

    // ---- xywh -> xyxy (w/2 and h/2 are exact, division by 2 is exact) ----
    const float cx = yb[a];
    const float cy = yb[(size_t)anchors + a];
    const float hw = yb[(size_t)2 * (size_t)anchors + a] * 0.5f;
    const float hh = yb[(size_t)3 * (size_t)anchors + a] * 0.5f;

    float* __restrict__ out = cb + (size_t)a * (size_t)cS;
    out[0] = cx - hw;
    out[1] = cy - hh;
    out[2] = cx + hw;
    out[3] = cy + hh;
    out[4] = best;
    out[5] = (float)bestI;

    const float* __restrict__ ex_ptr = cls_ptr + (size_t)nc * (size_t)anchors;
    for (int e = 0; e < extra; ++e)
    {
        out[6 + e] = ex_ptr[(size_t)e * (size_t)anchors + a];
    }
}

// =============================================================================
//  Launch helper
// =============================================================================
void launchDecode(cudaStream_t stream, int batch,
                  const float* y, float* cand,
                  int nc, int extra, int anchors)
{
    if (batch <= 0 || anchors <= 0 || nc <= 0)
    {
        return;
    }
    const dim3 block(kBlock1D, 1, 1);
    const dim3 grid((unsigned)((anchors + kBlock1D - 1) / kBlock1D), (unsigned)batch, 1);
    decodeYolo26Kernel<<<grid, block, 0, stream>>>(y, cand, nc, extra, anchors);
}

}  // namespace

// =============================================================================
//  CvDecodeYolo26Plan
// =============================================================================
CvDecodeYolo26Plan::CvDecodeYolo26Plan(const DecodeYolo26Config& cfg)
    : cfg_(cfg)
{
    if (cfg_.nc <= 0 || cfg_.anchors <= 0 || cfg_.extra < 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }
    err_ = cudaSuccess;
}

CvDecodeYolo26Plan::~CvDecodeYolo26Plan() = default;

CvDecodeYolo26Plan::CvDecodeYolo26Plan(CvDecodeYolo26Plan&& other) noexcept = default;

void CvDecodeYolo26Plan::launch(cudaStream_t stream, int batch,
                                const float* y, float* cand) const
{
    if (err_ != cudaSuccess || y == nullptr || cand == nullptr)
    {
        return;
    }
    launchDecode(stream, batch, y, cand, cfg_.nc, cfg_.extra, cfg_.anchors);
    err_ = cudaGetLastError();
}

// =============================================================================
//  decodeYolo26Device
// =============================================================================
void decodeYolo26Device(cudaStream_t stream, int batch,
                        const float* y, float* cand,
                        const DecodeYolo26Config& cfg)
{
    launchDecode(stream, batch, y, cand, cfg.nc, cfg.extra, cfg.anchors);
}

}  // namespace trt_alpha::kernels::ops
