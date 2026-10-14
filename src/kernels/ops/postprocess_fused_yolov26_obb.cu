// =============================================================================
//  trt_alpha :: kernels :: ops :: postprocess_fused_yolov26_obb  (implementation)
// -----------------------------------------------------------------------------
//  One block per image, three phases:
//      1) conf filter + compact          (parallel over the anchors)
//      2) stable sort by conf            (detail::rankSortByConf, shared)
//      3) Fast-NMS with probiou          (this file)
//
//  Only phase 1's box half and phase 3 differ from the detection operator, which
//  is exactly why they are separate plans instead of one plan with a flag: the
//  shared pieces (the conf sort) come from nms_utral_detail.cuh, while the IoU
//  is a genuinely different function, not a parameter of the existing one.
// =============================================================================
#include "trt_alpha/kernels/ops/postprocess_fused_yolov26_obb.hpp"

#include "trt_alpha/kernels/common.cuh"

#include "nms_utral_detail.cuh"   // detail::rankSortByConf

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

using detail::rankSortByConf;

//! batch_probiou's epsilon (ultralytics/utils/metrics.py:287).
constexpr float kProbEps = 1e-7f;

// =============================================================================
//  Probabilistic IoU between two oriented boxes -- a line-by-line port of
//  ultralytics batch_probiou (metrics.py:287-320) with eps = 1e-7.
//
//      a1,b1,c1 = covariance of box a      (a = w^2/12, b = h^2/12, c = angle)
//      a2,b2,c2 = covariance of box b
//      t1 = ((a1+a2)*(dy)^2 + (b1+b2)*(dx)^2) / ((a1+a2)*(b1+b2) - (c1+c2)^2 + eps) * 0.25
//      t2 = ((c1+c2)*(x2-x1)*(y1-y2))       / ((a1+a2)*(b1+b2) - (c1+c2)^2 + eps) * 0.5
//      t3 = log( ((a1+a2)*(b1+b2) - (c1+c2)^2)
//                / (4*sqrt(max(a1*b1-c1^2,0) * max(a2*b2-c2^2,0)) + eps) + eps ) * 0.5
//      bd = clamp(t1+t2+t3, eps, 100)
//      hd = sqrt(1 - exp(-bd) + eps)
//      iou = 1 - hd
//
//  Rows are the OBB candidate layout [cx, cy, w, h, conf, cls, angle], so the
//  angle sits at index 6 -- NOT at index 4 as in the bare xywhr the reference
//  feeds in (the reference rebuilds the row before NMS, adding conf/cls).
//
//  As in the reference, the class offset (cls * maxWh) is added to the stored
//  centres rather than reconstructing class-free coordinates by subtracting a
//  large number (which would not be exact in float32).
// =============================================================================
__device__ __forceinline__ float iouProbiou(const float* __restrict__ a,
                                            const float* __restrict__ b,
                                            float maxWh)
{
    const float ca = a[5] * maxWh;
    const float cb = b[5] * maxWh;
    const float ax = a[0] + ca;
    const float ay = a[1] + ca;
    const float bx = b[0] + cb;
    const float by = b[1] + cb;

    // covariance: a = w^2/12, b = h^2/12, c = (a - b) * cos * sin
    const float a1 = (a[2] * a[2]) / 12.0f;
    const float b1 = (a[3] * a[3]) / 12.0f;
    const float a2 = (b[2] * b[2]) / 12.0f;
    const float b2 = (b[3] * b[3]) / 12.0f;

    const float cos1 = cosf(a[6]);
    const float sin1 = sinf(a[6]);
    const float cos2 = cosf(b[6]);
    const float sin2 = sinf(b[6]);

    const float A1 = a1 * (cos1 * cos1) + b1 * (sin1 * sin1);
    const float B1 = a1 * (sin1 * sin1) + b1 * (cos1 * cos1);
    const float C1 = (a1 - b1) * cos1 * sin1;

    const float A2 = a2 * (cos2 * cos2) + b2 * (sin2 * sin2);
    const float B2 = a2 * (sin2 * sin2) + b2 * (cos2 * cos2);
    const float C2 = (a2 - b2) * cos2 * sin2;

    const float asum = A1 + A2;
    const float bsum = B1 + B2;
    const float csum = C1 + C2;

    const float dy = ay - by;
    const float dx = ax - bx;
    const float den = asum * bsum - csum * csum + kProbEps;

    const float t1 = ((asum * dy * dy + bsum * dx * dx) / den) * 0.25f;
    const float t2 = ((csum * (bx - ax) * (ay - by)) / den) * 0.5f;

    const float det1 = fmaxf(A1 * B1 - C1 * C1, 0.0f);
    const float det2 = fmaxf(A2 * B2 - C2 * C2, 0.0f);
    const float t3 = logf((asum * bsum - csum * csum)
                          / (4.0f * sqrtf(det1 * det2) + kProbEps) + kProbEps) * 0.5f;

    const float bd = fminf(fmaxf(t1 + t2 + t3, kProbEps), 100.0f);
    const float hd = sqrtf(1.0f - expf(-bd) + kProbEps);
    return 1.0f - hd;
}

// =============================================================================
//  Fast-NMS -- the rotated branch's suppression rule, NOT the greedy one
// -----------------------------------------------------------------------------
//  The array is already sorted by descending conf.  Box j survives iff no
//  earlier (higher-scoring) box has probiou >= iouThresh with it -- the parallel
//  form of `(ious.triu(1) >= thresh).sum(0) <= 0`.
//
//  Survivors are emitted in visit order, which IS the reference's order
//  (`sorted_idx[nonzero(...)]`, and nonzero returns ascending indices).
//
//  sKeep / sHit are __shared__ owned by the caller; must be called by every
//  thread of the block. Returns the number kept.
// =============================================================================
__device__ __forceinline__ int fastRoundsProbiou(const float* __restrict__ bb,
                                                 const int*   __restrict__ ob,
                                                 int M, int cS,
                                                 float iouThresh, float maxWh, int maxDet,
                                                 int* sKeep, unsigned char* sHit,
                                                 float* __restrict__ out)
{
    for (int j = 0; j < M; ++j)
    {
        if (*sKeep >= maxDet) { break; }

        if (threadIdx.x == 0) { *sHit = 0; }
        __syncthreads();

        const float* __restrict__ bj = bb + (std::size_t)ob[j] * (std::size_t)cS;
        for (int i = (int)threadIdx.x; i < j; i += (int)blockDim.x)
        {
            const float* __restrict__ bi = bb + (std::size_t)ob[i] * (std::size_t)cS;
            if (iouProbiou(bi, bj, maxWh) >= iouThresh)
            {
                *sHit = 1;   // concurrent writes of the same value: benign
                break;
            }
        }
        __syncthreads();

        if (*sHit == 0 && threadIdx.x == 0)
        {
            float* __restrict__ dst = out + (std::size_t)(*sKeep) * (std::size_t)cS;
            for (int t = 0; t < cS; ++t) { dst[t] = bj[t]; }
            ++(*sKeep);
        }
        __syncthreads();
    }
    return *sKeep;
}

// =============================================================================
//  Fused kernel -- one block per image
// =============================================================================
__global__ void postprocessFusedYoloV26ObbKernel(
    const float* __restrict__ y, int nc, int extra, int anchors,
    float confThresh, float iouThresh, float maxWh, int maxDet,
    int* __restrict__ outCount, float* __restrict__ dets,
    float* __restrict__ buf, int* __restrict__ order,
    unsigned char* __restrict__ alive, int* __restrict__ cnt)
{
    const int n  = blockIdx.x;
    const int cS = 6 + extra;
    const int yC = 4 + nc + extra;

    const float* __restrict__ yb = y + (std::size_t)n * (std::size_t)yC * (std::size_t)anchors;
    float*       __restrict__ bb = buf   + (std::size_t)n * (std::size_t)anchors * (std::size_t)cS;
    int*         __restrict__ ob = order + (std::size_t)n * (std::size_t)anchors;
    unsigned char* __restrict__ ab = alive + (std::size_t)n * (std::size_t)anchors;

    __shared__ int sKeep;
    __shared__ unsigned char sHit;

    if (threadIdx.x == 0) { cnt[n] = 0; sKeep = 0; sHit = 0; }
    __syncthreads();

    // ---- phase 1: filter each anchor by max class score, keep xywh + angle ----
    const float* __restrict__ cls0 = yb + (std::size_t)4 * (std::size_t)anchors;
    for (int a = (int)threadIdx.x; a < anchors; a += (int)blockDim.x)
    {
        // class reduction: max value + first index, the same rule as torch.max
        float best  = cls0[a];
        int   bestI = 0;
        for (int c = 1; c < nc; ++c)
        {
            const float v = cls0[(std::size_t)c * (std::size_t)anchors + a];
            if (v > best) { best = v; bestI = c; }
        }
        if (!(best > confThresh)) { continue; }

        const int pos = atomicAdd(&cnt[n], 1);
        if (pos < anchors)
        {
            float* __restrict__ dst = bb + (std::size_t)pos * (std::size_t)cS;
            // xywh is kept as-is: the rotated branch never runs xywh2xyxy, so the
            // candidate row is already in the reference's post-NMS layout.
            dst[0] = yb[a];
            dst[1] = yb[(std::size_t)anchors + a];
            dst[2] = yb[(std::size_t)2 * (std::size_t)anchors + a];
            dst[3] = yb[(std::size_t)3 * (std::size_t)anchors + a];
            dst[4] = best;
            dst[5] = (float)bestI;

            const float* __restrict__ ex = cls0 + (std::size_t)nc * (std::size_t)anchors;
            for (int e = 0; e < extra; ++e)
            {
                dst[6 + e] = ex[(std::size_t)e * (std::size_t)anchors + a];
            }
        }
    }
    __syncthreads();

    const int M = cnt[n];
    if (M <= 0 || maxDet <= 0)
    {
        if (threadIdx.x == 0) { outCount[n] = 0; }
        return;
    }

    // ---- phase 2 + 3: sort, then Fast-NMS ----
    for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) { ab[i] = 1; }
    rankSortByConf(bb, M, cS, ob);
    __syncthreads();

    float* __restrict__ db = dets + (std::size_t)n * (std::size_t)maxDet * (std::size_t)cS;
    const int kept = fastRoundsProbiou(bb, ob, M, cS,
                                       iouThresh, maxWh, maxDet, &sKeep, &sHit, db);

    if (threadIdx.x == 0) { outCount[n] = kept; }
}

// =============================================================================
//  Launch helper
// =============================================================================
void launchFused(cudaStream_t stream, int batch, const float* y,
                 int* count, float* dets,
                 const ProprocessFusedYoloV26Config& cfg,
                 float* buf, int* order, unsigned char* alive, int* cnt)
{
    const dim3 blk((unsigned)kBlock1D);
    const dim3 grd((unsigned)batch);
    postprocessFusedYoloV26ObbKernel<<<grd, blk, 0, stream>>>(
        y, cfg.nc, cfg.extra, cfg.anchors,
        cfg.confThresh, cfg.iouThresh, cfg.maxWh, cfg.maxDet,
        count, dets, buf, order, alive, cnt);
}

}  // namespace

// =============================================================================
//  CvPostprocessFusedYoloV26ObbPlan
// =============================================================================
CvPostprocessFusedYoloV26ObbPlan::CvPostprocessFusedYoloV26ObbPlan(const ProprocessFusedYoloV26Config& cfg)
    : cfg_(cfg)
{
    // extra >= 1: the rotated branch reads the ANGLE from the extra block, so a
    // config without it would silently suppress on a constant angle.
    if (cfg_.anchors <= 0 || cfg_.nc <= 0 || cfg_.extra < 1 ||
        cfg_.maxBatch <= 0 || cfg_.maxDet < 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    const std::size_t cands = (std::size_t)cfg_.maxBatch * (std::size_t)cfg_.anchors;
    const std::size_t cS    = (std::size_t)cfg_.candStride();

    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_buf_), cands * cS * sizeof(float));
    if (e == cudaSuccess) { e = cudaMalloc(reinterpret_cast<void**>(&d_order_), cands * sizeof(int)); }
    if (e == cudaSuccess) { e = cudaMalloc(reinterpret_cast<void**>(&d_alive_), cands * sizeof(unsigned char)); }
    if (e == cudaSuccess) { e = cudaMalloc(reinterpret_cast<void**>(&d_cnt_), (std::size_t)cfg_.maxBatch * sizeof(int)); }

    if (e != cudaSuccess)
    {
        release();
        err_ = e;
        return;
    }
    err_ = cudaSuccess;
}

CvPostprocessFusedYoloV26ObbPlan::~CvPostprocessFusedYoloV26ObbPlan()
{
    release();
}

CvPostprocessFusedYoloV26ObbPlan::CvPostprocessFusedYoloV26ObbPlan(CvPostprocessFusedYoloV26ObbPlan&& other) noexcept
    : cfg_(other.cfg_)
    , d_buf_(other.d_buf_)
    , d_order_(other.d_order_)
    , d_alive_(other.d_alive_)
    , d_cnt_(other.d_cnt_)
    , err_(other.err_)
{
    other.d_buf_   = nullptr;
    other.d_order_ = nullptr;
    other.d_alive_ = nullptr;
    other.d_cnt_   = nullptr;
    other.err_     = cudaSuccess;
}

void CvPostprocessFusedYoloV26ObbPlan::release() noexcept
{
    if (d_buf_)   { cudaFree(d_buf_);   d_buf_   = nullptr; }
    if (d_order_) { cudaFree(d_order_); d_order_ = nullptr; }
    if (d_alive_) { cudaFree(d_alive_); d_alive_ = nullptr; }
    if (d_cnt_)   { cudaFree(d_cnt_);   d_cnt_   = nullptr; }
}

void CvPostprocessFusedYoloV26ObbPlan::launch(cudaStream_t stream, int batch,
                                              const float* y, int* count, float* dets) const
{
    if (err_ != cudaSuccess || y == nullptr || count == nullptr || dets == nullptr)
    {
        return;
    }
    if (batch <= 0) { return; }
    if (batch > cfg_.maxBatch) { err_ = cudaErrorInvalidValue; return; }

    launchFused(stream, batch, y, count, dets, cfg_, d_buf_, d_order_, d_alive_, d_cnt_);
    err_ = cudaGetLastError();
}

// =============================================================================
//  postprocessFusedYoloV26ObbDevice
// =============================================================================
void postprocessFusedYoloV26ObbDevice(cudaStream_t stream, int batch,
                                      const float* y, int* count, float* dets,
                                      const ProprocessFusedYoloV26Config& cfg)
{
    CvPostprocessFusedYoloV26ObbPlan plan(cfg);
    if (!plan.ok()) { return; }
    plan.launch(stream, batch, y, count, dets);
}

}  // namespace trt_alpha::kernels::ops
