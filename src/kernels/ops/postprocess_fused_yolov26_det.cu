// =============================================================================
//  trt_alpha :: kernels :: ops :: postprocess_fused_yolov26_det
// -----------------------------------------------------------------------------
//  Implementation described at the top of postprocess_fused_yolov26_det.hpp.
//
//  One block per image, four phases inside it:
//      1) decode + conf filter + compact      (parallel over the anchors)
//      2) stable sort by conf                 (detail::rankSortByConf)
//      3) greedy NMS                          (detail::greedyRounds)
//      4) write count / dets
// =============================================================================
#include "trt_alpha/kernels/ops/postprocess_fused_yolov26_det.hpp"
#include "trt_alpha/kernels/common.cuh"

#include "nms_utral_detail.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

using detail::greedyRounds;
using detail::rankSortByConf;

// =============================================================================
//  Fused kernel -- one block per image
// =============================================================================
__global__ void proprocessFusedYoloV26Kernel(
    const float* __restrict__ y, int nc, int extra, int anchors,
    float confThresh, float iouThresh, float maxWh, int maxDet,
    int* __restrict__ outCount, float* __restrict__ dets,
    float* __restrict__ buf, int* __restrict__ order,
    unsigned char* __restrict__ alive, int* __restrict__ cnt)
{
    const int n  = blockIdx.x;
    const int cS = 6 + extra;
    const int yC = 4 + nc + extra;

    const float* __restrict__ yb = y + (size_t)n * (size_t)yC * (size_t)anchors;
    float*       __restrict__ bb = buf   + (size_t)n * (size_t)anchors * (size_t)cS;
    int*         __restrict__ ob = order + (size_t)n * (size_t)anchors;
    unsigned char* __restrict__ ab = alive + (size_t)n * (size_t)anchors;

    __shared__ int sCur;
    __shared__ int sKeep;
    __shared__ int sStop;

    if (threadIdx.x == 0) { cnt[n] = 0; sCur = 0; sKeep = 0; sStop = 0; }
    __syncthreads();

    // ---- phase 1: decode each anchor, keep the ones above the threshold ----
    const float* __restrict__ cls0 = yb + (size_t)4 * (size_t)anchors;
    for (int a = (int)threadIdx.x; a < anchors; a += (int)blockDim.x)
    {
        // class reduction: max value + first index, the same rule as torch.max
        float best  = cls0[a];
        int   bestI = 0;
        for (int c = 1; c < nc; ++c)
        {
            const float v = cls0[(size_t)c * (size_t)anchors + a];
            if (v > best) { best = v; bestI = c; }
        }
        if (!(best > confThresh)) { continue; }

        const float cx = yb[a];
        const float cy = yb[(size_t)anchors + a];
        const float hw = yb[(size_t)2 * (size_t)anchors + a] * 0.5f;
        const float hh = yb[(size_t)3 * (size_t)anchors + a] * 0.5f;

        const int pos = atomicAdd(&cnt[n], 1);
        if (pos < anchors)
        {
            float* __restrict__ dst = bb + (size_t)pos * (size_t)cS;
            dst[0] = cx - hw;
            dst[1] = cy - hh;
            dst[2] = cx + hw;
            dst[3] = cy + hh;
            dst[4] = best;
            dst[5] = (float)bestI;

            const float* __restrict__ ex = cls0 + (size_t)nc * (size_t)anchors;
            for (int e = 0; e < extra; ++e)
            {
                dst[6 + e] = ex[(size_t)e * (size_t)anchors + a];
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

    // ---- phase 2 + 3: sort, then greedy suppression ----
    for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) { ab[i] = 1; }
    rankSortByConf(bb, M, cS, ob);
    __syncthreads();

    float* __restrict__ db = dets + (size_t)n * (size_t)maxDet * (size_t)cS;
    const int kept = greedyRounds(bb, ob, ab, M, cS,
                                  iouThresh, maxWh, maxDet,
                                  &sCur, &sKeep, &sStop, db);

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
    proprocessFusedYoloV26Kernel<<<grd, blk, 0, stream>>>(
        y, cfg.nc, cfg.extra, cfg.anchors,
        cfg.confThresh, cfg.iouThresh, cfg.maxWh, cfg.maxDet,
        count, dets, buf, order, alive, cnt);
}

}  // namespace

// =============================================================================
//  CvProprocessFusedYoloV26Plan
// =============================================================================
CvProprocessFusedYoloV26Plan::CvProprocessFusedYoloV26Plan(const ProprocessFusedYoloV26Config& cfg)
    : cfg_(cfg)
{
    if (cfg_.anchors <= 0 || cfg_.nc <= 0 || cfg_.extra < 0 ||
        cfg_.maxBatch <= 0 || cfg_.maxDet < 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    const size_t cands = (size_t)cfg_.maxBatch * (size_t)cfg_.anchors;
    const size_t cS    = (size_t)cfg_.candStride();

    cudaError_t e = cudaMalloc(reinterpret_cast<void**>(&d_buf_), cands * cS * sizeof(float));
    if (e == cudaSuccess) { e = cudaMalloc(reinterpret_cast<void**>(&d_order_), cands * sizeof(int)); }
    if (e == cudaSuccess) { e = cudaMalloc(reinterpret_cast<void**>(&d_alive_), cands * sizeof(unsigned char)); }
    if (e == cudaSuccess) { e = cudaMalloc(reinterpret_cast<void**>(&d_cnt_), (size_t)cfg_.maxBatch * sizeof(int)); }

    if (e != cudaSuccess)
    {
        release();
        err_ = e;
        return;
    }
    err_ = cudaSuccess;
}

CvProprocessFusedYoloV26Plan::~CvProprocessFusedYoloV26Plan()
{
    release();
}

CvProprocessFusedYoloV26Plan::CvProprocessFusedYoloV26Plan(CvProprocessFusedYoloV26Plan&& other) noexcept
    : cfg_(other.cfg_)
    , d_buf_(other.d_buf_)
    , d_order_(other.d_order_)
    , d_alive_(other.d_alive_)
    , d_cnt_(other.d_cnt_)
    , err_(other.err_)
{
    other.d_buf_ = nullptr;
    other.d_order_ = nullptr;
    other.d_alive_ = nullptr;
    other.d_cnt_ = nullptr;
    other.err_ = cudaSuccess;
}

void CvProprocessFusedYoloV26Plan::release() noexcept
{
    if (d_buf_)   { cudaFree(d_buf_);   d_buf_ = nullptr; }
    if (d_order_) { cudaFree(d_order_); d_order_ = nullptr; }
    if (d_alive_) { cudaFree(d_alive_); d_alive_ = nullptr; }
    if (d_cnt_)   { cudaFree(d_cnt_);   d_cnt_ = nullptr; }
}

void CvProprocessFusedYoloV26Plan::launch(cudaStream_t stream, int batch,
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
//  proprocessFusedYoloV26Device
// =============================================================================
void proprocessFusedYoloV26Device(cudaStream_t stream, int batch,
                                 const float* y, int* count, float* dets,
                                 const ProprocessFusedYoloV26Config& cfg)
{
    CvProprocessFusedYoloV26Plan plan(cfg);
    if (!plan.ok()) { return; }
    plan.launch(stream, batch, y, count, dets);
}

}  // namespace trt_alpha::kernels::ops
