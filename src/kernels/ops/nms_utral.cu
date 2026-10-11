// =============================================================================
//  trt_alpha :: kernels :: ops :: nms_utral
// -----------------------------------------------------------------------------
//  Implementation described at the top of nms_utral.hpp.
//  Three kernels, one block per image each:
//      1) nmsCompactKernel  conf filter + compaction
//      2) nmsSortKernel     stable descending sort by conf  (detail::rankSortByConf)
//      3) nmsGreedyKernel   the reference's greedy while-loop, parallelised round
//                           by round                          (detail::greedyRounds)
//
//  The sort and the greedy loop live in nms_utral_detail.cuh because the two
//  fused variants (postprocess_fused_yolo26_det / _vec) run the very same code after
//  decoding in place.
// =============================================================================
#include "trt_alpha/kernels/ops/nms_utral.hpp"
#include "trt_alpha/kernels/common.cuh"

#include "nms_utral_detail.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

using detail::greedyRounds;
using detail::rankSortByConf;

// =============================================================================
//  1) confidence filter + compaction
// =============================================================================
__global__ void nmsCompactKernel(const float* __restrict__ cand,
                                 int anchors, int cS, float confThresh,
                                 int* __restrict__ cnt,
                                 float* __restrict__ buf, int maxCand)
{
    const int n = blockIdx.x;
    if (threadIdx.x == 0) { cnt[n] = 0; }
    __syncthreads();

    const float* __restrict__ cb = cand + (size_t)n * (size_t)anchors * (size_t)cS;
    float*       __restrict__ bb = buf  + (size_t)n * (size_t)maxCand * (size_t)cS;

    for (int a = (int)threadIdx.x; a < anchors; a += (int)blockDim.x)
    {
        const float* __restrict__ src = cb + (size_t)a * (size_t)cS;
        if (src[4] > confThresh)                       // conf is channel 4
        {
            const int pos = atomicAdd(&cnt[n], 1);
            if (pos < maxCand)
            {
                float* __restrict__ dst = bb + (size_t)pos * (size_t)cS;
                for (int k = 0; k < cS; ++k) { dst[k] = src[k]; }
            }
        }
    }
}

// =============================================================================
//  2) stable descending sort by conf
// =============================================================================
__global__ void nmsSortKernel(const float* __restrict__ buf,
                              const int* __restrict__ cnt,
                              int maxCand, int cS,
                              int* __restrict__ order)
{
    const int n = blockIdx.x;
    const int M = cnt[n];
    if (M <= 0) { return; }

    const float* __restrict__ bb = buf   + (size_t)n * (size_t)maxCand * (size_t)cS;
    int*         __restrict__ ob = order + (size_t)n * (size_t)maxCand;

    rankSortByConf(bb, M, cS, ob);
}

// =============================================================================
//  3) greedy suppression
// =============================================================================
__global__ void nmsGreedyKernel(const float* __restrict__ buf,
                                const int* __restrict__ cnt,
                                const int* __restrict__ order,
                                int maxCand, int cS,
                                float iouThresh, float maxWh, int maxDet,
                                int* __restrict__ outCount,
                                float* __restrict__ dets,
                                unsigned char* __restrict__ alive)
{
    const int n = blockIdx.x;
    const int M = cnt[n];

    __shared__ int sCur;
    __shared__ int sKeep;
    __shared__ int sStop;

    unsigned char* __restrict__ ab = alive + (size_t)n * (size_t)maxCand;
    for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x) { ab[i] = 1; }

    if (threadIdx.x == 0) { sCur = 0; sKeep = 0; sStop = 0; }
    __syncthreads();

    if (M <= 0 || maxDet <= 0)
    {
        if (threadIdx.x == 0) { outCount[n] = 0; }
        return;
    }

    const float* __restrict__ bb = buf   + (size_t)n * (size_t)maxCand * (size_t)cS;
    const int*   __restrict__ ob = order + (size_t)n * (size_t)maxCand;
    float*       __restrict__ db = dets  + (size_t)n * (size_t)maxDet  * (size_t)cS;

    const int kept = greedyRounds(bb, ob, ab, M, cS,
                                  iouThresh, maxWh, maxDet,
                                  &sCur, &sKeep, &sStop, db);

    if (threadIdx.x == 0) { outCount[n] = kept; }
}

// =============================================================================
//  Launch helper
// =============================================================================
void launchNms(cudaStream_t stream, int batch,
               const float* cand, int* count, float* dets,
               const NmsUtralConfig& cfg,
               float* buf, int* order, unsigned char* alive, int* cnt)
{
    const int   cS  = cfg.candStride();
    const dim3  blk((unsigned)kBlock1D);
    const dim3  grd((unsigned)batch);

    nmsCompactKernel<<<grd, blk, 0, stream>>>(
        cand, cfg.anchors, cS, cfg.confThresh, cnt, buf, cfg.anchors);
    nmsSortKernel<<<grd, blk, 0, stream>>>(
        buf, cnt, cfg.anchors, cS, order);
    nmsGreedyKernel<<<grd, blk, 0, stream>>>(
        buf, cnt, order, cfg.anchors, cS,
        cfg.iouThresh, cfg.maxWh, cfg.maxDet, count, dets, alive);
}

}  // namespace

// =============================================================================
//  CvNmsUtralPlan
// =============================================================================
CvNmsUtralPlan::CvNmsUtralPlan(const NmsUtralConfig& cfg)
    : cfg_(cfg)
{
    if (cfg_.anchors <= 0 || cfg_.extra < 0 || cfg_.maxBatch <= 0 || cfg_.maxDet < 0)
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

CvNmsUtralPlan::~CvNmsUtralPlan()
{
    release();
}

CvNmsUtralPlan::CvNmsUtralPlan(CvNmsUtralPlan&& other) noexcept
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

void CvNmsUtralPlan::release() noexcept
{
    if (d_buf_)   { cudaFree(d_buf_);   d_buf_ = nullptr; }
    if (d_order_) { cudaFree(d_order_); d_order_ = nullptr; }
    if (d_alive_) { cudaFree(d_alive_); d_alive_ = nullptr; }
    if (d_cnt_)   { cudaFree(d_cnt_);   d_cnt_ = nullptr; }
}

void CvNmsUtralPlan::launch(cudaStream_t stream, int batch,
                            const float* cand, int* count, float* dets) const
{
    if (err_ != cudaSuccess || cand == nullptr || count == nullptr || dets == nullptr)
    {
        return;
    }
    if (batch <= 0)
    {
        return;
    }
    if (batch > cfg_.maxBatch)
    {
        err_ = cudaErrorInvalidValue;   // workspace was sized for maxBatch
        return;
    }

    launchNms(stream, batch, cand, count, dets, cfg_, d_buf_, d_order_, d_alive_, d_cnt_);
    err_ = cudaGetLastError();
}

// =============================================================================
//  nmsUtralDevice
// =============================================================================
void nmsUtralDevice(cudaStream_t stream, int batch,
                    const float* cand, int* count, float* dets,
                    const NmsUtralConfig& cfg)
{
    CvNmsUtralPlan plan(cfg);
    if (!plan.ok())
    {
        return;
    }
    plan.launch(stream, batch, cand, count, dets);
}

}  // namespace trt_alpha::kernels::ops
