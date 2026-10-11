// =============================================================================
//  trt_alpha :: kernels :: ops :: postprocess_fused_vec_yolo26_det
// -----------------------------------------------------------------------------
//  Implementation described at the top of postprocess_fused_vec_yolo26_det.hpp.
//
//  One block per image, four phases -- exactly like the scalar fused kernel:
//      1) decode + conf filter + compact
//      2) stable sort by conf                 (detail::rankSortByConf)
//      3) greedy NMS                          (detail::greedyRounds)
//      4) write count / dets
//
//  The only difference is phase 1: four consecutive anchors per thread, loaded
//  as 16B float4 per channel. anchors % 4 != 0 (and the tail quad) is handled by
//  a scalar loop, so any anchor count is legal.
// =============================================================================
#include "trt_alpha/kernels/ops/postprocess_fused_vec_yolo26_det.hpp"
#include "trt_alpha/kernels/common.cuh"

#include "nms_utral_detail.cuh"

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>

namespace trt_alpha::kernels::ops {
namespace {

using detail::greedyRounds;
using detail::rankSortByConf;

constexpr int kVec = 4;   //!< 4 consecutive anchors per thread (one float4)

// =============================================================================
//  commitOne -- one decoded candidate -> compacted candidate buffer
// -----------------------------------------------------------------------------
//  Same arithmetic as the scalar kernel: xywh -> xyxy first, then the atomic
//  reservation. The guard is `pos < anchors` (the buffer is sized for the worst
//  case where every anchor survives).
// =============================================================================
__device__ __forceinline__ void commitOne(
    int n, int nc, int extra, int anchors, int cS, int a,
    float cx, float cy, float w, float h, float conf, int clsI,
    const float* __restrict__ cls0,
    int* __restrict__ cnt, float* __restrict__ bb)
{
    const float hw = w * 0.5f;
    const float hh = h * 0.5f;

    const int pos = atomicAdd(&cnt[n], 1);
    if (pos >= anchors) { return; }

    float* __restrict__ dst = bb + (std::size_t)pos * (std::size_t)cS;
    dst[0] = cx - hw;
    dst[1] = cy - hh;
    dst[2] = cx + hw;
    dst[3] = cy + hh;
    dst[4] = conf;
    dst[5] = (float)clsI;

    if (extra > 0)
    {
        const float* __restrict__ ex = cls0 + (std::size_t)nc * (std::size_t)anchors;
        for (int e = 0; e < extra; ++e)
        {
            dst[6 + e] = ex[(std::size_t)e * (std::size_t)anchors + a];
        }
    }
}

// =============================================================================
//  Vectorized fused kernel -- one block per image
// =============================================================================
__global__ void proprocessFusedVecYolo26Kernel(
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

    __shared__ int sCur;
    __shared__ int sKeep;
    __shared__ int sStop;

    if (threadIdx.x == 0) { cnt[n] = 0; sCur = 0; sKeep = 0; sStop = 0; }
    __syncthreads();

    // ---- phase 1: decode each anchor, keep the ones above the threshold ----
    const float* __restrict__ cls0 = yb + (std::size_t)4 * (std::size_t)anchors;

    const int quads = anchors / kVec;    //!< number of complete float4 groups

    for (int q = (int)threadIdx.x; q < quads; q += (int)blockDim.x)
    {
        const int a0 = q * kVec;

        // class reduction: max value + first index, the same rule as torch.max.
        // The 4 lanes are independent, so the compare-and-select is byte for byte
        // the scalar one -- nothing is shared or approximated.
        float best0, best1, best2, best3;
        int   bestI0 = 0, bestI1 = 0, bestI2 = 0, bestI3 = 0;
        {
            const float4 v = *reinterpret_cast<const float4*>(cls0 + a0);
            best0 = v.x; best1 = v.y; best2 = v.z; best3 = v.w;
        }
        for (int c = 1; c < nc; ++c)
        {
            const float4 v = *reinterpret_cast<const float4*>(
                cls0 + (std::size_t)c * (std::size_t)anchors + a0);
            if (v.x > best0) { best0 = v.x; bestI0 = c; }
            if (v.y > best1) { best1 = v.y; bestI1 = c; }
            if (v.z > best2) { best2 = v.z; bestI2 = c; }
            if (v.w > best3) { best3 = v.w; bestI3 = c; }
        }

        const float4 c4 = *reinterpret_cast<const float4*>(yb + a0);
        const float4 y4 = *reinterpret_cast<const float4*>(yb + (std::size_t)anchors + a0);
        const float4 w4 = *reinterpret_cast<const float4*>(yb + (std::size_t)2 * (std::size_t)anchors + a0);
        const float4 h4 = *reinterpret_cast<const float4*>(yb + (std::size_t)3 * (std::size_t)anchors + a0);

        if (best0 > confThresh)
        {
            commitOne(n, nc, extra, anchors, cS, a0 + 0,
                      c4.x, y4.x, w4.x, h4.x, best0, bestI0, cls0, cnt, bb);
        }
        if (best1 > confThresh)
        {
            commitOne(n, nc, extra, anchors, cS, a0 + 1,
                      c4.y, y4.y, w4.y, h4.y, best1, bestI1, cls0, cnt, bb);
        }
        if (best2 > confThresh)
        {
            commitOne(n, nc, extra, anchors, cS, a0 + 2,
                      c4.z, y4.z, w4.z, h4.z, best2, bestI2, cls0, cnt, bb);
        }
        if (best3 > confThresh)
        {
            commitOne(n, nc, extra, anchors, cS, a0 + 3,
                      c4.w, y4.w, w4.w, h4.w, best3, bestI3, cls0, cnt, bb);
        }
    }

    // ---- phase 1 tail: the anchors past the last complete quad (scalar) ----
    for (int a = quads * kVec + (int)threadIdx.x; a < anchors; a += (int)blockDim.x)
    {
        float best  = cls0[a];
        int   bestI = 0;
        for (int c = 1; c < nc; ++c)
        {
            const float v = cls0[(std::size_t)c * (std::size_t)anchors + a];
            if (v > best) { best = v; bestI = c; }
        }
        if (!(best > confThresh)) { continue; }

        commitOne(n, nc, extra, anchors, cS, a,
                  yb[a], yb[(std::size_t)anchors + a],
                  yb[(std::size_t)2 * (std::size_t)anchors + a],
                  yb[(std::size_t)3 * (std::size_t)anchors + a],
                  best, bestI, cls0, cnt, bb);
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

    float* __restrict__ db = dets + (std::size_t)n * (std::size_t)maxDet * (std::size_t)cS;
    const int kept = greedyRounds(bb, ob, ab, M, cS,
                                  iouThresh, maxWh, maxDet,
                                  &sCur, &sKeep, &sStop, db);

    if (threadIdx.x == 0) { outCount[n] = kept; }
}

// =============================================================================
//  Launch helper
// =============================================================================
void launchFusedVec(cudaStream_t stream, int batch, const float* y,
                    int* count, float* dets,
                    const ProprocessFusedVecYolo26Config& cfg,
                    float* buf, int* order, unsigned char* alive, int* cnt)
{
    const dim3 blk((unsigned)kBlock1D);
    const dim3 grd((unsigned)batch);
    proprocessFusedVecYolo26Kernel<<<grd, blk, 0, stream>>>(
        y, cfg.nc, cfg.extra, cfg.anchors,
        cfg.confThresh, cfg.iouThresh, cfg.maxWh, cfg.maxDet,
        count, dets, buf, order, alive, cnt);
}

}  // namespace

// =============================================================================
//  CvProprocessFusedVecYolo26Plan
// =============================================================================
CvProprocessFusedVecYolo26Plan::CvProprocessFusedVecYolo26Plan(const ProprocessFusedVecYolo26Config& cfg)
    : cfg_(cfg)
{
    if (cfg_.anchors <= 0 || cfg_.nc <= 0 || cfg_.extra < 0 ||
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

CvProprocessFusedVecYolo26Plan::~CvProprocessFusedVecYolo26Plan()
{
    release();
}

CvProprocessFusedVecYolo26Plan::CvProprocessFusedVecYolo26Plan(CvProprocessFusedVecYolo26Plan&& other) noexcept
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

void CvProprocessFusedVecYolo26Plan::release() noexcept
{
    if (d_buf_)   { cudaFree(d_buf_);   d_buf_ = nullptr; }
    if (d_order_) { cudaFree(d_order_); d_order_ = nullptr; }
    if (d_alive_) { cudaFree(d_alive_); d_alive_ = nullptr; }
    if (d_cnt_)   { cudaFree(d_cnt_);   d_cnt_ = nullptr; }
}

void CvProprocessFusedVecYolo26Plan::launch(cudaStream_t stream, int batch,
                                            const float* y, int* count, float* dets) const
{
    if (err_ != cudaSuccess || y == nullptr || count == nullptr || dets == nullptr)
    {
        return;
    }
    if (batch <= 0) { return; }
    if (batch > cfg_.maxBatch) { err_ = cudaErrorInvalidValue; return; }

    // The float4 loads assume a 16B aligned base; a misaligned base would fault
    // (or, worse with some drivers, silently read the wrong bytes). Fail loudly
    // instead of guessing -- the same rule the engine follows for shapes.
    if ((reinterpret_cast<std::uintptr_t>(y) & (std::uintptr_t)(sizeof(float) * kVec - 1)) != 0)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    launchFusedVec(stream, batch, y, count, dets, cfg_, d_buf_, d_order_, d_alive_, d_cnt_);
    err_ = cudaGetLastError();
}

// =============================================================================
//  proprocessFusedVecYolo26Device
// =============================================================================
void proprocessFusedVecYolo26Device(cudaStream_t stream, int batch,
                                    const float* y, int* count, float* dets,
                                    const ProprocessFusedVecYolo26Config& cfg)
{
    CvProprocessFusedVecYolo26Plan plan(cfg);
    if (!plan.ok()) { return; }
    plan.launch(stream, batch, y, count, dets);
}

}  // namespace trt_alpha::kernels::ops
