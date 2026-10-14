// =============================================================================
//  trt_alpha :: kernels :: ops :: topk_yolov26_cls  (implementation)
// -----------------------------------------------------------------------------
//  One block per image. Each of the topK rounds performs a deterministic
//  parallel arg-max ("largest value, then smallest index") over the score
//  vector, skipping the entries already chosen, and thread 0 appends the winner.
//
//  Why shared memory: the score vector is read numClass times overall but only
//  once from global memory, and the round-to-round "already picked" mask lives
//  in shared memory too -- no global atomics, no per-round global traffic.
//
//  Cost: grid = batch blocks, block = 256 threads, dynamic shared =
//  numClass * (sizeof(float) + 1) bytes. For ImageNet (numClass = 1000) that is
//  ~5 KB per block. A classification head has no other work, so occupancy is
//  irrelevant here.
// =============================================================================
#include "trt_alpha/kernels/ops/topk_yolov26_cls.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cfloat>
#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

//! Sentinel for "no candidate yet". A plain constant, NOT
//! std::numeric_limits<int>::max(): that is a constexpr __host__ function and
//! nvcc rejects calling it from a __global__ without --expt-relaxed-constexpr.
constexpr int kNoIndex = 0x7fffffff;

//! Per-image score vector is addressed as scores[blockIdx.x * numClass + i].
__global__ void topkKernel(
    const float* __restrict__ scores,
    float* __restrict__ out,
    int numClass,
    int topK)
{
    // Layout of the dynamic shared block:
    //   float           s_val[numClass]   -- the scores, staged once
    //   unsigned char   s_used[numClass]  -- already-selected mask
    extern __shared__ unsigned char s_raw[];
    float*         s_val  = reinterpret_cast<float*>(s_raw);
    unsigned char* s_used = s_raw + static_cast<std::size_t>(numClass) * sizeof(float);

    const float* __restrict__ base = scores + static_cast<std::size_t>(blockIdx.x)
                                              * static_cast<std::size_t>(numClass);
    float* __restrict__ dst = out + static_cast<std::size_t>(blockIdx.x)
                                    * static_cast<std::size_t>(2) * static_cast<std::size_t>(topK);

    for (int i = (int)threadIdx.x; i < numClass; i += (int)blockDim.x)
    {
        s_val[i]  = base[i];
        s_used[i] = 0;
    }

    // Reduction scratch. kBlock1D is 256, so a fixed 256-slot array is enough --
    // guard with a compile-time assert instead of trusting it blindly.
    __shared__ float s_bestVal[kBlock1D];
    __shared__ int   s_bestIdx[kBlock1D];

    __syncthreads();

    for (int r = 0; r < topK; ++r)
    {
        // ---- gather this thread's local best among the untouched entries ----
        float bestVal = -FLT_MAX;
        int   bestIdx = kNoIndex;
        for (int i = (int)threadIdx.x; i < numClass; i += (int)blockDim.x)
        {
            if (s_used[i])
            {
                continue;
            }
            const float v = s_val[i];
            if (v > bestVal || (v == bestVal && i < bestIdx))
            {
                bestVal = v;
                bestIdx = i;
            }
        }

        s_bestVal[threadIdx.x] = bestVal;
        s_bestIdx[threadIdx.x] = bestIdx;
        __syncthreads();

        // ---- tree reduction with the identical comparison rule ----
        for (int s = (int)blockDim.x / 2; s > 0; s >>= 1)
        {
            if ((int)threadIdx.x < s)
            {
                const float ov = s_bestVal[threadIdx.x + s];
                const int   oi = s_bestIdx[threadIdx.x + s];
                if (ov > s_bestVal[threadIdx.x] ||
                    (ov == s_bestVal[threadIdx.x] && oi < s_bestIdx[threadIdx.x]))
                {
                    s_bestVal[threadIdx.x] = ov;
                    s_bestIdx[threadIdx.x] = oi;
                }
            }
            __syncthreads();
        }

        if (threadIdx.x == 0)
        {
            const int wi = s_bestIdx[0];
            // numClass > 0 is validated by the plan, so the sentinel cannot
            // survive unless every entry was already consumed (topK > numClass,
            // also rejected by the plan).
            if (wi >= 0 && wi < numClass)
            {
                s_used[wi] = 1;
            }
            dst[2 * r]     = static_cast<float>(wi);
            dst[2 * r + 1] = s_bestVal[0];
        }
        __syncthreads();
    }
}

}  // namespace

// =============================================================================
//  CvTopKPlan
// =============================================================================
CvTopKPlan::CvTopKPlan(int numClass, int topK, int maxBatch)
    : numClass_(numClass), topK_(topK), maxBatch_(maxBatch)
{
    if (numClass <= 0 || topK <= 0 || maxBatch <= 0 || topK > numClass)
    {
        TRT_LOG_ERROR("CvTopKPlan: bad geometry (numClass=" << numClass
                      << ", topK=" << topK << ", maxBatch=" << maxBatch << ")");
        err_ = cudaErrorInvalidValue;
        return;
    }
    // The label is carried through a float, so it must be exactly representable.
    if (numClass > (1 << 24))
    {
        TRT_LOG_ERROR("CvTopKPlan: numClass too large to carry a label in a float");
        err_ = cudaErrorInvalidValue;
        return;
    }
    err_ = cudaSuccess;
}

void CvTopKPlan::launch(cudaStream_t stream, int batch,
                        const float* scores, float* out) const
{
    if (err_ != cudaSuccess)
    {
        return;
    }
    if (scores == nullptr || out == nullptr || batch <= 0 || batch > maxBatch_)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    const std::size_t smem = static_cast<std::size_t>(numClass_) * (sizeof(float) + 1);
    topkKernel<<<batch, kBlock1D, smem, stream>>>(scores, out, numClass_, topK_);

    err_ = cudaGetLastError();
    if (err_ != cudaSuccess)
    {
        TRT_LOG_ERROR("CvTopKPlan: kernel launch failed: " << cudaGetErrorString(err_));
    }
}

void topKDevice(cudaStream_t stream, int batch,
                const float* scores, float* out,
                int numClass, int topK)
{
    CvTopKPlan plan(numClass, topK, batch > 0 ? batch : 1);
    plan.launch(stream, batch, scores, out);
    if (!plan.ok())
    {
        TRT_LOG_ERROR("topKDevice: " << cudaGetErrorString(plan.lastError()));
    }
}

}  // namespace trt_alpha::kernels::ops
