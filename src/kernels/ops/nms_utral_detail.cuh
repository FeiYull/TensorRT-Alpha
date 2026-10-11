// =============================================================================
//  trt_alpha :: kernels :: ops :: nms_utral_detail  (private)
// -----------------------------------------------------------------------------
//  The device-side core of NMS, shared by three translation units:
//
//      nms_utral.cu                  the standalone operator (decode then NMS)
//      proprocess_fused_yolo26_det.cu    decode + filter + NMS in ONE kernel
//      proprocess_fused_vec_yolo26_det.cu   same, with a vectorized decode
//
//  Keeping the sort and the greedy loop here (instead of copy-pasting them) is
//  the same reasoning that made computeLetterBox a single function: three copies
//  of a subtle algorithm WILL drift. The file lives in src/ (not include/) on
//  purpose -- it is not part of the public surface.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops::detail {

// =============================================================================
//  Reference IoU, verbatim from sry_yolo26_det_one2many.py::bbox_iou_one2many
// -----------------------------------------------------------------------------
//      x1 = max(a.x1, b.x1) ...  inter = clamp(x2-x1, 0) * clamp(y2-y1, 0)
//      iou = inter / (area_a + area_b - inter + 1e-9)
//
//  The class offset (cls * maxWh) is added at use time, exactly like the
//  reference's `b = boxes_xyxy + offs[:, None]`, so the stored coordinates are
//  never reconstructed by subtracting a large offset (which would not be exact
//  in float32).
// =============================================================================
__device__ __forceinline__ float iouRef(const float* __restrict__ a,
                                        const float* __restrict__ b,
                                        float maxWh)
{
    const float oa = a[5] * maxWh;
    const float ob = b[5] * maxWh;

    const float ax1 = a[0] + oa, ay1 = a[1] + oa, ax2 = a[2] + oa, ay2 = a[3] + oa;
    const float bx1 = b[0] + ob, by1 = b[1] + ob, bx2 = b[2] + ob, by2 = b[3] + ob;

    const float x1 = fmaxf(ax1, bx1);
    const float y1 = fmaxf(ay1, by1);
    const float x2 = fminf(ax2, bx2);
    const float y2 = fminf(ay2, by2);

    const float inter = fmaxf(x2 - x1, 0.0f) * fmaxf(y2 - y1, 0.0f);
    const float areaA = (ax2 - ax1) * (ay2 - ay1);
    const float areaB = (bx2 - bx1) * (by2 - by1);

    return inter / (areaA + areaB - inter + 1e-9f);
}

// =============================================================================
//  Stable descending sort by conf (column 4 of a candidate row)
// -----------------------------------------------------------------------------
//      rank(i) = #{j : conf_j > conf_i} + #{j < i : conf_j == conf_i}
//
//  Ranks are unique, so `order[rank] = i` is race free. torch.argsort(
//  descending=True) does not specify its tie order; for non-tied scores (the
//  realistic case) this stable order matches it.
//
//  Must be called by every thread of the block (it strides by blockDim.x).
// =============================================================================
__device__ __forceinline__ void rankSortByConf(const float* __restrict__ bb,
                                               int M, int cS,
                                               int* __restrict__ ob)
{
    for (int i = (int)threadIdx.x; i < M; i += (int)blockDim.x)
    {
        const float si = bb[(std::size_t)i * (std::size_t)cS + 4];
        int rank = 0;
        for (int j = 0; j < M; ++j)
        {
            if (j == i) { continue; }
            const float sj = bb[(std::size_t)j * (std::size_t)cS + 4];
            if (sj > si || (sj == si && j < i)) { ++rank; }
        }
        ob[rank] = i;
    }
}

// =============================================================================
//  Greedy suppression -- the reference's while loop, one round per kept box
// -----------------------------------------------------------------------------
//  Each round: the best surviving box is emitted, then every later box whose
//  IoU with it exceeds iouThresh is dropped. Because the array is sorted by
//  descending conf and the rounds visit it in exactly that order, this is the
//  same decision sequence as the sequential reference loop.
//
//  sCur / sKeep / sStop are __shared__ scalars owned by the caller. Must be
//  called by every thread of the block; returns the number kept.
// =============================================================================
__device__ __forceinline__ int greedyRounds(const float* __restrict__ bb,
                                            const int*   __restrict__ ob,
                                            unsigned char* __restrict__ ab,
                                            int M, int cS,
                                            float iouThresh, float maxWh, int maxDet,
                                            int* sCur, int* sKeep, int* sStop,
                                            float* __restrict__ out)
{
    while (true)
    {
        // ---- thread 0 picks the best surviving box and emits it ----
        if (threadIdx.x == 0)
        {
            int k = *sCur;
            while (k < M && ab[k] == 0) { ++k; }
            if (k >= M || *sKeep >= maxDet)
            {
                *sStop = 1;
            }
            else
            {
                const float* src = bb + (std::size_t)ob[k] * (std::size_t)cS;
                float*       dst = out + (std::size_t)(*sKeep) * (std::size_t)cS;
                for (int t = 0; t < cS; ++t) { dst[t] = src[t]; }
                *sCur = k + 1;
                ++(*sKeep);
            }
        }
        __syncthreads();
        if (*sStop != 0) { break; }

        // ---- drop every later box that overlaps it too much ----
        const int    k = *sCur - 1;               // the box just kept
        const float* a = bb + (std::size_t)ob[k] * (std::size_t)cS;
        for (int j = k + 1 + (int)threadIdx.x; j < M; j += (int)blockDim.x)
        {
            if (ab[j] == 0) { continue; }
            const float* b = bb + (std::size_t)ob[j] * (std::size_t)cS;
            if (iouRef(a, b, maxWh) > iouThresh) { ab[j] = 0; }
        }
        __syncthreads();
    }
    return *sKeep;
}

}  // namespace trt_alpha::kernels::ops::detail
