// =============================================================================
//  trt_alpha :: kernels :: ops :: nms_utral
// -----------------------------------------------------------------------------
//  Ultralytics-style non-maximum suppression ("utral" = ultralytics), CUDA C.
//
//  Input  cand : float [batch][anchors][6 + extra]
//                = [x1, y1, x2, y2, conf, cls, extra...]   (see decode_yolov26)
//  Output count: int   [batch]                       number of detections kept
//         dets : float [batch][maxDet][6 + extra]     kept rows, in keep order
//                coordinates are the INPUT coordinates (network pixels, no
//                reverse letterbox mapping -- that is a separate op).
//
//  It reproduces, step by step, the hand-written reference in
//  E:/Code/ultralytics/sry_yolo26_det_one2many.py:
//
//      1) candidates   conf > confThresh                       (STEP 6.1)
//      2) class offset boxes_xyxy + cls * maxWh                (my_nms: offs)
//      3) order        stable sort by conf, descending          (argsort)
//      4) greedy       keep first, drop the rest with
//                      IoU > iouThresh, repeat                 (while loop)
//      5) cap at       maxDet
//
//  ------------------------------------------------------------ bit-exactness
//  The IoU is
//      inter / (areaA + areaB - inter + 1e-9)
//  i.e. the same +1e-9 the reference adds, evaluated in the same left-to-right
//  order, in float32 with IEEE division. Class offsets are computed once per box
//  (o = cls * maxWh) and added at IoU time, exactly like the reference's
//  `b = boxes_xyxy + offs[:, None]`, so no value is ever reconstructed by
//  subtracting a large offset.
//
//  Sorting is a stable descending sort (ties -> smaller index first). The
//  reference uses torch.argsort(descending=True), whose tie order is not
//  specified; for non-tied scores (the realistic case) the two agree.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  NmsUtralConfig
// =============================================================================
struct NmsUtralConfig
{
    int   anchors    = 8400;    //!< candidate rows per image (A)
    int   extra      = 0;       //!< extra channels carried through
    int   maxBatch   = 8;       //!< workspace is sized for this many images
    int   maxDet     = 300;     //!< max detections kept per image
    float confThresh = 0.25f;   //!< conf threshold
    float iouThresh  = 0.70f;   //!< IoU threshold
    float maxWh      = 7680.0f; //!< per-class coordinate offset (class-aware NMS)

    int candStride() const noexcept { return 6 + extra; }

    //! Workspace bytes for a given config (exposed so a caller can budget).
    static size_t workspaceBytes(const NmsUtralConfig& c) noexcept
    {
        const size_t cands = (size_t)c.maxBatch * (size_t)c.anchors;
        return cands * (size_t)c.candStride() * sizeof(float)   // compacted candidates
             + cands * sizeof(int)                              // sort order
             + cands * sizeof(unsigned char)                    // alive flags
             + (size_t)c.maxBatch * sizeof(int);                // per-image counts
    }
};

// =============================================================================
//  CvNmsUtralPlan
// -----------------------------------------------------------------------------
//  The constructor allocates the workspace once (geometry and thresholds are
//  fixed for a stream); launch() only issues the three kernels.
//
//  Usage:
//      CvNmsUtralPlan nms(cfg);
//      nms.launch(stream, batch, d_cand, d_count, d_dets);
// =============================================================================
class CvNmsUtralPlan
{
public:
    explicit CvNmsUtralPlan(const NmsUtralConfig& cfg);

    ~CvNmsUtralPlan();

    CvNmsUtralPlan(const CvNmsUtralPlan&)            = delete;
    CvNmsUtralPlan& operator=(const CvNmsUtralPlan&) = delete;
    CvNmsUtralPlan(CvNmsUtralPlan&& other) noexcept;

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    const NmsUtralConfig& config() const noexcept { return cfg_; }

    int detStride() const noexcept { return cfg_.candStride(); }

    // -------------------------------------------------------------------------
    //! Hot path: three kernels, no allocation / copy / sync.
    //!   cand  : batch * anchors * (6+extra)
    //!   count : batch ints           (written)
    //!   dets  : batch * maxDet * (6+extra)   (written)
    //! batch must be <= cfg.maxBatch.
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const float* cand, int* count, float* dets) const;

private:
    void release() noexcept;

    NmsUtralConfig cfg_;
    float*         d_buf_   = nullptr;   //!< compacted candidates
    int*           d_order_ = nullptr;   //!< sort order (indices into d_buf)
    unsigned char* d_alive_ = nullptr;   //!< suppression flags
    int*           d_cnt_   = nullptr;   //!< per-image candidate counts
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  nmsUtralDevice -- one-shot convenience version
// -----------------------------------------------------------------------------
//  Allocates and frees the workspace internally: fine for a single call or a
//  test, but the hot loop should keep a CvNmsUtralPlan around.
// =============================================================================
void nmsUtralDevice(cudaStream_t stream, int batch,
                    const float* cand, int* count, float* dets,
                    const NmsUtralConfig& cfg);

}  // namespace trt_alpha::kernels::ops
