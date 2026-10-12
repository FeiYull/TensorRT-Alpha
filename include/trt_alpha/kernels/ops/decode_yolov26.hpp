// =============================================================================
//  trt_alpha :: kernels :: ops :: decode_yolov26
// -----------------------------------------------------------------------------
//  Turn a YOLOV26 one2many detection-head output into one candidate per anchor.
//
//  The exported ONNX already contains make_anchors + dist2bbox + "* stride" +
//  class sigmoid, so the engine's single output is
//
//      y : float  [batch][4 + nc + extra][A]            (channels first)
//
//          y[c][a], c in [0,4)        box centre + size, xywh, network pixels
//          c in [4, 4+nc)             class probabilities (sigmoid already applied)
//          c in [4+nc, 4+nc+extra)    task "extra" channels, carried through as-is
//                                     (seg: 32 mask coefficients, pose: 51 keypoints)
//
//  This operator only re-shapes / reduces it to one row per anchor:
//
//      cand : float  [batch][A][6 + extra] = [x1, y1, x2, y2, conf, cls, extra...]
//
//          conf = max over the nc class channels, cls = its index (first wins)
//          xywh -> xyxy exactly as ultralytics' xywh2xyxy:
//              x1 = cx - w/2, y1 = cy - h/2, x2 = cx + w/2, y2 = cy + h/2
//
//  It does NOT threshold: filtering belongs to nms_utral. Keeping this a pure
//  function makes it trivially comparable with the Python reference.
//
//  ------------------------------------------------------------ bit-exactness
//  Every step above is a single IEEE-754 float32 add / sub / divide-by-2, so the
//  result is bit-identical to Python's (torch .max(dim) + xywh2xyxy): no
//  transcendental function is involved anywhere.
//
//  ---------------------------------------------------------- memory layout
//  Thread mapping: x covers the anchors, y is the batch index. For a fixed
//  channel the anchors are contiguous, so consecutive threads read consecutive
//  addresses (coalesced), and the batch stride is 4+nc+extra channels.
// =============================================================================
#pragma once

#include <cuda_runtime.h>

namespace trt_alpha::kernels::ops {

// =============================================================================
//  DecodeYoloV26Config -- geometry of one decode pass (batch never enters it)
// =============================================================================
struct DecodeYoloV26Config
{
    int nc      = 80;     //!< number of classes
    int extra   = 0;      //!< extra channels carried through (seg 32 / pose 51)
    int anchors = 8400;   //!< A: anchors per image (sum of h*w over the 3 levels)

    //! Channels of y per anchor.
    int yChannels() const noexcept { return 4 + nc + extra; }
    //! Channels of cand per anchor.
    int candStride() const noexcept { return 6 + extra; }
};

// =============================================================================
//  CvDecodeYoloV26Plan -- interface fixed in the constructor, only launch in the
//  loop (mirrors CvPreprocessPlan / CvResizePlan in this directory)
// =============================================================================
class CvDecodeYoloV26Plan
{
public:
    explicit CvDecodeYoloV26Plan(const DecodeYoloV26Config& cfg);

    ~CvDecodeYoloV26Plan();

    CvDecodeYoloV26Plan(const CvDecodeYoloV26Plan&)            = delete;
    CvDecodeYoloV26Plan& operator=(const CvDecodeYoloV26Plan&) = delete;
    CvDecodeYoloV26Plan(CvDecodeYoloV26Plan&& other) noexcept;

    bool        ok()        const noexcept { return err_ == cudaSuccess; }
    cudaError_t lastError() const noexcept { return err_; }

    const DecodeYoloV26Config& config() const noexcept { return cfg_; }

    // -------------------------------------------------------------------------
    //! Hot path: issues exactly one kernel.
    //!   y    : batch * (4+nc+extra) * anchors
    //!   cand : batch * anchors * (6+extra)
    //! No allocation, no copy, no sync; errors are recorded (see lastError()).
    // -------------------------------------------------------------------------
    void launch(cudaStream_t stream, int batch,
                const float* y, float* cand) const;

private:
    DecodeYoloV26Config cfg_;
    mutable cudaError_t err_ = cudaSuccess;
};

// =============================================================================
//  decodeYoloV26Device -- one-shot convenience version
// =============================================================================
void decodeYoloV26Device(cudaStream_t stream, int batch,
                        const float* y, float* cand,
                        const DecodeYoloV26Config& cfg);

}  // namespace trt_alpha::kernels::ops
