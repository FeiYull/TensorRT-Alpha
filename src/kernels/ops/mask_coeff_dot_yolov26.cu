// =============================================================================
//  trt_alpha :: kernels :: ops :: mask_coeff_dot_yolov26  (implementation)
// -----------------------------------------------------------------------------
//  Flat 1D kernel: one thread per prototype pixel. The prototypes are stored
//  channel-major ([k][Y][X]), so the k-th read of a thread is strided by
//  protoH*protoW while the X range inside a warp stays contiguous -- coalesced.
//
//  The coefficients are re-read from global memory by every thread (32 floats,
//  not worth staging). Putting them in shared memory would need a syncthreads
//  for data that the L1 already serves; not worth it for 32 values.
// =============================================================================
#include "trt_alpha/kernels/ops/mask_coeff_dot_yolov26.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/kernels/common.cuh"

#include <cuda_runtime.h>

#include <cstddef>

namespace trt_alpha::kernels::ops {
namespace {

__global__ void maskCoeffDotKernel(
    const float* __restrict__ coeffs,
    const float* __restrict__ protos,
    float* __restrict__ out,
    int numCoeffs,
    int elems)
{
    const int idx = (int)blockIdx.x * (int)blockDim.x + (int)threadIdx.x;
    if (idx >= elems)
    {
        return;
    }

    float acc = 0.f;
    for (int k = 0; k < numCoeffs; ++k)
    {
        acc += coeffs[k] * protos[static_cast<std::size_t>(k) * elems + idx];
    }
    out[idx] = acc;
}

}  // namespace

// =============================================================================
//  CvMaskCoeffDotPlan
// =============================================================================
CvMaskCoeffDotPlan::CvMaskCoeffDotPlan(int numCoeffs, int protoH, int protoW)
    : numCoeffs_(numCoeffs), protoH_(protoH), protoW_(protoW)
{
    if (numCoeffs <= 0 || protoH <= 0 || protoW <= 0)
    {
        TRT_LOG_ERROR("CvMaskCoeffDotPlan: bad geometry (numCoeffs=" << numCoeffs
                      << ", proto=" << protoW << "x" << protoH << ")");
        err_ = cudaErrorInvalidValue;
        return;
    }
    err_ = cudaSuccess;
}

void CvMaskCoeffDotPlan::launch(cudaStream_t stream,
                                const float* coeffs, const float* protos,
                                float* out) const
{
    if (err_ != cudaSuccess)
    {
        return;
    }
    if (coeffs == nullptr || protos == nullptr || out == nullptr)
    {
        err_ = cudaErrorInvalidValue;
        return;
    }

    const int elems = protoH_ * protoW_;
    const int grid  = (elems + kBlock1D - 1) / kBlock1D;

    maskCoeffDotKernel<<<grid, kBlock1D, 0, stream>>>(coeffs, protos, out,
                                                      numCoeffs_, elems);

    err_ = cudaGetLastError();
    if (err_ != cudaSuccess)
    {
        TRT_LOG_ERROR("CvMaskCoeffDotPlan: kernel launch failed: "
                      << cudaGetErrorString(err_));
    }
}

void maskCoeffDotDevice(cudaStream_t stream,
                        const float* coeffs, const float* protos, float* out,
                        int numCoeffs, int protoH, int protoW)
{
    CvMaskCoeffDotPlan plan(numCoeffs, protoH, protoW);
    plan.launch(stream, coeffs, protos, out);
    if (!plan.ok())
    {
        TRT_LOG_ERROR("maskCoeffDotDevice: " << cudaGetErrorString(plan.lastError()));
    }
}

}  // namespace trt_alpha::kernels::ops
