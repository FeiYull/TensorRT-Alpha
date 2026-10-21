// =============================================================================
//  trt_alpha :: core :: batch_stage
// -----------------------------------------------------------------------------
//  stageBatchPixels -- put a batch's pixels where the pre-process kernels can
//  read them, and return the pointer to use.
//
//  Why a helper instead of three lines in every model:
//    Every model used to do "grow my scratch, cudaMemcpy the whole batch up,
//    synchronize" inside its own setBatch(). That hard-codes the assumption that
//    the source only ever produces host memory. A GPU decoder breaks exactly that
//    assumption -- the pixels are already on the card, and re-uploading them is
//    the single largest per-frame cost at 1080p (6.55 MB / 0.573 ms measured).
//    One function keeps the "already on the device" rule in one place, so a model
//    cannot silently get it wrong.
//
//  Contract:
//    * batch.deviceBuffer != nullptr -> zero copy; the model's own scratch is
//      left untouched (so a run that switches backends does not thrash it)
//    * otherwise -> upload batch.buffer into `staging`, growing it if needed
//    * The returned pointer is valid for as long as the batch (or `staging`)
//      lives; it is meant to be used immediately by preprocess().
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"
#include "trt_alpha/core/device_buffer.hpp"

#include <cuda_runtime.h>

namespace trt_alpha::core {

//! Ensure the batch's pixels are readable from device memory.
//!
//! `staging` is the caller's own device scratch (only used on the upload path).
//! `who` tags the messages (usually the model's m_tag).
//! Throws when the batch is malformed (no pixels, non-tight stride, or a device
//! buffer that is smaller than the batch it claims to hold).
[[nodiscard]] const void* stageBatchPixels(const Batch& batch,
                                           DeviceBuffer& staging,
                                           cudaStream_t stream,
                                           const char* who);

}  // namespace trt_alpha::core
