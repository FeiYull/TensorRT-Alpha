// =============================================================================
//  trt_alpha :: datasource :: i_data_source
// -----------------------------------------------------------------------------
//  IDataSource -- the data-source abstraction.
//
//  Responsibilities:
//    * read frames from an image / video / camera
//    * accumulate a full batch and produce a core::Batch
//    * when a batch is not full: zero-fill the extra frames and mark the valid
//      count in validCount
//
//  Not responsible for:
//    * inference / rendering / preprocessing
//    * dropping frames on its own (the caller manages that)
//
//  Lifetime:
//    * construction = open the resource (image / video / camera)
//    * next() = read one batch
//    * requestStop() = ask to stop (thread-safe)
//    * destruction = release the resource
//
//  Threading model:
//    * each IDataSource instance is called by [one data-source thread]
//    * requestStop() may be called from other threads (thread-safe)
//
//  Error handling:
//    * construction failure (missing file / camera that will not open) -> throw
//    * next() read failure -> return false (end of stream) or throw (unexpected
//      error)
// =============================================================================
#pragma once

#include "trt_alpha/core/batch.hpp"

namespace trt_alpha::datasource {

class IDataSource
{
public:
    virtual ~IDataSource() = default;

    IDataSource(const IDataSource&) = delete;
    IDataSource& operator=(const IDataSource&) = delete;

    //! Read the next batch. Returning false means "there is no more" (the file
    //! is exhausted, or a stop was requested).
    //! Output parameter out:
    //!   * out.buffer is contiguous memory, sized batchSize x H x W x C
    //!   * out.views[i] points at block i of that buffer
    //!   * out.validCount marks the number of valid frames
    //!   * a partial batch: the last (batchSize - validCount) frames are zero-filled
    [[nodiscard]] virtual bool next(core::Batch& out) = 0;

    //! Request a stop (thread-safe; callable from any thread).
    //! Once called, a blocking next() should return false as soon as possible.
    virtual void requestStop() = 0;

    //! Data-source type name (for logs / debugging).
    [[nodiscard]] virtual const char* typeName() const noexcept = 0;

protected:
    IDataSource() = default;
};

}  // namespace trt_alpha::datasource
