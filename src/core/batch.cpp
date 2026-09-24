// =============================================================================
//  trt_alpha :: core :: batch（实现）
// =============================================================================
#include "trt_alpha/core/batch.hpp"

#include "trt_alpha/core/logger.hpp"

namespace trt_alpha::core {

bool Batch::validate(std::string* errorMsg) const
{
    auto fail = [errorMsg](const char* msg) -> bool {
        if (errorMsg != nullptr) { *errorMsg = msg; }
        TRT_LOG_DEBUG("Batch::validate failed: " << msg);
        return false;
    };

    if (buffer == nullptr)               return fail("Batch::validate: buffer is nullptr");
    if (views.empty())                   return fail("Batch::validate: views is empty");
    if (validCount < 0 || validCount > static_cast<int>(views.size()))
        return fail("Batch::validate: validCount out of range");

    const BufferView& first = views[0];
    if (first.data == nullptr)           return fail("Batch::validate: views[0].data is nullptr");
    if (first.width <= 0 || first.height <= 0 || first.channels <= 0)
        return fail("Batch::validate: views[0] has invalid dimensions");

    for (std::size_t i = 1; i < views.size(); ++i)
    {
        const BufferView& v = views[i];
        if (v.data == nullptr)
            return fail("Batch::validate: some view has nullptr data");
        if (v.width != first.width || v.height != first.height ||
            v.channels != first.channels || v.dtype != first.dtype)
            return fail("Batch::validate: views have inconsistent layout");
    }
    return true;
}

}  // namespace trt_alpha::core