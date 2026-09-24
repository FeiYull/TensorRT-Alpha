// =============================================================================
//  trt_alpha :: core :: data_type（实现）
// =============================================================================
#include "trt_alpha/core/data_type.hpp"

namespace trt_alpha::core {

const char* nameOf(DataType dt) noexcept
{
    switch (dt)
    {
    case DataType::Int8:        return "int8";
    case DataType::UInt8:       return "uint8";
    case DataType::Int16:       return "int16";
    case DataType::UInt16:      return "uint16";
    case DataType::Int32:       return "int32";
    case DataType::UInt32:      return "uint32";
    case DataType::Float16:     return "fp16";
    case DataType::BFloat16:    return "bf16";
    case DataType::Float32:     return "fp32";
    case DataType::Float64:     return "fp64";
    case DataType::Float8_E4M3: return "fp8_e4m3";
    case DataType::Float8_E5M2: return "fp8_e5m2";
    case DataType::Bool:        return "bool";
    }
    return "unknown";
}

}  // namespace trt_alpha::core