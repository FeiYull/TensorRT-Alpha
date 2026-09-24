// =============================================================================
//  trt_alpha :: core :: buffer（实现）
// =============================================================================
#include "trt_alpha/core/buffer.hpp"

#include "trt_alpha/core/logger.hpp"

#include <cuda_runtime.h>

#include <cstring>
#include <stdexcept>
#include <string>

namespace trt_alpha::core {
namespace {

void validateSize(int width, int height, int channels)
{
    if (width <= 0 || height <= 0 || channels <= 0)
    {
        throw std::runtime_error(
            "Buffer: invalid size (width=" + std::to_string(width) +
            ", height=" + std::to_string(height) +
            ", channels=" + std::to_string(channels) + ")");
    }
}

int computeStride(int width, int channels, DataType dtype) noexcept
{
    return width * channels * static_cast<int>(sizeOf(dtype));
}

void checkCudaMalloc(cudaError_t err, std::size_t bytes)
{
    if (err != cudaSuccess)
    {
        TRT_LOG_ERROR("Buffer: cudaMalloc failed for " << bytes
                      << " bytes: " << cudaGetErrorString(err));
        throw std::runtime_error(
            "Buffer: cudaMalloc failed for " + std::to_string(bytes) +
            " bytes: " + cudaGetErrorString(err));
    }
}

}  // namespace

Buffer::Buffer(std::uint8_t* data, int width, int height, int channels,
               int stride, DataType dtype, MemorySpace space) noexcept
    : m_data(data)
    , m_width(width)
    , m_height(height)
    , m_channels(channels)
    , m_stride(stride)
    , m_dtype(dtype)
    , m_space(space)
{
}

Buffer::~Buffer()
{
    release();
}

void Buffer::release() noexcept
{
    if (m_data == nullptr)
    {
        return;
    }
    if (m_space == MemorySpace::Device)
    {
        cudaFree(m_data);
    }
    else
    {
        delete[] m_data;
    }
    m_data = nullptr;
}

std::shared_ptr<Buffer> Buffer::createHost(int width, int height, int channels,
                                           DataType dtype)
{
    validateSize(width, height, channels);
    const int stride = computeStride(width, channels, dtype);
    const std::size_t bytes = static_cast<std::size_t>(stride) *
                              static_cast<std::size_t>(height);

    std::uint8_t* data = new (std::nothrow) std::uint8_t[bytes];
    if (data == nullptr)
    {
        TRT_LOG_ERROR("Buffer::createHost: allocation failed for " << bytes
                      << " bytes (Host)");
        throw std::runtime_error("Buffer::createHost: allocation failed for " +
                                 std::to_string(bytes) + " bytes");
    }

    TRT_LOG_DEBUG("Buffer: createHost " << width << "x" << height
                  << "x" << channels << " " << nameOf(dtype)
                  << " (" << bytes << " bytes)");

    return std::shared_ptr<Buffer>(
        new Buffer(data, width, height, channels, stride, dtype, MemorySpace::Host));
}

std::shared_ptr<Buffer> Buffer::createDevice(int width, int height, int channels,
                                             DataType dtype)
{
    validateSize(width, height, channels);
    const int stride = computeStride(width, channels, dtype);
    const std::size_t bytes = static_cast<std::size_t>(stride) *
                              static_cast<std::size_t>(height);

    void* ptr = nullptr;
    checkCudaMalloc(cudaMalloc(&ptr, bytes), bytes);

    TRT_LOG_DEBUG("Buffer: createDevice " << width << "x" << height
                  << "x" << channels << " " << nameOf(dtype)
                  << " (" << bytes << " bytes)");

    return std::shared_ptr<Buffer>(
        new Buffer(static_cast<std::uint8_t*>(ptr), width, height, channels, stride,
                   dtype, MemorySpace::Device));
}

std::shared_ptr<Buffer> Buffer::fromHostData(const std::uint8_t* srcData,
                                             int width, int height, int channels,
                                             int srcStride, DataType dtype)
{
    validateSize(width, height, channels);
    if (srcData == nullptr)
    {
        TRT_LOG_ERROR("Buffer::fromHostData: srcData is nullptr");
        throw std::runtime_error("Buffer::fromHostData: srcData is nullptr");
    }

    const int dstStride = computeStride(width, channels, dtype);
    if (srcStride < dstStride)
    {
        TRT_LOG_ERROR("Buffer::fromHostData: srcStride (" << srcStride
                      << ") < tight stride (" << dstStride << ")");
        throw std::runtime_error(
            "Buffer::fromHostData: srcStride (" + std::to_string(srcStride) +
            ") < tight stride (" + std::to_string(dstStride) + ")");
    }

    const std::size_t dstBytes = static_cast<std::size_t>(dstStride) *
                                 static_cast<std::size_t>(height);

    std::uint8_t* data = new (std::nothrow) std::uint8_t[dstBytes];
    if (data == nullptr)
    {
        TRT_LOG_ERROR("Buffer::fromHostData: allocation failed for " << dstBytes
                      << " bytes");
        throw std::runtime_error("Buffer::fromHostData: allocation failed for " +
                                 std::to_string(dstBytes) + " bytes");
    }

    for (int y = 0; y < height; ++y)
    {
        std::memcpy(data + static_cast<std::size_t>(y) * dstStride,
                    srcData + static_cast<std::size_t>(y) * srcStride,
                    static_cast<std::size_t>(dstStride));
    }

    TRT_LOG_DEBUG("Buffer: fromHostData " << width << "x" << height
                  << "x" << channels << " " << nameOf(dtype)
                  << " (srcStride=" << srcStride << ", dstStride=" << dstStride << ")");

    return std::shared_ptr<Buffer>(
        new Buffer(data, width, height, channels, dstStride, dtype, MemorySpace::Host));
}

BufferView Buffer::view() const noexcept
{
    BufferView v;
    v.data = m_data;
    v.width = m_width;
    v.height = m_height;
    v.stride = m_stride;
    v.channels = m_channels;
    v.dtype = m_dtype;
    v.space = m_space;
    return v;
}

}  // namespace trt_alpha::core