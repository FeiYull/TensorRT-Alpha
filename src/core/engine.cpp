// =============================================================================
//  trt_alpha :: core :: engine（实现）
// =============================================================================
#include "trt_alpha/core/engine.hpp"

#include "trt_alpha/core/logger.hpp"
#include "trt_alpha/core/paths.hpp"

#include <NvInferPlugin.h>

#include <filesystem>
#include <fstream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace fs = std::filesystem;

namespace trt_alpha::core {
namespace {

core::DataType trtToCore(nvinfer1::DataType trt) noexcept
{
    switch (trt)
    {
    case nvinfer1::DataType::kFLOAT: return DataType::Float32;
    case nvinfer1::DataType::kHALF:  return DataType::Float16;
    case nvinfer1::DataType::kINT8:  return DataType::Int8;
    case nvinfer1::DataType::kINT32: return DataType::Int32;
    case nvinfer1::DataType::kBOOL:  return DataType::Bool;
    case nvinfer1::DataType::kUINT8: return DataType::UInt8;
    case nvinfer1::DataType::kFP8:   return DataType::Float8_E4M3;
    case nvinfer1::DataType::kBF16:  return DataType::BFloat16;
    case nvinfer1::DataType::kINT64: return DataType::Int32;
#ifdef TRT_ALPHA_HAS_KINT4
    case nvinfer1::DataType::kINT4:  return DataType::Int8;
#endif
    }
    return DataType::Float32;
}

bool hasDynamicDim(const nvinfer1::Dims& dims) noexcept
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0) { return true; }
    }
    return false;
}

std::string dimsToString(const nvinfer1::Dims& dims)
{
    std::ostringstream oss;
    oss << "[";
    for (int i = 0; i < dims.nbDims; ++i)
    {
        oss << dims.d[i] << (i + 1 < dims.nbDims ? ", " : "");
    }
    oss << "]";
    return oss.str();
}

std::vector<std::uint8_t> readFile(const fs::path& path)
{
    std::ifstream in(path, std::ios::binary);
    if (!in.is_open())
    {
        throw std::runtime_error("cannot open file: " + path.string() +
                                 "\n  project root : " + Paths::toDisplay(Paths::root()));
    }
    in.seekg(0, std::ios::end);
    const std::streampos length = in.tellg();
    in.seekg(0, std::ios::beg);
    if (length <= 0)
    {
        return {};
    }
    std::vector<std::uint8_t> data(static_cast<std::size_t>(length));
    in.read(reinterpret_cast<char*>(data.data()), length);
    return data;
}

}  // namespace

std::size_t TensorDesc::volume() const noexcept
{
    std::size_t v = 1;
    for (int i = 0; i < shape.nbDims; ++i)
    {
        if (shape.d[i] > 0) { v *= static_cast<std::size_t>(shape.d[i]); }
    }
    return v;
}

// =============================================================================
//  Engine
// =============================================================================
Engine::Engine(const std::string& engineFile)
{
    const fs::path file = Paths::resolve(engineFile);
    TRT_LOG_INFO("Engine: loading " << Paths::toDisplay(file));

    const std::vector<std::uint8_t> blob = readFile(file);
    if (blob.empty())
    {
        TRT_LOG_ERROR("Engine: engine file is empty: " << Paths::toDisplay(file));
        throw std::runtime_error("engine file is empty: " + Paths::toDisplay(file));
    }

    // 注册 TensorRT 官方插件（EfficientNMS_TRT 等）。幂等。
    initLibNvInferPlugins(&trtLogger().trtLogger(), "");

    std::unique_ptr<nvinfer1::IRuntime> runtime(
        nvinfer1::createInferRuntime(trtLogger().trtLogger()));
    if (runtime == nullptr)
    {
        TRT_LOG_ERROR("Engine: createInferRuntime returned nullptr");
        throw std::runtime_error("createInferRuntime() returned nullptr");
    }

    nvinfer1::ICudaEngine* raw = runtime->deserializeCudaEngine(blob.data(), blob.size());
    if (raw == nullptr)
    {
        TRT_LOG_ERROR("Engine: deserializeCudaEngine failed");
        throw std::runtime_error("deserializeCudaEngine() failed");
    }
    m_engine.reset(raw);

    discoverIo();
    TRT_LOG_INFO("Engine: loaded (" << m_io.size() << " io tensors)");
}

void Engine::discoverIo()
{
    m_io.clear();
    const int nbIo = m_engine->getNbIOTensors();
    m_io.reserve(static_cast<std::size_t>(nbIo));
    for (int i = 0; i < nbIo; ++i)
    {
        TensorDesc desc;
        desc.name = m_engine->getIOTensorName(i);
        desc.isInput = (m_engine->getTensorIOMode(desc.name.c_str()) ==
                        nvinfer1::TensorIOMode::kINPUT);
        desc.shape = m_engine->getTensorShape(desc.name.c_str());
        desc.dtype = trtToCore(m_engine->getTensorDataType(desc.name.c_str()));
        m_io.push_back(std::move(desc));

        const TensorDesc& t = m_io.back();
        TRT_LOG_INFO("Engine: io[" << i << "] "
                     << (t.isInput ? "input " : "output")
                     << " '" << t.name << "' "
                     << nameOf(t.dtype) << " " << dimsToString(t.shape));
    }
}

const TensorDesc* Engine::find(const std::string& name) const noexcept
{
    for (const auto& t : m_io)
    {
        if (t.name == name) { return &t; }
    }
    return nullptr;
}

// =============================================================================
//  Context
// =============================================================================
Context::Context(Engine& engine)
{
    m_engine = engine.get();
    if (m_engine == nullptr)
    {
        throw std::runtime_error("Context: engine is null");
    }
    m_context.reset(m_engine->createExecutionContext());
    if (m_context == nullptr)
    {
        throw std::runtime_error("Context: createExecutionContext() failed");
    }
}

void Context::setInputShape(const std::string& name, const nvinfer1::Dims& dims)
{
    const nvinfer1::Dims current = m_context->getTensorShape(name.c_str());
    if (!hasDynamicDim(current))
    {
        TRT_LOG_DEBUG("Context: setInputShape skipped for static tensor '" << name << "'");
        return;
    }
    if (!m_context->setInputShape(name.c_str(), dims))
    {
        TRT_LOG_ERROR("Context: setInputShape failed for '" << name
                      << "' target " << dimsToString(dims));
        throw std::runtime_error("setInputShape failed for tensor '" + name + "'");
    }
    TRT_LOG_INFO("Context: setInputShape '" << name << "' -> " << dimsToString(dims));
}

nvinfer1::Dims Context::contextShape(const std::string& name) const
{
    return m_context->getTensorShape(name.c_str());
}

// =============================================================================
//  TrtEngine（兼容壳）
// =============================================================================
TrtEngine::TrtEngine(const std::string& engineFile)
    : m_sharedEngine(std::make_shared<Engine>(engineFile))
    , m_context(std::make_unique<Context>(*m_sharedEngine))
{
}

TrtEngine::TrtEngine(std::shared_ptr<Engine> sharedEngine)
    : m_sharedEngine(std::move(sharedEngine))
{
    if (!m_sharedEngine)
    {
        throw std::runtime_error("TrtEngine: sharedEngine is null");
    }
    m_context = std::make_unique<Context>(*m_sharedEngine);
}

void TrtEngine::buildFromOnnx(const std::string& /*onnxFile*/,
                              const std::string& /*engineFile*/,
                              const BuildOptions& /*options*/)
{
    TRT_LOG_ERROR("TrtEngine::buildFromOnnx not implemented yet");
    throw std::logic_error("TrtEngine::buildFromOnnx not implemented yet "
                          "(will move to builder module)");
}

}  // namespace trt_alpha::core