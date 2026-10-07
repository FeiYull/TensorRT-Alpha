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

core::DataType trtToCore(nvinfer1::DataType trt, const std::string& tensorName)
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
    default: break;
    }
    // 绝不静默降级：kINT64 / kINT4 等无法用 core::DataType 表示，强行降级会按错误
    // 位宽分配显存 → 越界（且全程无报错）。这里直接失败。
    throw std::runtime_error(
        "unsupported tensor data type (nvinfer1::DataType value " +
        std::to_string(static_cast<int>(trt)) + ") for tensor '" + tensorName + "'");
}

bool hasDynamicDim(const nvinfer1::Dims& dims) noexcept
{
    for (int i = 0; i < dims.nbDims; ++i)
    {
        if (dims.d[i] < 0) { return true; }
    }
    return false;
}

bool dimsEqual(const nvinfer1::Dims& a, const nvinfer1::Dims& b) noexcept
{
    if (a.nbDims != b.nbDims) { return false; }
    for (int i = 0; i < a.nbDims; ++i)
    {
        if (a.d[i] != b.d[i]) { return false; }
    }
    return true;
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

int TensorDesc::pick(const nvinfer1::Dims& d, int axis, int fallback) const noexcept
{
    if (axis >= 0 && axis < d.nbDims && d.d[axis] > 0) { return d.d[axis]; }
    return fallback;
}

BatchRange TensorDesc::batchRange(int axis) const noexcept
{
    if (axis < 0 || axis >= shape.nbDims)
    {
        return BatchRange{1, 1, 1};   // 无 batch 轴：batch 恒为 1
    }
    // 静态引擎（该轴无动态维）：shape.d[axis] > 0，作为固定值兜底
    const int fixed = (shape.d[axis] > 0) ? shape.d[axis] : 1;
    return BatchRange{ pick(minShape, axis, fixed),
                       pick(optShape, axis, fixed),
                       pick(maxShape, axis, fixed) };
}

ResolvedBatch resolveBatch(const TensorDesc& input, int requested,
                           const std::string& who, int declaredMax, int batchAxis)
{
    ResolvedBatch r;
    const BatchRange br = input.batchRange(batchAxis);
    r.min = br.min;
    r.opt = br.opt;
    r.max = br.max;
    r.isDynamic = input.isDynamicBatch(batchAxis);

    // 可选：配置声明的上界契约，必须与引擎 profile max 一致
    if (declaredMax > 0 && declaredMax != br.max)
    {
        TRT_LOG_ERROR(who << ": declared max_batch_size " << declaredMax
                      << " != engine batch max " << br.max);
        throw std::runtime_error(
            who + ": declared max_batch_size " + std::to_string(declaredMax) +
            " does not match engine batch max " + std::to_string(br.max) +
            (r.isDynamic ? " (engine range [" + std::to_string(br.min) + ", " +
                               std::to_string(br.max) + "])"
                         : " (engine batch is static)")
            );
    }

    // 引擎侧的合法值描述 —— 每条报错都必须带上它：
    //   静态引擎 → 明确"要求多少"；动态引擎 → 明确"区间 + 最大是多少"。
    const std::string limit =
        r.isDynamic
            ? "engine batch range [" + std::to_string(br.min) + ", " +
                  std::to_string(br.max) + "], max batch " + std::to_string(br.max)
            : "engine batch is fixed at " + std::to_string(br.max);
    const std::string hint =
        r.isDynamic
            ? "use --batch within that range, or re-export the engine with a"
              " larger --maxShapes to raise the limit"
            : "set [input] batch_size = " + std::to_string(br.max) +
              ", or re-export the engine with a dynamic batch";

    // 所有拒绝路径共用一份措辞：ERROR 日志 + 抛异常（调用方一路向上，最终退出码 1）。
    const auto reject = [&](const std::string& why) {
        TRT_LOG_ERROR(who << ": input '" << input.name << "': " << why
                      << "; " << limit);
        throw std::runtime_error(who + ": input '" + input.name + "': " + why +
                                 "; " + limit + "; " + hint);
    };

    // 非法请求值：ini 写 0 / 负数（CLI 侧由 options 层拦，ini 侧没有前置校验，
    // 所以这里必须兜住）—— 报错仍按引擎能力给出该填多少。
    if (requested <= 0)
    {
        reject("invalid batch " + std::to_string(requested) + " (batch must be > 0)");
    }

    // 布局里没有 batch 轴（如 CHW / HWC）：batch 概念上恒为 1。
    // 请求别的值就是"配置与引擎能力不符" → 显式失败（这里以前会去读轴 0 的
    // 通道数当 batch，报出 "requested batch 1 does not match the static engine"
    // 这种完全指不到原因的错）。
    if (batchAxis < 0)
    {
        if (requested != 1)
        {
            reject("requested batch " + std::to_string(requested) +
                   " but the input has no batch axis (layout declares no 'N')");
        }
        r.batch = 1;
        return r;
    }

    if (!r.isDynamic)
    {
        // 静态引擎：batch 由引擎（onnx 导出时）写死，请求值无法生效。
        // 口径与动态越界一致 —— 配置与引擎能力不符就是错误，显式失败；
        // 静默纠正会让人以为配置生效，把 ini / CLI 里写错的值掩盖掉。
        if (requested != br.max)
        {
            reject("requested batch " + std::to_string(requested) +
                   " does not match the static engine");
        }
        r.batch = br.max;
        return r;
    }

    // 动态引擎：batch 必须落在 [min, max] 内
    if (requested < br.min)
    {
        reject("requested batch " + std::to_string(requested) +
               " is below engine min " + std::to_string(br.min));
    }
    if (requested > br.max)
    {
        reject("requested batch " + std::to_string(requested) +
               " exceeds engine max " + std::to_string(br.max));
    }
    r.batch = requested;
    return r;
}

void validateInputTensor(const TensorDesc& input, const Layout& layout,
                         int channels, const std::string& who)
{
    if (layout.empty())
    {
        throw std::runtime_error(
            who + ": input '" + input.name +
            "' needs an explicit layout (e.g. NCHW / NHWC / NCDHW); none was given");
    }

    // ① 秩必须与布局一致
    if (input.shape.nbDims != layout.rank())
    {
        throw std::runtime_error(
            who + ": input '" + input.name + "' rank " +
            std::to_string(input.shape.nbDims) + " != layout '" + layout.str() +
            "' rank " + std::to_string(layout.rank()));
    }

    // ② 物理格式必须是线性（分块 / 向量化排布本框架不认，绝不静默喂错内存）
    if (input.format != nvinfer1::TensorFormat::kLINEAR)
    {
        throw std::runtime_error(
            who + ": input '" + input.name + "' uses non-linear tensor format '" +
            (input.formatDesc.empty() ? std::to_string(static_cast<int>(input.format))
                                      : input.formatDesc) +
            "'; this framework feeds plain linear buffers. Rebuild the engine with a "
            "linear I/O format (trtexec --inputIOFormats=<type>:chw)");
    }

    // ③ 通道轴校验：布局声明与引擎不符时立刻报错，而不是拿错轴当 H/W
    if (channels > 0 && layout.has(Layout::kChannel))
    {
        const int cIdx = layout.indexOf(Layout::kChannel);
        const int dC   = input.shape.d[cIdx];
        if (dC > 0 && dC != channels)
        {
            throw std::runtime_error(
                who + ": input '" + input.name + "' layout '" + layout.str() +
                "' puts channel at index " + std::to_string(cIdx) +
                " where the engine declares size " + std::to_string(dC) +
                " (expected " + std::to_string(channels) +
                "); layout declaration does not match the engine");
        }
    }
}

void resolveInputShape(const TensorDesc& input, const Layout& layout, int channels,
                       const ResolvedInputShape& intent, const std::string& who,
                       ResolvedInputShape& out)
{
    validateInputTensor(input, layout, channels, who);

    ResolvedInputShape r;

    // 逐轴解析：静态维以引擎为唯一真相源；动态维采用调用方意图值。
    const auto axis = [&](char letter, int wanted) -> int
    {
        const int idx = layout.indexOf(letter);
        if (idx < 0) { return 0; }
        const int declared = input.shape.d[idx];
        if (declared > 0) { return declared; }
        r.dynamic = true;
        return wanted > 0 ? wanted : 0;
    };

    r.depth  = axis(Layout::kDepth,  intent.depth);
    r.height = axis(Layout::kHeight, intent.height);
    r.width  = axis(Layout::kWidth,  intent.width);

    // 只比较"布局里真实存在、且调用方给了意图值"的轴
    r.corrected = (layout.has(Layout::kDepth)  && intent.depth  > 0 && r.depth  != intent.depth) ||
                  (layout.has(Layout::kHeight) && intent.height > 0 && r.height != intent.height) ||
                  (layout.has(Layout::kWidth)  && intent.width  > 0 && r.width  != intent.width);

    if (r.corrected)
    {
        TRT_LOG_WARN(who << ": config size " << intent.width << "x" << intent.height
                     << " != engine declared shape, using engine shape ("
                     << r.width << "x" << r.height << ")");
    }

    out = r;
}

void applyInputShape(TrtEngine& engine, const std::string& tensorName,
                     const Layout& modelLayout, int channels, ModelConfig& cfg)
{
    const TensorDesc* input = engine.find(tensorName);
    if (input == nullptr)
    {
        throw std::runtime_error("applyInputShape: input tensor '" + tensorName +
                                 "' not found in engine");
    }

    // batch 与空间维同口径：引擎 profile 是唯一真相源，配置不符一律抛（不静默改值）。
    // 落定之后再构造 dims，N 轴就用这个已过校验的值。
    // 放在这里 = 所有走 applyInputShape 的模型（13 个）+ bench + sample + 单测
    // 共用同一道护栏，不必各写一遍（yunet 因 H/W 取自原图不走这里，自带一行，见 yunet.cpp）。
    //
    // 布局优先级：INI 的 input.layout（覆盖）> 模型规范布局（默认）
    const Layout& layout = cfg.layout.empty() ? modelLayout : cfg.layout;

    // batch 轴由【布局】决定，不硬取轴 0 —— 否则 CHW / HWCN 这类布局会读错轴。
    // 布局声明里没有 N 轴时 batchAxis == -1，resolveBatch 按"batch 恒为 1"处理。
    const int batchAxis = input->batchAxisIndex(layout);
    cfg.batchSize = resolveBatch(*input, cfg.batchSize, "InputShape",
                                 cfg.maxBatchSize, batchAxis).batch;

    ResolvedInputShape intent;
    intent.height = cfg.dstH;
    intent.width  = cfg.dstW;

    ResolvedInputShape resolved;
    resolveInputShape(*input, layout, channels, intent, "InputShape", resolved);

    // 按 layout 逐轴构造目标形状 → 天然支持 3~8 维的任意排列
    nvinfer1::Dims dims{};
    dims.nbDims = layout.rank();
    for (int i = 0; i < layout.rank(); ++i)
    {
        const char a = layout.at(i);
        int v = 0;
        switch (a)
        {
        case Layout::kBatch:   v = cfg.batchSize; break;
        case Layout::kChannel: v = channels;      break;
        case Layout::kDepth:   v = resolved.depth;  break;
        case Layout::kHeight:  v = resolved.height; break;
        case Layout::kWidth:   v = resolved.width;  break;
        default:               v = input->shape.d[i]; break;   // T / E / ? → 引擎声明值
        }
        if (v <= 0)
        {
            throw std::runtime_error(
                "applyInputShape: cannot determine size of axis '" + std::string(1, a) +
                "' (index " + std::to_string(i) + ") for input '" + tensorName +
                "'; declare it via layout / config, or use a static engine dimension");
        }
        dims.d[i] = v;
    }

    engine.setInputShape(tensorName, dims);

    // 写回解析结果，供 letterbox / 后处理 / 显存分配使用
    if (resolved.height > 0) { cfg.dstH = resolved.height; }
    if (resolved.width  > 0) { cfg.dstW = resolved.width;  }

    TRT_LOG_INFO("InputShape: '" << tensorName << "' layout=" << layout.str()
                 << " -> " << dimsToString(dims));
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
    const bool hasProfile = (m_engine->getNbOptimizationProfiles() > 0);
    m_io.reserve(static_cast<std::size_t>(nbIo));
    for (int i = 0; i < nbIo; ++i)
    {
        TensorDesc desc;
        desc.name = m_engine->getIOTensorName(i);
        desc.isInput = (m_engine->getTensorIOMode(desc.name.c_str()) ==
                        nvinfer1::TensorIOMode::kINPUT);
        desc.shape = m_engine->getTensorShape(desc.name.c_str());
        desc.dtype  = trtToCore(m_engine->getTensorDataType(desc.name.c_str()), desc.name);
        desc.format = m_engine->getTensorFormat(desc.name.c_str());
        if (const char* fmtDesc = m_engine->getTensorFormatDesc(desc.name.c_str()))
        {
            desc.formatDesc = fmtDesc;
        }

        // 输入张量：填 profile 形状（静态引擎三段相同；动态引擎为 min/opt/max）
        if (desc.isInput && hasProfile)
        {
            desc.minShape = m_engine->getProfileShape(
                desc.name.c_str(), 0, nvinfer1::OptProfileSelector::kMIN);
            desc.optShape = m_engine->getProfileShape(
                desc.name.c_str(), 0, nvinfer1::OptProfileSelector::kOPT);
            desc.maxShape = m_engine->getProfileShape(
                desc.name.c_str(), 0, nvinfer1::OptProfileSelector::kMAX);
        }
        m_io.push_back(std::move(desc));

        const TensorDesc& t = m_io.back();
        TRT_LOG_INFO("Engine: io[" << i << "] "
                     << (t.isInput ? "input " : "output")
                     << " '" << t.name << "' "
                     << nameOf(t.dtype) << " " << dimsToString(t.shape)
                     << (t.format == nvinfer1::TensorFormat::kLINEAR
                             ? std::string()
                             : (" fmt=" + t.formatDesc)));

        if (t.isInput)
        {
            // 诊断日志：此处还不知道模型规范布局，按约定"N 在轴 0"展示。
            // 权威判定在 applyInputShape（它拿得到布局），这里仅作参考。
            const BatchRange br = t.batchRange(0);
            TRT_LOG_INFO("Engine: io[" << i << "] input batch "
                         << (t.isDynamicBatch(0) ? "dynamic" : "static")
                         << ", min/opt/max = " << br.min << "/" << br.opt
                         << "/" << br.max);
        }
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
    // 判据必须是【引擎声明形状】（永远含 -1），不能用 context 当前形状：
    // context 形状在 setInputShape 后会被具体化（-1 → 实际值），
    // 用它判断的话，动态引擎第二次换 batch 时会被误判成"静态"而抛错。
    const nvinfer1::Dims declared = m_engine->getTensorShape(name.c_str());
    if (!hasDynamicDim(declared))
    {
        // 静态张量：形状由引擎（onnx 导出时）写死，无法改变。
        // 请求形状与引擎固定形状不一致 = 配置与引擎不匹配 → 直接报错，
        // 杜绝"按错误 batch 分配显存 → 静默越界"。
        if (!dimsEqual(declared, dims))
        {
            TRT_LOG_ERROR("Context: static tensor '" << name << "' is fixed at "
                          << dimsToString(declared) << " but requested "
                          << dimsToString(dims));
            throw std::runtime_error(
                "tensor '" + name + "' is static (fixed at " + dimsToString(declared) +
                ") but requested " + dimsToString(dims) +
                "; engine/onnx batch or input size mismatch");
        }
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