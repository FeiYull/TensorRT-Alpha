// =============================================================================
//  trt_alpha :: core :: engine
// -----------------------------------------------------------------------------
//  Engine  -- a shared ICudaEngine wrapper (thread-safe, referencable by
//             several Contexts).
//  Context -- an exclusive IExecutionContext wrapper (not thread-safe; one per
//             worker).
//  TrtEngine -- a compatibility shell: holds a shared_ptr<Engine> + an
//               exclusive Context.
//
//  1 engine + N context model:
//    * deserialize the engine once (one copy of the weights)
//    * one context per worker
//
//  Compatibility with older usage:
//    * TrtEngine(file) deserializes its own engine copy (independent, not shared)
// =============================================================================
#pragma once

#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/layout.hpp"
#include "trt_alpha/core/model_config.hpp"

#include <NvInfer.h>

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace trt_alpha::core {

struct BatchRange
{
    int min = 1;
    int opt = 1;
    int max = 1;
    [[nodiscard]] bool isDynamic() const noexcept { return min != max; }
};

//! I/O tensor description.
struct TensorDesc
{
    std::string name;
    nvinfer1::Dims shape{};      // shape declared by the engine (dynamic axes are -1)
    nvinfer1::Dims minShape{};   // profile kMIN
    nvinfer1::Dims optShape{};   // profile kOPT
    nvinfer1::Dims maxShape{};   // profile kMAX
    DataType dtype = DataType::Float32;
    //! Physical memory layout (linear / blocked-vectorized). Orthogonal to
    //! shape's [logical order]: shape decides which axis is H/W/C, format decides
    //! how those values sit in memory.
    nvinfer1::TensorFormat format = nvinfer1::TensorFormat::kLINEAR;
    std::string formatDesc;      //!< human-readable format name (for logs / exception messages)
    bool isInput = false;

    [[nodiscard]] std::size_t volume() const noexcept;

    //! Index of the N axis (batch axis) within the layout; -1 when the layout
    //! declares no batch axis. With it, "which axis is batch" is decided by the
    //! [layout] instead of being hard-wired to axis 0 -- layouts such as CHW (no
    //! N axis) / HWCN then read the right axis.
    [[nodiscard]] int batchAxisIndex(const Layout& layout) const noexcept
    { return layout.has(Layout::kBatch) ? layout.indexOf(Layout::kBatch) : -1; }

    //! Whether the given axis is dynamic (-1). axis < 0 (no batch axis) is always false.
    [[nodiscard]] bool isDynamicBatch(int axis = 0) const noexcept
    { return axis >= 0 && axis < shape.nbDims && shape.d[axis] < 0; }

    //! The batch range of the given axis (reading min/opt/max from the profile).
    //! axis < 0 returns {1,1,1}.
    [[nodiscard]] BatchRange batchRange(int axis = 0) const noexcept;   // see the .cpp

private:
    [[nodiscard]] int pick(const nvinfer1::Dims& d, int axis, int fallback) const noexcept;
};

//! The result of resolving the engine input's batch (unified static / dynamic
//! semantics).
struct ResolvedBatch
{
    int  batch     = 1;      //!< the batch actually used (== the requested value; a mismatch throws instead of returning)
    bool isDynamic = false;  //!< whether the engine input is a dynamic batch
    int  min = 1;            //!< engine profile minimum batch
    int  opt = 1;            //!< engine profile optimal batch
    int  max = 1;            //!< engine profile maximum batch
};

//! Validate the batch the caller requested against the engine's real capability;
//! any mismatch throws std::runtime_error:
//!   * Static engine (fixed batch in the onnx): the batch is hard-wired by the
//!     engine, so the requested value must equal it; otherwise it throws
//!     (**no silent correction** -- correcting it would make people believe the
//!     ini / CLI value took effect).
//!   * Dynamic engine (onnx -1 plus trtexec min/opt/max): the batch must lie
//!     within [min, max]; a request > max or < min throws (**no silent
//!     clamping** -- a config exceeding the engine's capability is an error, and
//!     a silent downgrade would make people think the setting took effect while
//!     the data source may still pack by the original value).
//!   * Optional upper-bound contract: when declaredMax > 0 it must equal the
//!     engine profile's max, otherwise it throws std::runtime_error (the
//!     capability declared by the config disagrees with the engine).
//! In one line: the config only expresses "intent", the engine profile is the
//! single source of truth, and a conflict fails explicitly instead of silently
//! changing values.
//! @param input       engine input tensor description (profile shapes must be filled in)
//! @param requested   the batch the caller requested
//! @param who         caller name (for logs and exception messages)
//! @param declaredMax the upper-bound contract declared by the config; <= 0 = undeclared
//! @param batchAxis   batch axis index (decided by the layout, see
//!                    TensorDesc::batchAxisIndex); -1 means the layout has no
//!                    batch axis (e.g. CHW), in which case the batch is
//!                    conceptually always 1
[[nodiscard]] ResolvedBatch resolveBatch(const TensorDesc& input,
                                         int requested,
                                         const std::string& who,
                                         int declaredMax = 0,
                                         int batchAxis = 0);

//! The result of resolving the engine input's [spatial axes] (named
//! semantically, independent of the layout's ordering).
struct ResolvedInputShape
{
    int  depth     = 0;      //!< 0 when the layout has no D axis / the engine does not declare one
    int  height    = 0;
    int  width     = 0;
    bool dynamic   = false;  //!< a dynamic (-1) axis exists among the spatial axes (the caller's intent value is used)
    bool corrected = false;  //!< a static axis was corrected by the engine (differs from the intent value)
};

//! Validate the input tensor's [rank / channel axis / physical format]; any
//! mismatch throws std::runtime_error:
//!   * shape.nbDims != layout.rank()  -> the layout declaration disagrees with the engine
//!   * a static C-axis extent != channels -> the layout declaration is wrong
//!     (prevents mistaking H/W for channels and silently reading garbage)
//!   * format != kLINEAR             -> the engine demands blocked/vectorized
//!     layout while this framework only feeds linear buffers
//! These three are the "rather fail than take a wrong step" guards: an explicit
//! error beats silently using the wrong axis.
//! @param channels expected channel count; <= 0 skips the channel check
void validateInputTensor(const TensorDesc& input, const Layout& layout,
                         int channels, const std::string& who);

//! Resolve the input's spatial axes (D/H/W) using the [engine-declared shape] as
//! the single source of truth.
//!   * static axis (> 0): ignore the intent and use the engine's value; if it
//!     differs from the intent, corrected = true
//!   * dynamic axis (-1): use the intent (the caller's intended value); 0 when no
//!     intent is provided
//! The pre-checks are the same as validateInputTensor.
void resolveInputShape(const TensorDesc& input, const Layout& layout, int channels,
                       const ResolvedInputShape& intent, const std::string& who,
                       ResolvedInputShape& out);

//! A shared ICudaEngine (thread-safe, referencable by several Contexts).
class Engine
{
public:
    //! Load from a serialized engine file. Throws std::runtime_error on failure.
    explicit Engine(const std::string& engineFile);

    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;
    Engine(Engine&&) = delete;
    Engine& operator=(Engine&&) = delete;

    [[nodiscard]] nvinfer1::ICudaEngine* get() noexcept { return m_engine.get(); }
    [[nodiscard]] const nvinfer1::ICudaEngine* get() const noexcept { return m_engine.get(); }

    [[nodiscard]] const std::vector<TensorDesc>& ioTensors() const noexcept { return m_io; }
    [[nodiscard]] const TensorDesc* find(const std::string& name) const noexcept;

private:
    void discoverIo();

    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::vector<TensorDesc> m_io;
};

//! An exclusive IExecutionContext (not thread-safe; one per worker).
class Context
{
public:
    //! Create a context from a shared engine. The engine must outlive the Context.
    explicit Context(Engine& engine);

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;
    Context(Context&&) = delete;
    Context& operator=(Context&&) = delete;

    [[nodiscard]] nvinfer1::IExecutionContext* get() noexcept { return m_context.get(); }
    [[nodiscard]] const nvinfer1::IExecutionContext* get() const noexcept { return m_context.get(); }

    //! Set the actual input shape.
    //!  * Dynamic tensor: forwarded to TRT; an out-of-profile value is rejected by
    //!    TRT, which throws std::runtime_error.
    //!  * Static tensor: the shape is hard-wired by the engine; a matching request
    //!    is skipped silently, while a mismatching one throws std::runtime_error
    //!    (config / engine mismatch; never a silent out-of-range).
    void setInputShape(const std::string& name, const nvinfer1::Dims& dims);

    //! The actual shape at context level (after setInputShape has been applied).
    [[nodiscard]] nvinfer1::Dims contextShape(const std::string& name) const;

private:
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;
    nvinfer1::ICudaEngine* m_engine = nullptr;   // non-owning
};

//! Compatibility shell: holds a shared_ptr<Engine> + an exclusive Context.
class TrtEngine
{
public:
    struct BuildOptions
    {
        std::size_t workspaceBytes = 1ULL << 30;
        bool fp16 = true;
    };

    //! Legacy constructor: deserializes its own engine copy (independent, not shared).
    explicit TrtEngine(const std::string& engineFile);

    //! New constructor: reuse a shared engine (1 engine + N contexts).
    explicit TrtEngine(std::shared_ptr<Engine> sharedEngine);

    //! ONNX -> engine (TODO).
    static void buildFromOnnx(const std::string& onnxFile,
                              const std::string& engineFile,
                              const BuildOptions& options = {});

    TrtEngine(const TrtEngine&) = delete;
    TrtEngine& operator=(const TrtEngine&) = delete;

    [[nodiscard]] nvinfer1::ICudaEngine* engine() noexcept
    {
        return m_sharedEngine ? m_sharedEngine->get() : nullptr;
    }
    [[nodiscard]] nvinfer1::IExecutionContext* context() noexcept
    {
        return m_context ? m_context->get() : nullptr;
    }
    [[nodiscard]] const std::vector<TensorDesc>& ioTensors() const noexcept
    {
        static const std::vector<TensorDesc> kEmpty;
        return m_sharedEngine ? m_sharedEngine->ioTensors() : kEmpty;
    }
    [[nodiscard]] const TensorDesc* find(const std::string& name) const noexcept
    {
        return m_sharedEngine ? m_sharedEngine->find(name) : nullptr;
    }
    [[nodiscard]] const Engine* sharedEngine() const noexcept
    {
        return m_sharedEngine.get();
    }

    void setInputShape(const std::string& name, const nvinfer1::Dims& dims)
    {
        if (m_context) { m_context->setInputShape(name, dims); }
    }
    [[nodiscard]] nvinfer1::Dims contextShape(const std::string& name) const
    {
        return m_context ? m_context->contextShape(name) : nvinfer1::Dims{};
    }

private:
    std::shared_ptr<Engine> m_sharedEngine;
    std::unique_ptr<Context> m_context;
};

//! One-stop call: settle the input batch, resolve the input layout / spatial
//! axes and issue setInputShape, writing the result back into cfg.batchSize /
//! cfg.dstH / cfg.dstW in place.
//!   * batch: goes through resolveBatch first (the engine profile is the single
//!     source of truth; a mismatch throws)
//!   * layout priority: cfg.layout (the INI's input.layout) > modelLayout (the
//!     model's canonical layout)
//!   * the target shape is built axis by axis from the layout, so any permutation
//!     of 3 to 8 dimensions works (NCHW/NHWC/NCDHW/CHWN/...)
//!   * dst_h / dst_w in the config are downgraded to "intent values": they only
//!     take effect when the corresponding engine axis is dynamic (-1); static
//!     axes always follow the engine, with a WARN about the correction
//! A failure (no such input / a failed check / an undeterminable axis extent)
//! throws std::runtime_error.
void applyInputShape(TrtEngine& engine, const std::string& tensorName,
                     const Layout& modelLayout, int channels, ModelConfig& cfg);

}  // namespace trt_alpha::core
