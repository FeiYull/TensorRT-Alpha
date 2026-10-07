// =============================================================================
//  trt_alpha :: renderer :: opencv_renderer
// -----------------------------------------------------------------------------
//  OpenCVRenderer —— IRenderer 的 OpenCV 实现。
//
//  渲染风格（内部固定，不暴露给用户）：
//    * 框线宽 2，字体 FONT_HERSHEY_DUPLEX，字号 0.5
//    * 按 label 循环取色（同一 label 恒定同色）
//    * 掩码按像素级混合（alpha = 0.35）
//
//  安全约定：
//    * 一切 ROI 访问先裁剪到图像范围内（防越界）
//    * 掩码类型不符时跳过而非崩溃
//    * save() 拒绝对"输入源文件"的写入 —— 输出路径与输入同一文件 = 覆盖原图
// =============================================================================
#pragma once

#include "trt_alpha/renderer/i_renderer.hpp"

#include <string>
#include <unordered_set>

namespace trt_alpha::renderer {

class OpenCVRenderer final : public IRenderer
{
public:
    OpenCVRenderer() = default;
    ~OpenCVRenderer() override = default;

    //! 登记一个输入源（文件或目录，绝对路径）。目录会被展开为其中的文件。
    //! save() 命中这些路径时拒绝写盘 —— 这是"输出 == 输入"的最后一道闸门，
    //! 任何数据源、任何调用方式（CLI / Infer）都躲不过。
    //! 必须在 start() 之前调用（save() 只读它，故 save 可保持 const）。
    void addInputSource(const std::string& resolvedPath);

    [[nodiscard]] bool protectsInputs() const noexcept { return !m_inputPaths.empty(); }

    void drawResult(core::BatchResult& result,
                    const std::vector<core::ClassInfo>& classNames) const override;

    void save(const core::BatchResult& result,
              const std::string& outputDir) const override;

    void show(const core::BatchResult& result,
              const std::string& windowName) const override;

private:
    //! 归一化后的输入文件路径集合（见 cpp 里的 pathKey）。
    std::unordered_set<std::string> m_inputPaths;
};

}  // namespace trt_alpha::renderer