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
// =============================================================================
#pragma once

#include "trt_alpha/renderer/i_renderer.hpp"

namespace trt_alpha::renderer {

class OpenCVRenderer final : public IRenderer
{
public:
    OpenCVRenderer() = default;
    ~OpenCVRenderer() override = default;

    void drawResult(core::BatchResult& result,
                    const std::vector<core::ClassInfo>& classNames) const override;

    void save(const core::BatchResult& result,
              const std::string& outputDir,
              const std::string& prefix = "result_") const override;

    void show(const core::BatchResult& result,
              const std::string& windowName) const override;
};

}  // namespace trt_alpha::renderer