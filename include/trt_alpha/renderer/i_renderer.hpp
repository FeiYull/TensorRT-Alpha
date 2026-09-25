// =============================================================================
//  trt_alpha :: renderer :: i_renderer
// -----------------------------------------------------------------------------
//  IRenderer —— 渲染抽象接口。
//
//  设计原则：
//    * 只认识结果结构体（BatchResult / Detection / Segmentation / ClassScore），
//      不认识具体模型
//    * 换后端（OpenCV → Qt / Skia / JSON）时新增实现即可，调用方 switch 指针
//    * 【不依赖 OpenCV】：接口本身不暴露 cv::Mat，只用 core 的类型
//    * 只做"画 / 存 / 显示"，不做"读图 / 推理 / 调度"
//
//  调用时序契约：
//    drawResult() → save() / show()
//    先画，再存 / 显。drawResult() 就地修改 result.views 指向的内存。
//
//  线程安全：
//    * 实现方负责自己的线程安全
//    * 通常一个渲染线程串行调用，不需要额外同步
// =============================================================================
#pragma once

#include "trt_alpha/core/batch_result.hpp"
#include "trt_alpha/core/class_info.hpp"

#include <string>
#include <vector>

namespace trt_alpha::renderer {

class IRenderer
{
public:
    virtual ~IRenderer() = default;

    IRenderer(const IRenderer&) = delete;
    IRenderer& operator=(const IRenderer&) = delete;

    //! 画整批结果（in-place，写回 result.views 指向的内存）。
    //! 遍历 result.views[0..validCount-1]，按 detections / segmentations /
    //! classifications 自动分发。
    //! classNames 是"类别名 + 颜色"的数组（来自 ModelConfig.classNames）。
    virtual void drawResult(core::BatchResult& result,
                            const std::vector<core::ClassInfo>& classNames) const = 0;

    //! 存盘：每张有效图存成 <outputDir>/<prefix><index>.jpg
    //! index = result.firstFrameIndex + i
    //! 自动创建 outputDir（如不存在）。
    virtual void save(const core::BatchResult& result,
                      const std::string& outputDir,
                      const std::string& prefix = "result_") const = 0;

    //! 显示：cv::imshow + cv::waitKey(1)（不阻塞），只显示第一张有效图。
    //! 实时场景"每帧一张"用这个。批量场景请用 save()。
    virtual void show(const core::BatchResult& result,
                      const std::string& windowName) const = 0;

protected:
    IRenderer() = default;
};

}  // namespace trt_alpha::renderer