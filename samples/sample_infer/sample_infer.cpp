// =============================================================================
//  sample_infer —— trt_alpha 高层 API（Infer）最简调用示例
// -----------------------------------------------------------------------------
//  YOLOV26 全任务族，每个任务一个最简样例函数：推理一段视频，并排显示
//  【原图】与【渲染图】，顺带打印该任务的结果该怎么读。
//
//  用法（在仓库根目录运行）：
//      sample_infer                 # 默认 det + data/people.mp4
//      sample_infer depth           # 深度估计
//      sample_infer pose out.mp4    # pose + 指定源（视频 / 图片 / 目录 / 流 / 摄像头 id）
//    第一个参数：det | cls | seg | pose | obb | depth | sem
//    窗口快捷键：ESC 或 q 退出。
//
//  三个要点（每个样例都适用，区别只在 model_type + config_path + 结果访问器）：
//    1. model_type 必须显式设置 —— 枚举默认值是 yolov3，不设就会拿 yolov26 的
//       config 去喂 V3 模型类，构造期直接失败。
//    2. 默认就会渲染：frame.rendered() 是画好框 / 掩码的副本，同时 frame.image()
//       仍是【原图】，渲染不会污染它。不要渲染图就 p.draw = false（省掉每帧绘制+拷贝）。
//       渲染默认不落盘、不显示；要存盘 / 弹窗再开 p.save / p.show。
//    3. 结果坐标 / 掩码统一为【原图尺寸】：
//         det   frame.boxes()            left/top/right/bottom + confidence + label
//         cls   frame.classifications()  label + score
//         seg   frame.masks()            box + mask(网络尺寸) + maskOrig(原图尺寸)
//         pose  frame.keypoints()        box + 17 个 {x, y, confidence}
//         obb   frame.obbs()             cx, cy, w, h, angle(rad) + confidence + label
//         depth frame.depth()            原图尺寸, vector<float>, 单位【米】
//         sem   frame.semantic()         原图尺寸, vector<uint8_t>, 类别 id
// =============================================================================
#include "trt_alpha/infer/infer.hpp"
#include <opencv2/opencv.hpp>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <string>

namespace {

//! BufferView -> cv::Mat：零拷贝，仅引用像素（只要 Result 活着就有效）。
//! 第 5 参（step）必须是 stride（行字节，含 padding），不是 width * channels。
cv::Mat toMat(const trt_alpha::core::BufferView& v)
{
    if (v.empty()) { return {}; }
    return cv::Mat(v.height, v.width, CV_MAKETYPE(CV_8U, v.channels),
                   const_cast<std::uint8_t*>(v.data), v.stride);
}

//! 显示这一帧的原图 + 渲染图。返回 false = 用户按了 ESC / q，该收工了。
bool showFrame(const std::string& tag, const trt_alpha::Frame& f)
{
    const cv::Mat raw   = toMat(f.image());      // 原图（原始像素，无框）
    const cv::Mat drawn = toMat(f.rendered());   // 渲染图（框 / 掩码已画好）
    if (!raw.empty())   { cv::imshow(tag + " | raw", raw); }
    if (!drawn.empty()) { cv::imshow(tag + " | rendered", drawn); }

    const int key = cv::waitKey(0);
    return !(key == 27 || key == 'q');
}

}  // namespace

// -----------------------------------------------------------------------------
//  每个任务一个样例：只有 model_type / config_path / 结果访问器不同。
// -----------------------------------------------------------------------------

//! 目标检测：boxes() —— 原图坐标的 left/top/right/bottom + confidence + label。
void sample_yolov26_det(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26;
    p.config_path = "configs/yolov26.ini";
    //p.source      = source;
    p.source = "F:/VideoAlpha/huaxue1.mp4";
    p.engine = "D:/VS2022_Project/TensorRT-Alpha/engines/yolov26n.trt";
    p.draw        = true;                      // 产出渲染图（默认即 true）

    // ---- 解码 / 回拷：四个参数都能在这里直接设，也都能在 ini 里设 ----
    // 优先级：这里的赋值 > configs/base.ini 的 [source] 段 > 代码默认。
    // 不赋值（保持空串 / -1）= 完全交给 ini；下面写的是显式的等值形式，
    // 删掉某一行就退回"读 ini"，效果一样。
    p.decode          = "ffmpeg";  // opencv | ffmpeg | auto（图片/目录/摄像头恒走 opencv）
    p.hwaccel         = "auto";    // auto | none | nvdec（仅 decode=ffmpeg 有意义）
    p.need_host_copy  = "auto";    // auto | on | off
    p.timeout_ms       = 5000;     // 仅 URL 流；0 = 不限，-1 = 用 ini
    //   need_host_copy 三个取值的区别：
    //     auto = 只在真有人读主机像素时才拷（draw / show / save 打开就叫"有人"）
    //     on   = 总是拷 —— draw=false 但还想用 frame.image() 时设它
    //     off  = 总不拷，像素留在显存。NVDEC 1080p 每帧省一次 6.2 MB 的 D2H
    //            （实测 0.475 ms / 帧）。⚠ 必须同时 draw=false，否则构造期就报错
    //            （渲染器不能在显存上画）；此时 frame.image() 返回空视图。

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            for (const auto& b : f.boxes())
                std::cout << "  det label=" << b.label << " conf=" << b.confidence
                          << " box=[" << b.left << "," << b.top << ","
                          << b.right << "," << b.bottom << "]\n";
            if (!showFrame("yolov26 det", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

//! 图像分类：classifications() —— 每帧一组 {label, score}。
void sample_yolov26_cls(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26_cls;
    p.config_path = "configs/yolov26_cls.ini";
    p.engine = "D:/VS2022_Project/TensorRT-Alpha/engines/yolov26n_cls.trt";
    p.source      = source;
    p.draw        = true;

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            for (const auto& c : f.classifications())
                std::cout << "  cls label=" << c.label << " score=" << c.score << "\n";
            if (!showFrame("yolov26 cls", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

//! 实例分割：masks() —— box + 网络尺寸 mask + 原图尺寸 maskOrig（0/255）。
void sample_yolov26_seg(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26_seg;
    p.config_path = "configs/yolov26_seg.ini";
    p.source      = source;
    p.draw        = true;
    p.engine = "D:/VS2022_Project/TensorRT-Alpha/engines/yolov26n_seg.trt";

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            for (const auto& s : f.masks()){
                cv::Mat maskNet = toMat(s.mask);        // 网络尺寸，始终有值（如 640x640，含 letterbox 黑边）
                cv::Mat maskOrig = toMat(s.maskOrig);    // 原图尺寸，默认空（见下面坑）
                std::cout << "  seg label=" << s.box.label
                    << " conf=" << s.box.confidence
                    << " mask=" << s.mask.width << "x" << s.mask.height     // 网络尺寸
                    << " maskOrig=" << s.maskOrig.width << "x" << s.maskOrig.height << "\n";
            }
            if (!showFrame("yolov26 seg", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

//! 姿态估计：keypoints() —— box + confidence + label，以及 17 个 {x, y, confidence}。
void sample_yolov26_pose(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26_pose;
    p.config_path = "configs/yolov26_pose.ini";
    p.source      = source;
    p.engine = "D:/VS2022_Project/TensorRT-Alpha/engines/yolov26n_pose.trt";
    p.draw        = true;

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            for (const auto& k : f.keypoints())
            {
                std::cout << "  pose label=" << k.box.label
                          << " conf=" << k.box.confidence
                          << " kpts=" << k.keypoints.size();
                if (!k.keypoints.empty())
                    std::cout << " p0=(" << k.keypoints[0].x << ","
                              << k.keypoints[0].y << ","
                              << k.keypoints[0].confidence << ")";
                std::cout << "\n";
            }
            if (!showFrame("yolov26 pose", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

//! 旋转框：obbs() —— cx, cy, w, h, angle(弧度) + confidence + label。
//! 注意：这是 DOTA 航拍 15 类模型，街景里通常只有零星检出，属正常。
void sample_yolov26_obb(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26_obb;
    p.config_path = "configs/yolov26_obb.ini";
    p.source      = source;
    p.draw        = true;

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            for (const auto& o : f.obbs())
                std::cout << "  obb label=" << o.label << " conf=" << o.confidence
                          << " cx=" << o.cx << " cy=" << o.cy
                          << " w=" << o.w << " h=" << o.h << " angle=" << o.angle << "\n";
            if (!showFrame("yolov26 obb", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

//! 深度估计：depth() —— 原图尺寸的 float 图，单位【米】（不是 uint8！）。
void sample_yolov26_depth(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26_depth;
    p.config_path = "configs/yolov26_depth.ini";
    p.source      = source;
    p.draw        = true;

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            const auto& d = f.depth();                 // width/height = 原图尺寸
            if (!d.empty())
            {
                const float center =
                    d.data[static_cast<std::size_t>(d.height / 2) * d.width + d.width / 2];
                std::cout << "  depth " << d.width << "x" << d.height
                          << " data=vector<float> center=" << center << " m\n";
            cv::Mat depth(d.height, d.width, CV_32FC1, const_cast<float*>(d.data.data()));
            cv::Mat u8, vis;
            cv::normalize(depth, u8, 0, 255, cv::NORM_MINMAX, CV_8U);   // 逐帧 min/max，画面会闪
            cv::applyColorMap(u8, vis, cv::COLORMAP_JET);

            }
            if (!showFrame("yolov26 depth", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

//! 语义分割：semantic() —— 原图尺寸的 uint8 类别图。
void sample_yolov26_sem(const std::string& source)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov26_sem;
    p.config_path = "configs/yolov26_sem.ini";
    p.source      = source;
    p.draw        = true;

    trt_alpha::Infer  model(p);
    trt_alpha::Stream stream = model.async();

    trt_alpha::Result r;
    while (stream.get(r))
    {
        for (const auto& f : r.frames())
        {
            const auto& s = f.semantic();              // width/height = 原图尺寸
            if (!s.empty())
            {
                const int center =
                    s.data[static_cast<std::size_t>(s.height / 2) * s.width + s.width / 2];
                std::cout << "  sem " << s.width << "x" << s.height
                          << " data=vector<uint8_t> center_class=" << center << "\n";
            }
            if (!showFrame("yolov26 sem", f)) { stream.stop(); return; }
        }
    }
    stream.stop();
}

int main(int argc, char** argv)
{
    // Infer 在参数非法时抛 std::exception（例如 need_host_copy=off 却还开着
    // draw，或引擎与 ini 的类别数不符）。示例里接住它，把原因打出来 ——
    // 不接的话进程直接 abort，看不到任何提示。
    try
    {
        //sample_yolov26_det("F:/VideoAlpha/huaxue1.mp4");
        //sample_yolov26_cls("F:/VideoAlpha/huaxue1.mp4");
        //sample_yolov26_seg("D:/Data/bus.jpg");
        //sample_yolov26_cls("D:/Data/bus.jpg");
        //sample_yolov26_pose("F:/VideoAlpha/huaxue1.mp4");
        //sample_yolov26_pose("D:/Data/bus.jpg");
        //sample_yolov26_obb("D:/Data/boats.mp4");
        //sample_yolov26_depth("F:/VideoAlpha/korea-night.mp4");
        sample_yolov26_depth("D:/Data/bus.jpg");
    }
    catch (const std::exception& e)
    {
        std::cerr << "sample_infer: " << e.what() << "\n";
        return 1;
    }
    return 0;
}
