// =============================================================================
//  sample_infer —— Infer 高层 API 使用示例
// -----------------------------------------------------------------------------
//  用法：
//    sample_infer                          # 默认 data/bus.jpg
//    sample_infer <path>                   # 图片 / 视频 / URL
//    sample_infer <camera_id>              # 摄像头（如 0）
// =============================================================================
#include "trt_alpha/infer/infer.hpp"

#include <cctype>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov8;
    p.config_path = "configs/yolov8.ini";
    p.show        = true;

    std::string display_source;
    bool is_stream = false;   // 视频 / 摄像头 / 流

    if (argc >= 2)
    {
        const std::string arg = argv[1];

        // 纯数字 → 摄像头 ID
        bool is_num = !arg.empty();
        for (char c : arg) {
            if (!std::isdigit(static_cast<unsigned char>(c))) { is_num = false; break; }
        }

        if (is_num)
        {
            p.camera_id = std::stoi(arg);
            display_source = "camera " + std::to_string(p.camera_id);
            is_stream = true;
        }
        else
        {
            p.source = arg;
            display_source = arg;
            is_stream =
                arg.find(".mp4") != std::string::npos ||
                arg.find(".avi") != std::string::npos ||
                arg.find(".mov") != std::string::npos ||
                arg.find("rtsp://") == 0 ||
                arg.find("http://") == 0  ||
                arg.find("https://") == 0;
        }
    }
    else
    {
        p.source = "data/bus.jpg";
        display_source = "data/bus.jpg";
    }

    trt_alpha::Infer model(p);

    std::cout << "=== sample_infer ===\n";
    std::cout << "source : " << display_source << "\n";

    if (is_stream)
    {
        // ---- 视频 / 摄像头 / 流：异步 ----
        auto stream = model.async();

        int batchCount = 0;
        trt_alpha::Result result;
        while (stream.get(result))
        {
            ++batchCount;
            for (auto& frame : result.frames())
            {
                auto img = frame.image();
                std::cout << "  batch " << batchCount
                          << " frame: " << img.width << "x" << img.height
                          << " boxes=" << frame.boxes().size() << "\n";
            }
        }
        stream.stop();
        std::cout << "total batches: " << batchCount << "\n";
    }
    else
    {
        // ---- 图片：同步 ----
        trt_alpha::Result result = model.run();

        std::cout << "size        : " << result.size() << "\n";
        std::cout << "valid_count : " << result.valid_count() << "\n";
        std::cout << "inference_ms: " << result.inference_ms() << "\n";

        for (auto& frame : result.frames())
        {
            auto img = frame.image();
            std::cout << "  frame: " << img.width << "x" << img.height << "\n";
            for (auto& box : frame.boxes())
            {
                std::cout << "    box: label=" << box.label
                          << " conf=" << box.confidence << "\n";
            }
        }
    }

    return 0;
}