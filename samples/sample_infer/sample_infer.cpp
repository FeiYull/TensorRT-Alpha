#include "trt_alpha/infer/infer.hpp"
#include <cctype>
#include <iostream>
#include <string>

int main(int argc, char** argv)
{
    trt_alpha::InferParams p;

    // 模型选择：sample_infer <model> <source>
    // 默认 yolov8
    std::string model_name = "yolov8";
    std::string source = "data/bus.jpg";

    if (argc >= 2) model_name = argv[1];
    if (argc >= 3) source = argv[2];

    // model 名 → ModelType
    if (model_name == "yolov8") {
        p.model_type = trt_alpha::ModelType::yolov8;
        p.config_path = "configs/yolov8.ini";
    } else if (model_name == "yolov8_seg") {
        p.model_type = trt_alpha::ModelType::yolov8_seg;
        p.config_path = "configs/yolov8_seg.ini";
    } else if (model_name == "u2net") {
        p.model_type = trt_alpha::ModelType::u2net;
        p.config_path = "configs/u2net.ini";
    } else if (model_name == "yolov8_pose") {
        p.model_type = trt_alpha::ModelType::yolov8_pose;
        p.config_path = "configs/yolov8_pose.ini";
    } else {
        std::cerr << "unknown model: " << model_name << "\n";
        return 1;
    }

    p.show = true;
    p.workers = 1;
    p.max_queue_size = 1;    // max_queue_size：推理池任务队列能放多少个任务。每个任务 = 一个 batch。
    p.result_queue_size = 1;    // 结果队列能放多少个future。每个 future = 一个 batch 的结果。

    
    // 摄像头（纯数字）
    bool is_num = !source.empty();
    for (char c : source) {
        if (!std::isdigit(static_cast<unsigned char>(c))) { is_num = false; break; }
    }
    if (is_num) p.camera_id = std::stoi(source);
    else        p.source = source;

    trt_alpha::Infer model(p);
    auto stream = model.async();

    int batchCount = 0;
    trt_alpha::Result r;
    while (stream.get(r)) {
        ++batchCount;
        for (auto& frame : r.frames()) {
            auto img = frame.image();
            std::cout << "  batch " << batchCount
                      << " frame: " << img.width << "x" << img.height
                      << " boxes=" << frame.boxes().size()
                      << " masks=" << frame.masks().size()
                      << " kpts=" << frame.keypoints().size() << "\n";
        }
    }
    stream.stop();
    std::cout << "total batches: " << batchCount << "\n";
    return 0;
}