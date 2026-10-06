// =============================================================================
//  sample_infer —— Infer 高层 API 使用示例
// -----------------------------------------------------------------------------
//  用法：
//    sample_infer                    # 默认 data/bus.jpg
//    sample_infer <path>             # 图片 / 视频 / URL
//    sample_infer <camera_id>        # 摄像头（如 0）
// =============================================================================
#include "trt_alpha/infer/infer.hpp"

#include <cctype>
#include <iostream>
#include <string>

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main(int argc, char** argv)
{
    trt_alpha::InferParams p;
    p.model_type  = trt_alpha::ModelType::yolov8;
    p.config_path = "configs/yolov8.ini";
    p.show        = true;

    std::string display_source;

    if (argc >= 2)
    {
        const std::string arg = argv[1];
        bool is_num = !arg.empty();
        for (char c : arg) {
            if (!std::isdigit(static_cast<unsigned char>(c))) { is_num = false; break; }
        }

        if (is_num) {
            p.camera_id = std::stoi(arg);
            display_source = "camera " + std::to_string(p.camera_id);
        } else {
            p.source = arg;
            display_source = arg;
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

    // 统一用异步
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

    // ---- [1] Stream 生命周期：Infer 先析构，Stream 必须仍可用 ----
    // Pipeline 持有 pool / renderer 的**裸指针**（都挂在 Infer::Impl 上），
    // 因此 Stream 必须保活 Impl —— 否则这里就是 use-after-free
    // （崩溃，或静默拿到垃圾结果）。
    {
        trt_alpha::Stream s;
        {
            trt_alpha::InferParams q;
            q.model_type  = p.model_type;
            q.config_path = p.config_path;
            q.source      = "data/bus.jpg";
            q.show        = false;    // 不渲染，get() 才会真的产出 Result
            q.save        = false;
            trt_alpha::Infer inner(q);
            s = inner.async();
        }   // inner 在此析构：pool / renderer 若被释放，下面的 get() 即悬垂

        int batches = 0;
        trt_alpha::Result r;
        while (s.get(r)) { ++batches; }
        s.stop();

        std::cout << "       batches after Infer dtor = " << batches << "\n";
        check(batches >= 1, "[1] Stream outlives Infer (no use-after-free)");
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}