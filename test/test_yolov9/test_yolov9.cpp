// =============================================================================
//  test/test_yolov9/test_yolov9.cpp
// -----------------------------------------------------------------------------
//  YOLOV9 的接入检查（不需要引擎，CTest 里恒能跑）。
//
//  YOLOV9 与 YOLO26 共用同一套实现（子类只改 m_tag），所以这里不重复逐框对拍
//  —— 数值等价由 test_yolov26_tasks --net yolov9 承担：
//
//      test_yolov26_tasks --net yolov9 --det <e.trt> --seg <e.trt> \
//                         --refdir D:/Data/trt_exp/ref/yolov9 <image...>
//
//  这个测试盯的是"接线"层面的四件事，每一件都曾经在别的模型上静默出过错：
//
//    [1] 注册中心确实有这 2 个名字（link 时 WHOLE_ARCHIVE 生效）
//    [2] name() 返回注册名 —— 证明子类的 m_tag 真的被用上，而不是还叫 yolov26
//    [3] configs/yolov9*.ini 能被 loadModelConfig 解出预期值
//        （engine / batch / num_class / extra；键名写错段会在这里暴露）
//    [4] init 在引擎缺失时显式抛异常，且异常前缀带自己的标签
//
//  ⚠️ YOLOV9 只有 det 与 seg：官方权重是 yolov9t/s/m/c/e 与 yolov9c-seg/e-seg
//  （没有 t/s 的 seg），也没有 cls / obb / pose。多注册一个就会让 --net 选到
//  语义完全不同的模型，所以这里显式断言"不该存在"。
//
//  退出码：0 = 全通过，1 = 有失败。
// =============================================================================
#include "trt_alpha/core/config.hpp"
#include "trt_alpha/core/model.hpp"
#include "trt_alpha/core/model_registry.hpp"

#include <algorithm>
#include <exception>
#include <iostream>
#include <string>
#include <vector>

namespace {

int g_failures = 0;

void check(bool cond, const std::string& what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

struct Case
{
    const char* net;        //!< registered name == INI stem
    const char* engine;     //!< expected [model] engine value
    int         numClass;
    int         extra;      //!< -1 = the INI must NOT define it
};

const Case kCases[] = {
    {"yolov9",     "engines/yolov9t.trt",    80, -1},   // det: extra 缺省 0
    {"yolov9_seg", "engines/yolov9c_seg.trt", 80, 32},
};

}  // namespace

int main()
{
    std::cout << "============ test_yolov9 ============\n";

    // ---- [1] registry ----
    std::cout << "\n---- [1] registry ----\n";
    {
        const std::vector<std::string> all = trt_alpha::ModelRegistry::instance().names();
        for (const Case& c : kCases)
        {
            const bool ok = std::find(all.begin(), all.end(), std::string(c.net)) != all.end();
            check(ok, std::string("registered: ") + c.net);
        }
        for (const char* absent : {"yolov9_cls", "yolov9_obb", "yolov9_pose"})
        {
            const bool ok = std::find(all.begin(), all.end(), std::string(absent)) == all.end();
            check(ok, std::string("not registered (no such task): ") + absent);
        }
    }

    // ---- [2] name() == registered name ----
    std::cout << "\n---- [2] name() carries the tag ----\n";
    for (const Case& c : kCases)
    {
        std::unique_ptr<trt_alpha::IModel> m =
            trt_alpha::ModelRegistry::instance().create(c.net);
        check(m != nullptr && m->name() == c.net,
              std::string("name() == '") + c.net + "' (got '" +
              (m ? m->name() : std::string("<null>")) + "')");
    }

    // ---- [3] INI 解析 ----
    std::cout << "\n---- [3] configs/<net>.ini ----\n";
    for (const Case& c : kCases)
    {
        const std::string ini = std::string("configs/") + c.net + ".ini";
        trt_alpha::core::ModelConfig cfg;
        bool loaded = true;
        try { cfg = trt_alpha::core::loadModelConfig(ini); }
        catch (const std::exception& e)
        {
            loaded = false;
            std::cout << "       " << ini << " -> " << e.what() << "\n";
        }
        check(loaded, ini + " loads");
        if (!loaded) { continue; }

        check(cfg.engine == c.engine,
              ini + " engine = " + c.engine + " (got '" + cfg.engine + "')");
        check(cfg.batchSize == 2,
              ini + " batch_size = 2 (got " + std::to_string(cfg.batchSize) + ")");
        check(cfg.getInt("num_class", -1) == c.numClass,
              ini + " num_class = " + std::to_string(c.numClass) + " (got " +
              std::to_string(cfg.getInt("num_class", -1)) + ")");

        if (c.extra >= 0)
        {
            check(cfg.getInt("extra", -1) == c.extra,
                  ini + " extra = " + std::to_string(c.extra) + " (got " +
                  std::to_string(cfg.getInt("extra", -1)) + ")");
        }
        else
        {
            // extra 没写：det 由 C++ 缺省成 0。写进 INI 就是两个必须互相一致的
            // 真相源，这个仓库不接受。
            const auto it = cfg.extras.find("extra");
            check(it == cfg.extras.end(), ini + " does not define extra");
        }

        // 布局必须在 INI 里显式写出（不是靠全局兜底），且是 yolo 系的 NCHW。
        // cfg.layout 是已解析的 Layout，写错会被 loadModelConfig 直接拒绝。
        check(!cfg.layout.empty(), ini + " defines input.layout");
        check(cfg.layout.str() == "NCHW",
              ini + " input.layout = NCHW (got '" + cfg.layout.str() + "')");
    }

    // ---- [4] 缺引擎必须显式抛；校验失败时前缀是自己的标签 ----
    std::cout << "\n---- [4] init fails loudly, tagged with the model name ----\n";
    for (const Case& c : kCases)
    {
        std::unique_ptr<trt_alpha::IModel> m =
            trt_alpha::ModelRegistry::instance().create(c.net);
        if (!m) { continue; }

        trt_alpha::core::ModelConfig cfg;
        try { cfg = trt_alpha::core::loadModelConfig(std::string("configs/") + c.net + ".ini"); }
        catch (...) { continue; }

        trt_alpha::core::ModelConfig missingEngine = cfg;
        missingEngine.engine = "D:/definitely/not/exist/yolov9_probe.trt";
        std::string msg;
        try { m->init(missingEngine); }
        catch (const std::exception& e) { msg = e.what(); }
        check(!msg.empty(), std::string(c.net) + " threw on a missing engine");

        trt_alpha::core::ModelConfig badClass = cfg;
        badClass.extras["num_class"] = "0";
        std::string tagged;
        try { m->init(badClass); }
        catch (const std::exception& e) { tagged = e.what(); }
        check(!tagged.empty(), std::string(c.net) + " rejected num_class = 0");
        if (!tagged.empty())
        {
            std::cout << "       " << tagged << "\n";
            check(tagged.rfind(c.net, 0) == 0,
                  std::string("exception prefix is '") + c.net + "'");
        }
    }

    std::cout << "\n============ summary ============\n";
    if (g_failures == 0) { std::cout << "  ALL PASS\n"; return 0; }
    std::cout << "  FAILURES: " << g_failures << "\n";
    return 1;
}
