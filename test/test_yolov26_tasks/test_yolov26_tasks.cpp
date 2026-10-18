// =============================================================================
//  test/test_yolov26_tasks/test_yolov26_tasks.cpp
// -----------------------------------------------------------------------------
//  yolo26 的 depth / sem / cls / pose / seg / obb 六个任务。
//
//  这些测试【不】照抄 test_yolov8 的空壳：每一条都是能被证伪的真断言。
//
//    [1] 注册中心能创建 yolov26_depth / yolov26_sem / yolov26_cls
//    [2] init 在引擎缺失时显式抛异常（不是静默返回）
//    [3] CvTopKPlan 的 top-k 结果逐位正确（含并列时取小索引的规则）
//    [4] CvScaleMaskPlan<float> 的裁剪窗 == computeLetterBox（同一真相源）
//    [5] CvScaleMaskPlan<unsigned char> 的 Bilinear 必须被拒绝
//        （对类别 id 做加权平均会造出不存在的类，必须显式失败）
//    [6] CvScaleMaskPlan<unsigned char>(Nearest) 的最近邻索引正确
//    [7] CvPreprocessVecClsPlan 的几何 == torchvision
//        _compute_resized_output_size + center_crop
//        810x1080 -> 224 时 resize 后必须是 298（int() 截断），不是 299；
//        这是踩过的坑，留成回归守卫
//    [8] 真推理（可选，需要 cls 引擎 + 图片，由命令行给出）：跑完
//        setBatch/preprocess/infer/postprocess，把 top-1 与 py 官方
//        YOLO.predict(imgsz=224) 的期望类别逐一断言，并检查概率向量归一化。
//        期望值来自 E:\Code\ultralytics\sry_export.py 导出的 224 ONNX
//        所对应的 .pt 模型，见表 kClsExpect。
//    [9] 端到端 depth（可选）：与 py 官方参考【逐元素】比对整张深度图。
//    [10] 端到端 sem（可选）：与 py 官方参考【逐像素】比对类别图。
//    [11] 端到端 pose（可选）：与 py 官方参考【逐框 + 逐关键点】比对人的姿态。
//    [12] 端到端 seg（可选）：与 py 官方参考【逐框 + 逐像素】比对实例掩码。
//    [13] 端到端 obb（可选）：与 py 官方参考【逐框】比对旋转框（含 Fast-NMS
//         + probiou，与检测的贪心 NMS 不是一回事）。
//
//  ⚠️ [9]/[10] 的参考必须取【引擎/ONNX 后端】，不能取 .pt 的 predict()。
//     ultralytics 自己按后端分叉了两处（见 D:\Data\trt_exp\dump_depth_sem_ref.py）：
//       (a) 预处理 engine/predictor.py:230
//             auto = same_shapes and rect and (model.format == "pt" or dynamic)
//           → .pt 走【矩形】letterbox（1280x720 会变成 384x640），静态
//             onnx/engine 走【正方形】640x640。静态引擎根本喂不进矩形版，
//             而且 depth 头融合了 P3/P4/P5，灰边留几行会改变结果 ——
//             实测同一张图差到 19.7 m，不是舍入误差。
//       (b) 后处理 semantic/predict.py:67
//             class_map_input = pred.ndim == 2   # 图里烘了 ArgMax 就发 [H,W]
//           → sem 的 ONNX 把 8x 上采样和 argmax 都烘进去了，走"类别图"分支
//             （只 nearest）；跑 .pt 会走"logits"分支
//             （bilinear 上采样 → scale_masks(bilinear) → argmax），
//             算子顺序不同，答案就不同（实测差 2.8%~18.5% 像素）。
//     [11] 同理：参考必须走 onnxruntime + 官方 nms.non_max_suppression
//     （见 D:\Data\trt_exp\dump_pose_ref.py），拿 .pt 的 predict() 会同时踩
//     上上面两条分叉。[12] 一样（见 D:\Data\trt_exp\dump_seg_ref.py），而且
//     seg 的掩码是 1/4 分辨率的特征图，灰边留几行同样会改结果。[13] 还多一层：
//     rotated=True 走的是 Fast-NMS(probiou)，不是贪心 NMS，所以参考必须用
//     rotated=True 跑（见 D:\Data\trt_exp\dump_obb_ref.py）。
//
//  用法：
//      test_yolov26_tasks
//      test_yolov26_tasks <cls.trt> <image> [image...]          # 旧式位置参数
//      test_yolov26_tasks [--net <prefix>]
//                         [--cls <e.trt>] [--depth <e.trt>] [--sem <e.trt>]
//                         [--pose <e.trt>] [--seg <e.trt>] [--obb <e.trt>]
//                         [--refdir <dir>[,<dir>...]] <image> [image...]
//
//      --net 模型名前缀，默认 yolov26；给 yolov11 就跑 YOLOV11 那五个任务
//            （YOLOV11 没有 depth / sem）。两边共用同一套实现与同一套断言。
//
//      --refdir 指向 D:\Data\trt_exp\dump_*_ref.py 的输出目录；
//      没有它时 [9]/[10]/[11]/[12]/[13] 只打印统计量、记 SKIP，不做断言。
//
//  退出码：0 = 全通过，1 = 有失败，77 = 可选段被跳过。
// =============================================================================
#include "trt_alpha/cls/classifier.hpp"
#include "trt_alpha/core/buffer.hpp"
#include "trt_alpha/core/data_type.hpp"
#include "trt_alpha/core/model_registry.hpp"
#include "trt_alpha/det/detector.hpp"
#include "trt_alpha/depth/depth_estimator.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_cls.hpp"
#include "trt_alpha/kernels/ops/preprocess_fused_vec_yolov26_det.hpp"
#include "trt_alpha/kernels/ops/scale_mask_yolov26.hpp"
#include "trt_alpha/kernels/ops/topk_yolov26_cls.hpp"
#include "trt_alpha/kpt/keypointer.hpp"
#include "trt_alpha/kpt/types.hpp"
#include "trt_alpha/obb/obb_detector.hpp"
#include "trt_alpha/obb/types.hpp"
#include "trt_alpha/seg/segmentor.hpp"
#include "trt_alpha/seg/types.hpp"
#include "trt_alpha/sem/semantic_segmentor.hpp"

#include <opencv2/imgcodecs.hpp>

#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <fstream>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

using trt_alpha::IModel;
using trt_alpha::ModelRegistry;
using trt_alpha::cls::ClassScore;
using trt_alpha::cls::IClassifier;
using trt_alpha::core::Batch;
using trt_alpha::core::Buffer;
using trt_alpha::core::BufferView;
using trt_alpha::core::DataType;
using trt_alpha::core::MemorySpace;
using trt_alpha::core::ModelConfig;
using trt_alpha::depth::DepthMap;
using trt_alpha::depth::IDepthEstimator;
using trt_alpha::det::Detection;
using trt_alpha::det::IDetector;
using trt_alpha::kernels::ops::computeLetterBox;
using trt_alpha::kernels::ops::CvPreprocessVecClsPlan;
using trt_alpha::kernels::ops::CvScaleMaskPlan;
using trt_alpha::kernels::ops::CvTopKPlan;
using trt_alpha::kernels::ops::LetterBoxInfo;
using trt_alpha::kernels::ops::MaskInterp;
using trt_alpha::kpt::IKeypointer;
using trt_alpha::kpt::KeypointResult;
using trt_alpha::obb::IObbDetector;
using trt_alpha::obb::ObbDetection;
using trt_alpha::seg::ISegmentor;
using trt_alpha::seg::Segmentation;
using trt_alpha::sem::ISemanticSegmentor;
using trt_alpha::sem::SemanticMask;

namespace {

int g_failures = 0;

// Defined further down, next to the depth/sem reference loaders; the [8]
// classification section needs them earlier.
std::string stripExtension(const std::string& p);
std::string locateRef(const std::vector<std::string>& dirs, const std::string& file);

//! Model-name prefix: "yolov26" (default), "yolov11", "yolov8", "yolov9",
//! "yolov10".  All families share one implementation (they all derive from
//! YOLOV26), so the same assertions apply; only the registered names and the
//! engines differ.
std::string g_netPrefix = "yolov26";

//! "<prefix>_<task>", e.g. "yolov26_seg" / "yolov11_seg".
std::string netName(const char* task)
{
    return g_netPrefix + "_" + task;
}

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

void checkEq(long long got, long long want, const char* what)
{
    const bool ok = (got == want);
    std::cout << (ok ? "[PASS] " : "[FAIL] ") << what
              << "  (got " << got << ", want " << want << ")\n";
    if (!ok) { ++g_failures; }
}

// =============================================================================
//  [3] TopK -- 手算的期望值，不做"看起来合理"的软检查
// =============================================================================
void testTopK()
{
    std::cout << "\n---- [3] CvTopKPlan ----\n";

    // 故意制造并列：0.9 出现两次（idx 2 和 idx 6），必须取小索引 2。
    const std::vector<float> scores = {
        0.10f, 0.50f, 0.90f, 0.20f, 0.70f, 0.30f, 0.90f, 0.05f,
    };
    const int   nc = static_cast<int>(scores.size());
    const int   k  = 4;
    const int   B  = 2;

    // 期望： 2(0.90) -> 6(0.90) -> 4(0.70) -> 1(0.50)
    const int   wantLabel[4] = {2, 6, 4, 1};
    const float wantScore[4] = {0.90f, 0.90f, 0.70f, 0.50f};

    std::vector<float> host(static_cast<std::size_t>(B) * nc);
    for (int b = 0; b < B; ++b)
    {
        std::memcpy(host.data() + static_cast<std::size_t>(b) * nc,
                    scores.data(), sizeof(float) * static_cast<std::size_t>(nc));
    }

    CvTopKPlan plan(nc, k, B);
    check(plan.ok(), "topk plan builds");
    checkEq(plan.outStride(), 2 * k, "topk outStride == 2*k");

    float* dIn  = nullptr;
    float* dOut = nullptr;
    cudaMalloc(&dIn,  host.size() * sizeof(float));
    cudaMalloc(&dOut, static_cast<std::size_t>(B) * 2 * k * sizeof(float));
    cudaMemcpy(dIn, host.data(), host.size() * sizeof(float), cudaMemcpyHostToDevice);

    plan.launch(/*stream=*/0, B, dIn, dOut);
    check(plan.ok(), "topk launch ok");

    std::vector<float> out(static_cast<std::size_t>(B) * 2 * k);
    cudaMemcpy(out.data(), dOut, out.size() * sizeof(float), cudaMemcpyDeviceToHost);

    for (int b = 0; b < B; ++b)
    {
        for (int i = 0; i < k; ++i)
        {
            const float lbl = out[static_cast<std::size_t>(b) * 2 * k + 2 * i];
            const float scr = out[static_cast<std::size_t>(b) * 2 * k + 2 * i + 1];
            const bool  ok  = (static_cast<int>(lbl) == wantLabel[i]) && (scr == wantScore[i]);
            if (!ok)
            {
                std::cout << "[FAIL] topk batch" << b << " rank" << i
                          << "  got (label " << lbl << ", score " << scr << ")"
                          << "  want (label " << wantLabel[i] << ", score " << wantScore[i] << ")\n";
                ++g_failures;
            }
        }
    }
    if (g_failures == 0)
    {
        std::cout << "[PASS] topk values + labels exact, ties resolve to the smaller index\n";
    }

    cudaFree(dIn);
    cudaFree(dOut);
}

// =============================================================================
//  [4][5] CvScaleMaskPlan -- 几何 + 显式失败
// =============================================================================
void testScaleMaskGeometry()
{
    std::cout << "\n---- [4][5] CvScaleMaskPlan geometry / explicit failure ----\n";

    const int cases[][2] = {{500, 375}, {1280, 720}, {333, 500}, {1920, 1080}};
    for (const auto& c : cases)
    {
        const int srcW = c[0];
        const int srcH = c[1];

        CvScaleMaskPlan<float> plan(srcW, srcH, 640, 640, MaskInterp::Bilinear);
        check(plan.ok(), "scale_mask<float> plan builds");

        const LetterBoxInfo lb = computeLetterBox(srcW, srcH, 640, 640);
        const auto& info = plan.info();

        if (info.newW != lb.newW || info.newH != lb.newH ||
            info.cropLeft != lb.padLeft || info.cropTop != lb.padTop)
        {
            std::cout << "[FAIL] scale_mask crop window != computeLetterBox for "
                      << srcW << "x" << srcH
                      << "  plan=[" << info.cropTop << ":" << (info.cropTop + info.newH)
                      << "," << info.cropLeft << ":" << (info.cropLeft + info.newW) << "]"
                      << "  letterbox=[" << lb.padTop << ":" << (lb.padTop + lb.newH)
                      << "," << lb.padLeft << ":" << (lb.padLeft + lb.newW) << "]\n";
            ++g_failures;
        }
        else
        {
            std::cout << "[PASS] scale_mask crop window == computeLetterBox for "
                      << srcW << "x" << srcH << "\n";
        }
    }

    // 对类别 id 做双线性 = 造出不存在的类 -> 必须显式失败，不能静默算下去
    CvScaleMaskPlan<unsigned char> bad(500, 375, 640, 640, MaskInterp::Bilinear);
    check(!bad.ok(), "scale_mask<u8> bilinear is rejected (ok() == false)");
    check(bad.lastError() == cudaErrorInvalidValue,
          "scale_mask<u8> bilinear reports cudaErrorInvalidValue");

    CvScaleMaskPlan<unsigned char> good(500, 375, 640, 640, MaskInterp::Nearest);
    check(good.ok(), "scale_mask<u8> nearest is accepted");
}

// =============================================================================
//  [6] Nearest 索引 -- 用可预测的斜坡输入验证
// =============================================================================
void testNearestIndices()
{
    std::cout << "\n---- [6] CvScaleMaskPlan<u8> nearest indices ----\n";

    // 4x4 源图，每个像素 = row*16 + col（可唯一反推坐标）。
    const int netW = 4, netH = 4;
    std::vector<std::uint8_t> src(static_cast<std::size_t>(netW) * netH);
    for (int y = 0; y < netH; ++y)
    {
        for (int x = 0; x < netW; ++x)
        {
            src[static_cast<std::size_t>(y) * netW + x] =
                static_cast<std::uint8_t>(y * 16 + x);
        }
    }

    // padding=false -> 整个源图都是内容，裁剪窗 = 全图
    CvScaleMaskPlan<unsigned char> plan(
        2, 2, netW, netH, MaskInterp::Nearest, /*padding=*/false);
    check(plan.ok(), "nearest 4x4 -> 2x2 plan builds");

    std::uint8_t* dSrc = nullptr;
    std::uint8_t* dDst = nullptr;
    cudaMalloc(reinterpret_cast<void**>(&dSrc), src.size());
    cudaMalloc(reinterpret_cast<void**>(&dDst), 2 * 2);
    cudaMemcpy(dSrc, src.data(), src.size(), cudaMemcpyHostToDevice);

    plan.launch(0, 1, dSrc, dDst, 1);
    check(plan.ok(), "nearest launch ok");

    std::uint8_t out[4] = {0, 0, 0, 0};
    cudaMemcpy(out, dDst, 4, cudaMemcpyDeviceToHost);

    // rh = rw = 4/2 = 2；oy=0 -> y=floor(0*2)=0；oy=1 -> y=floor(1*2)=2
    // 所以取的是源图 (0,0) (0,2) (2,0) (2,2) = 0, 2, 32, 34
    const std::uint8_t want[4] = {0, 2, 32, 34};
    for (int i = 0; i < 4; ++i)
    {
        const bool ok = (out[i] == want[i]);
        std::cout << (ok ? "[PASS] " : "[FAIL] ") << "nearest out[" << i << "]"
                  << " = " << (int)out[i] << " want " << (int)want[i] << "\n";
        if (!ok) { ++g_failures; }
    }

    cudaFree(dSrc);
    cudaFree(dDst);
}

// =============================================================================
//  [7] cls 几何 -- torchvision 的 int() 截断，不是四舍五入
// =============================================================================
void testClsGeometry()
{
    std::cout << "\n---- [7] CvPreprocessVecClsPlan geometry ----\n";

    struct Case { int srcW, srcH, out; int newW, newH, cropLeft, cropTop; };

    // 期望值来自 onnxruntime/ultralytics 官方链实测（见 D:\Data\trt_exp\probe_cls_geom.py）
    const Case cases[] = {
        // bus.jpg 810x1080 -> 短边 810 缩到 224 -> 224x298（299 是错答案！）
        {1080, 810, 224, 298, 224, 37, 0},
        // zidane.jpg 1280x720 -> 398x224
        {1280,  720, 224, 398, 224, 87, 0},
        // 正方图 -> 不做裁剪
        {1280, 1280, 224, 224, 224,   0, 0},
        // 6.jpg 800x657 -> 272x224（273 是错答案！）
        { 800,  657, 224, 272, 224, 24, 0},
        // 引擎导出的分辨率是 640
        {1080,  810, 640, 853, 640, 106, 0},
    };

    for (const auto& c : cases)
    {
        CvPreprocessVecClsPlan plan =
            CvPreprocessVecClsPlan::byDivisor(c.srcW, c.srcH, c.out, c.out, 255.0f);
        if (!plan.ok())
        {
            std::cout << "[FAIL] cls plan builds for " << c.srcW << "x" << c.srcH
                      << " -> " << c.out << "\n";
            ++g_failures;
            continue;
        }

        const auto& g = plan.info();
        const bool ok = (g.newW == c.newW) && (g.newH == c.newH)
                     && (g.cropLeft == c.cropLeft) && (g.cropTop == c.cropTop);
        std::cout << (ok ? "[PASS] " : "[FAIL] ")
                  << "cls " << c.srcW << "x" << c.srcH << " -> " << c.out
                  << "  resize " << g.newW << "x" << g.newH
                  << " crop(" << g.cropLeft << "," << g.cropTop << ")"
                  << "  want resize " << c.newW << "x" << c.newH
                  << " crop(" << c.cropLeft << "," << c.cropTop << ")\n";
        if (!ok) { ++g_failures; }
    }
}

}  // namespace

// =============================================================================
//  [8] End-to-end classification, asserted against the Python reference
// -----------------------------------------------------------------------------
//  Expected values come from ultralytics itself on the same .pt that produced
//  yolo26n-cls.onnx @ imgsz=224:
//      YOLO("yolo26n-cls.pt").predict(img, imgsz=224)[0].probs.data
//  A wrong Pillow geometry, a wrong centre-crop offset or a wrong top-k would
//  all move the label, so matching top-1 end to end exercises the whole chain.
// =============================================================================
namespace {

struct ClsExpect
{
    const char* file;
    int         top1;
    float       prob;
};

//! filename -> (top1 class, its probability) measured on the reference.
const ClsExpect kClsExpect[] = {
    {"bus.jpg",       654, 0.478036f},
    {"zidane.jpg",    834, 0.491066f},
    {"6.jpg",         617, 0.186033f},
    {"12801.jpg",     558, 0.057597f},
    {"12802.jpg",     654, 0.510398f},
    {"road1.jpg",     867, 0.256400f},
    {"dog.jpg",       248, 0.219346f},
    {"sailboat3.jpg", 484, 0.538285f},
};

std::string baseName(const std::string& p)
{
    const std::size_t s = p.find_last_of("/\\");
    return (s == std::string::npos) ? p : p.substr(s + 1);
}

Batch batchFromMat(const cv::Mat& img, int batchSize)
{
    const int W = img.cols;
    const int H = img.rows;
    const std::size_t oneFrame = static_cast<std::size_t>(W) * H * 3;

    auto buffer = Buffer::createHost(W * batchSize, H, 3, DataType::UInt8);
    std::memcpy(buffer->mutableData(), img.data, oneFrame);
    if (batchSize > 1)
    {
        std::memset(buffer->mutableData() + oneFrame, 0,
                    oneFrame * static_cast<std::size_t>(batchSize - 1));
    }

    Batch b;
    b.sourceId = 0;
    b.firstFrameIndex = 0;
    b.buffer = buffer;
    b.validCount = 1;
    for (int i = 0; i < batchSize; ++i)
    {
        BufferView v;
        v.data     = buffer->data() + static_cast<std::size_t>(i) * oneFrame;
        v.width    = W;
        v.height   = H;
        v.stride   = W * 3;
        v.channels = 3;
        v.dtype    = DataType::UInt8;
        v.space    = MemorySpace::Host;
        b.views.push_back(v);
    }
    return b;
}

int runClsEndToEnd(const std::string& enginePath, int batch,
                   const std::vector<std::string>& images,
                   const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [8] end-to-end classification (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(netName("cls"));
    auto* model = dynamic_cast<IClassifier*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] yolov26_cls is not an IClassifier\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 224;          // the engine's declared shape wins anyway
    cfg.dstH      = 224;
    cfg.extras["num_class"] = "1000";
    cfg.extras["top_k"]     = "5";
    cfg.extras["scale"]     = "255.0";
    cfg.extras["pad_value"] = "0.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const auto& scores = model->classifications();
        if (scores.empty() || scores[0].empty())
        {
            std::cout << "[FAIL] " << baseName(p) << ": no classification produced\n";
            ++failures;
            model->reset();
            continue;
        }
        // NOTE: `scores` is a REFERENCE into the model, and reset() clears that
        // very vector -- so the reference must not be used after reset(). Take a
        // copy and reset at the end of the iteration instead.
        const std::vector<ClassScore> top = scores[0];

        const std::string nm = baseName(p);
        const ClsExpect* exp = nullptr;
        for (const auto& e : kClsExpect)
        {
            if (nm == e.file) { exp = &e; break; }
        }

        // top-1 always reported
        std::cout << "       " << nm << "  top5 =";
        for (const ClassScore& cs : top)
        {
            std::cout << " " << cs.label << ":" << cs.score;
        }
        std::cout << "\n";

        // Per-model reference (D:\Data\trt_exp\ref_dump.py --task cls):
        //     <stem>.cls.f32.bin
        //         int32  n            (n = top_k, best first)
        //         int32  labels[n]
        //         float  scores[n]    softmax probabilities, descending
        // The kClsExpect table below only holds values for one particular
        // model, so it is the fallback when no bin is present.
        const std::string clsRef =
            locateRef(refDirs, stripExtension(nm) + ".cls.f32.bin");
        if (!clsRef.empty())
        {
            std::ifstream rin(clsRef, std::ios::binary | std::ios::ate);
            const std::streamsize rtotal = rin.tellg();
            rin.seekg(0);
            std::vector<char> rraw(static_cast<std::size_t>(rtotal));
            rin.read(rraw.data(), rtotal);

            int refN = 0;
            const std::size_t head = sizeof(std::int32_t);
            if (rtotal >= static_cast<std::streamsize>(head))
            {
                std::memcpy(&refN, rraw.data(), head);
            }
            const std::size_t rwant =
                head + static_cast<std::size_t>(refN) *
                           (sizeof(std::int32_t) + sizeof(float));
            if (refN <= 0 || static_cast<std::size_t>(rtotal) != rwant)
            {
                std::cout << "[FAIL] " << nm << ": reference " << clsRef
                          << " has " << rtotal << " bytes, expected " << rwant
                          << " (n=" << refN << ")\n";
                ++failures;
                model->reset();
                continue;
            }

            const auto* rlab = reinterpret_cast<const std::int32_t*>(
                rraw.data() + head);
            const auto* rsc = reinterpret_cast<const float*>(
                rraw.data() + head +
                static_cast<std::size_t>(refN) * sizeof(std::int32_t));

            // Softmax over the same logits: TRT fp32 vs torch CPU fp32 differ by
            // rounding only. The real number is printed below either way.
            constexpr float kScoreTol = 5e-4f;

            float maxScoreD = 0.f;
            int   bad = 0;
            const int cmp = std::min(static_cast<int>(top.size()), refN);
            for (int k = 0; k < cmp; ++k)
            {
                const float d = std::fabs(
                    top[static_cast<std::size_t>(k)].score - rsc[k]);
                maxScoreD = std::max(maxScoreD, d);
                if (top[static_cast<std::size_t>(k)].label != rlab[k] ||
                    d > kScoreTol)
                {
                    if (bad < 4)
                    {
                        std::cout << "[FAIL] " << nm << ": top-" << (k + 1)
                                  << " got " << top[static_cast<std::size_t>(k)].label
                                  << ":" << top[static_cast<std::size_t>(k)].score
                                  << " want " << rlab[k] << ":" << rsc[k] << "\n";
                    }
                    ++bad;
                }
            }
            if (static_cast<int>(top.size()) != refN)
            {
                std::cout << "[FAIL] " << nm << ": top_k size got " << top.size()
                          << " want " << refN << "\n";
                ++bad;
            }

            std::cout << "       " << nm << "  n=" << refN
                      << "  max|d| score=" << maxScoreD << "\n";
            if (bad == 0)
            {
                std::cout << "[PASS] " << nm << " top-k matches the reference\n";
            }
            else
            {
                std::cout << "[FAIL] " << nm << ": " << bad
                          << " top-k entr(ies) out of tolerance\n";
                ++failures;
            }
            model->reset();
            continue;
        }

        if (exp == nullptr)
        {
            std::cout << "[SKIP] " << nm << " (no reference value)\n";
            model->reset();
            continue;
        }

        const bool labelOk = (top[0].label == exp->top1);
        const bool probOk  = std::fabs(top[0].score - exp->prob) <= 2e-3f;
        const bool topkOk  = (top.size() == 5);

        if (labelOk && probOk && topkOk)
        {
            std::cout << "[PASS] " << nm << " top1 = class " << exp->top1
                      << " (" << top[0].score << " vs reference "
                      << exp->prob << ")\n";
        }
        else
        {
            std::cout << "[FAIL] " << nm
                      << "  top1 got " << top[0].label
                      << " want " << exp->top1
                      << "  prob got " << top[0].score
                      << " want " << exp->prob
                      << (topkOk ? "" : "  (top_k size != 5)") << "\n";
            ++failures;
        }
        model->reset();
    }

    return failures;
}

// =============================================================================
//  Reference bins (shared by [9] and [10])
// -----------------------------------------------------------------------------
//  Produced by  D:\Data\trt_exp\dump_depth_sem_ref.py
//      <stem>.depth.f32.bin   H*W float32, C order, original image size
//      <stem>.sem.u8.bin      H*W uint8,   C order, original image size
//  The depth reference comes from the official .pt predictor; the sem reference
//  comes from the ONNX through onnxruntime + the official ops.scale_masks --
//  because ultralytics itself branches on whether the backend baked the argmax
//  in (semantic/predict.py:67), and the TRT engine IS a baked-argmax backend.
// =============================================================================

bool readBin(const std::string& path, void* dst, std::size_t bytes)
{
    std::ifstream in(path, std::ios::binary | std::ios::ate);
    if (!in)
    {
        return false;
    }
    const std::streamsize n = in.tellg();
    if (n < 0 || static_cast<std::size_t>(n) != bytes)
    {
        std::cout << "       (reference size mismatch: " << path << " has " << n
                  << " bytes, expected " << bytes << ")\n";
        return false;
    }
    in.seekg(0);
    in.read(static_cast<char*>(dst), n);
    return in.good() || in.eof();
}

std::string stripExtension(const std::string& p)
{
    const std::size_t s = p.find_last_of('.');
    return (s == std::string::npos) ? p : p.substr(0, s);
}

//! Look for `file` in every --refdir, in the order they were given. Returns an
//! empty string when the reference does not exist, so a partially populated
//! directory is reported as SKIP rather than FAIL.
std::string locateRef(const std::vector<std::string>& dirs, const std::string& file)
{
    for (const std::string& d : dirs)
    {
        const std::string p = d + "/" + file;
        std::ifstream in(p, std::ios::binary);
        if (in.good()) { return p; }
    }
    return std::string();
}

// -----------------------------------------------------------------------------
//  [9] depth end to end, element by element against the Python reference
// -----------------------------------------------------------------------------
int runDepthEndToEnd(const std::string& enginePath, int batch,
                     const std::vector<std::string>& images,
                     const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [9] end-to-end depth (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(netName("depth"));
    auto* model = dynamic_cast<IDepthEstimator*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] yolov26_depth is not an IDepthEstimator\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 640;          // the engine's declared shape wins anyway
    cfg.dstH      = 640;
    cfg.extras["scale"]     = "255.0";
    cfg.extras["pad_value"] = "114.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    // A bilinear resample is a convex combination, so a bounded input error
    // stays bounded.  Both sides are fp32 but different implementations
    // (TRT kernels vs torch CPU), so pick a tolerance well above 1 ULP yet
    // far below anything that could hide a geometry mistake (which would move
    // whole regions by metres).
    const float kDepthAbsTol = 2e-3f;   // metres

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const std::string nm = baseName(p);
        const std::vector<DepthMap> out = model->depths();   // copy: see [8] note
        if (out.empty() || out[0].empty())
        {
            std::cout << "[FAIL] " << nm << ": no depth map produced\n";
            ++failures;
            model->reset();
            continue;
        }
        const DepthMap dm = out[0];

        std::vector<float> ref(static_cast<std::size_t>(dm.width) * dm.height);
        const std::string binPath =
            locateRef(refDirs, stripExtension(nm) + ".depth.f32.bin");
        if (binPath.empty())
        {
            std::cout << "[SKIP] " << nm << " (no depth reference)\n";
            model->reset();
            continue;
        }
        if (!readBin(binPath, ref.data(), ref.size() * sizeof(float)))
        {
            std::cout << "[SKIP] " << nm << " (no reference " << binPath << ")\n";
            model->reset();
            continue;
        }

        float maxAbs = 0.f;
        float sumAbs = 0.f;
        std::size_t worst = 0;
        for (std::size_t i = 0; i < ref.size(); ++i)
        {
            const float d = std::fabs(dm.data[i] - ref[i]);
            sumAbs += d;
            if (d > maxAbs) { maxAbs = d; worst = i; }
        }
        const float meanAbs = sumAbs / static_cast<float>(ref.size());
        const bool geomOk = (dm.width == img.cols && dm.height == img.rows);
        const bool ok     = geomOk && maxAbs <= kDepthAbsTol;

        std::cout << "       " << nm << "  " << dm.width << "x" << dm.height
                  << "  ours=[" << *std::min_element(dm.data.begin(), dm.data.end())
                  << ", " << *std::max_element(dm.data.begin(), dm.data.end())
                  << "] m  ref=[" << *std::min_element(ref.begin(), ref.end())
                  << ", " << *std::max_element(ref.begin(), ref.end())
                  << "] m  max|d|=" << maxAbs << "  mean|d|=" << meanAbs
                  << "  worst@(" << (worst % dm.width) << "," << (worst / dm.width)
                  << ")\n";

        if (ok)
        {
            std::cout << "[PASS] " << nm << " depth matches the reference\n";
        }
        else
        {
            std::cout << "[FAIL] " << nm
                      << (geomOk ? "" : "  (output geometry != source image)")
                      << "  max|d| " << maxAbs << " > " << kDepthAbsTol << "\n";
            ++failures;
        }
        model->reset();
    }

    return failures;
}

// -----------------------------------------------------------------------------
//  [10] sem end to end, pixel by pixel against the Python reference
// -----------------------------------------------------------------------------
int runSemEndToEnd(const std::string& enginePath, int batch,
                   const std::vector<std::string>& images,
                   const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [10] end-to-end semantic (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(netName("sem"));
    auto* model = dynamic_cast<ISemanticSegmentor*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] yolov26_sem is not an ISemanticSegmentor\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 640;
    cfg.dstH      = 640;
    cfg.extras["num_class"] = "19";
    cfg.extras["scale"]     = "255.0";
    cfg.extras["pad_value"] = "114.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    // Both sides run the *same* algorithm on discrete class ids (crop +
    // nearest).  The only source of disagreement left is a logit tie flipping
    // inside the engine vs inside torch, so this must be ~0.  Keep a tiny
    // budget instead of demanding bit equality, and print the real number.
    const double kSemMaxMismatchRatio = 0.002;   // 0.2 % of the pixels

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const std::string nm = baseName(p);
        const std::vector<SemanticMask> out = model->semantics();   // copy
        if (out.empty() || out[0].empty())
        {
            std::cout << "[FAIL] " << nm << ": no semantic mask produced\n";
            ++failures;
            model->reset();
            continue;
        }
        const SemanticMask sm = out[0];

        std::vector<std::uint8_t> ref(static_cast<std::size_t>(sm.width) * sm.height);
        const std::string binPath =
            locateRef(refDirs, stripExtension(nm) + ".sem.u8.bin");
        if (binPath.empty())
        {
            std::cout << "[SKIP] " << nm << " (no semantic reference)\n";
            model->reset();
            continue;
        }
        if (!readBin(binPath, ref.data(), ref.size()))
        {
            std::cout << "[SKIP] " << nm << " (no reference " << binPath << ")\n";
            model->reset();
            continue;
        }

        std::size_t diff = 0;
        for (std::size_t i = 0; i < ref.size(); ++i)
        {
            if (sm.data[i] != ref[i]) { ++diff; }
        }
        const double ratio   = static_cast<double>(diff) / static_cast<double>(ref.size());
        const bool   geomOk  = (sm.width == img.cols && sm.height == img.rows);
        const bool   ok      = geomOk && ratio <= kSemMaxMismatchRatio;

        std::cout << "       " << nm << "  " << sm.width << "x" << sm.height
                  << "  mismatched=" << diff << " / " << ref.size()
                  << " (" << (ratio * 100.0) << " %)\n";

        if (ok)
        {
            std::cout << "[PASS] " << nm << " class map matches the reference\n";
        }
        else
        {
            std::cout << "[FAIL] " << nm
                      << (geomOk ? "" : "  (output geometry != source image)")
                      << "  mismatch " << (ratio * 100.0) << " % > "
                      << (kSemMaxMismatchRatio * 100.0) << " %\n";
            ++failures;
        }
        model->reset();
    }

    return failures;
}

// -----------------------------------------------------------------------------
//  [11] pose end to end, box by box and keypoint by keypoint
// -----------------------------------------------------------------------------
//  Reference bins (D:\Data\trt_exp\dump_pose_ref.py):
//      <stem>.pose.f32.bin
//          int32   n
//          float   boxes[n][6]     x1, y1, x2, y2, conf, cls   (source pixels)
//          float   kpts[n][17*3]   x, y, conf                  (source pixels)
//
//  The reference runs onnxruntime (SQUARE letterbox) + the official
//  nms.non_max_suppression (nc=1, so the 51 keypoint channels ride along as
//  "extra") + ops.scale_boxes + ops.scale_coords -- i.e. exactly the same four
//  steps the C++ side performs.  See the header note for why .pt is unusable.
//
//  Detection ORDER is not part of the contract (ours comes out of our own
//  greedy rounds, theirs out of TorchNMS), so the two sets are matched by IoU
//  first and only then compared value by value.  The count itself IS asserted:
//  a different survivor set means the NMS genuinely disagrees.
// -----------------------------------------------------------------------------
int runPoseEndToEnd(const std::string& enginePath, int batch,
                    const std::vector<std::string>& images,
                    const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [11] end-to-end pose (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    constexpr int kNumKpt  = 17;
    constexpr int kKptDim  = 3;
    constexpr int kKptVals = kNumKpt * kKptDim;

    // Both sides do the same discrete work (filter + NMS + linear map) on the
    // same network output; the only gap left is TRT fp32 vs torch CPU fp32
    // rounding.  Sub-pixel tolerances keep a geometry mistake (which would move
    // a keypoint by tens of pixels) impossible to hide.
    constexpr float kBoxTol  = 0.05f;   // pixels
    constexpr float kConfTol = 1e-4f;   // probability
    constexpr float kKptTol  = 0.05f;   // pixels
    constexpr float kMatchIou = 0.99f;  // boxes must be the same object

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(netName("pose"));
    auto* model = dynamic_cast<IKeypointer*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] yolov26_pose is not an IKeypointer\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 640;          // the engine's declared shape wins anyway
    cfg.dstH      = 640;
    cfg.extras["num_class"]   = "1";
    cfg.extras["num_kpts"]    = "17";
    cfg.extras["conf_thresh"] = "0.25";
    cfg.extras["iou_thresh"]  = "0.7";
    cfg.extras["top_k"]       = "300";
    cfg.extras["scale"]       = "255.0";
    cfg.extras["pad_value"]   = "114.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const std::string nm = baseName(p);

        // `keypoints()` is a reference into the model and reset() clears it --
        // take the copy before anything else (see the [8] note).
        const std::vector<std::vector<KeypointResult>> all = model->keypoints();
        if (all.empty())
        {
            std::cout << "[FAIL] " << nm << ": no pose output slot\n";
            ++failures;
            model->reset();
            continue;
        }
        const std::vector<KeypointResult> ours = all[0];

        const std::string binPath =
            locateRef(refDirs, stripExtension(nm) + ".pose.f32.bin");
        if (binPath.empty())
        {
            std::cout << "[SKIP] " << nm << " (no pose reference)\n";
            model->reset();
            continue;
        }
        std::ifstream in(binPath, std::ios::binary | std::ios::ate);
        if (!in)
        {
            std::cout << "[SKIP] " << nm << " (no reference " << binPath << ")\n";
            model->reset();
            continue;
        }
        const std::streamsize total = in.tellg();
        in.seekg(0);
        std::vector<char> raw(static_cast<std::size_t>(total));
        in.read(raw.data(), total);

        if (total < 4)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath
                      << " is truncated\n";
            ++failures;
            model->reset();
            continue;
        }
        int refN = 0;
        std::memcpy(&refN, raw.data(), sizeof(refN));
        const std::size_t want =
            4 + static_cast<std::size_t>(refN) * (6 + kKptVals) * sizeof(float);
        if (refN < 0 || static_cast<std::size_t>(total) != want)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath << " has "
                      << total << " bytes, expected " << want
                      << " (n=" << refN << ")\n";
            ++failures;
            model->reset();
            continue;
        }

        std::vector<float> refBox(static_cast<std::size_t>(refN) * 6);
        std::vector<float> refKpt(static_cast<std::size_t>(refN) * kKptVals);
        std::memcpy(refBox.data(), raw.data() + 4, refBox.size() * sizeof(float));
        std::memcpy(refKpt.data(), raw.data() + 4 + refBox.size() * sizeof(float),
                    refKpt.size() * sizeof(float));

        std::cout << "       " << nm << "  " << img.cols << "x" << img.rows
                  << "  ours=" << ours.size() << "  ref=" << refN << "\n";

        if (static_cast<int>(ours.size()) != refN)
        {
            std::cout << "[FAIL] " << nm << ": pose count got " << ours.size()
                      << " want " << refN
                      << " (NMS survivor set differs)\n";
            ++failures;
            model->reset();
            continue;
        }

        const auto iouXyxy = [](const float* a, const KeypointResult& b)
        {
            const float x1 = std::max(a[0], b.box.left);
            const float y1 = std::max(a[1], b.box.top);
            const float x2 = std::min(a[2], b.box.right);
            const float y2 = std::min(a[3], b.box.bottom);
            const float iw = std::max(0.f, x2 - x1);
            const float ih = std::max(0.f, y2 - y1);
            const float inter = iw * ih;
            const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
            const float areaB = (b.box.right - b.box.left) * (b.box.bottom - b.box.top);
            const float uni = areaA + areaB - inter + 1e-9f;
            return inter / uni;
        };

        std::vector<char>  used(ours.size(), 0);
        float maxBoxD = 0.f, maxConfD = 0.f, maxKptD = 0.f;
        int   unmatched = 0;

        for (int r = 0; r < refN; ++r)
        {
            const float* rb = refBox.data() + static_cast<std::size_t>(r) * 6;

            int   best    = -1;
            float bestIou = -1.f;
            for (int o = 0; o < static_cast<int>(ours.size()); ++o)
            {
                if (used[static_cast<std::size_t>(o)]) { continue; }
                const float v = iouXyxy(rb, ours[static_cast<std::size_t>(o)]);
                if (v > bestIou) { bestIou = v; best = o; }
            }
            if (best < 0 || bestIou < kMatchIou)
            {
                std::cout << "[FAIL] " << nm << ": reference pose #" << r
                          << " has no counterpart (best IoU " << bestIou << ")\n";
                ++unmatched;
                continue;
            }
            used[static_cast<std::size_t>(best)] = 1;
            const KeypointResult& o = ours[static_cast<std::size_t>(best)];

            const float ob[4] = {o.box.left, o.box.top, o.box.right, o.box.bottom};
            const char* bx[4] = {"x1", "y1", "x2", "y2"};
            for (int c = 0; c < 4; ++c)
            {
                const float d = std::fabs(ob[c] - rb[c]);
                if (d > maxBoxD) { maxBoxD = d; }
                if (d > kBoxTol)
                {
                    std::cout << "[FAIL] " << nm << ": pose #" << r << " box." << bx[c]
                              << " got " << ob[c] << " want " << rb[c]
                              << "  (|d| " << d << ")\n";
                    ++failures;
                }
            }
            const float dc = std::fabs(o.box.confidence - rb[4]);
            if (dc > maxConfD) { maxConfD = dc; }
            if (dc > kConfTol)
            {
                std::cout << "[FAIL] " << nm << ": pose #" << r << " conf got "
                          << o.box.confidence << " want " << rb[4] << "\n";
                ++failures;
            }
            if (o.box.label != static_cast<int>(rb[5]))
            {
                std::cout << "[FAIL] " << nm << ": pose #" << r << " class got "
                          << o.box.label << " want " << rb[5] << "\n";
                ++failures;
            }

            if (static_cast<int>(o.keypoints.size()) != kNumKpt)
            {
                std::cout << "[FAIL] " << nm << ": pose #" << r << " has "
                          << o.keypoints.size() << " keypoints, want " << kNumKpt
                          << "\n";
                ++failures;
                continue;
            }
            const float* rk = refKpt.data() + static_cast<std::size_t>(r) * kKptVals;
            for (int k = 0; k < kNumKpt; ++k)
            {
                const float* r3 = rk + k * kKptDim;
                const auto&  ok3 = o.keypoints[static_cast<std::size_t>(k)];
                const float  got[3] = {ok3.x, ok3.y, ok3.confidence};
                for (int c = 0; c < 3; ++c)
                {
                    const float d = std::fabs(got[c] - r3[c]);
                    if (d > maxKptD) { maxKptD = d; }
                    if (d > kKptTol)
                    {
                        std::cout << "[FAIL] " << nm << ": pose #" << r
                                  << " kpt[" << k << "]." << (c == 0 ? "x" : c == 1 ? "y" : "conf")
                                  << " got " << got[c] << " want " << r3[c] << "\n";
                        ++failures;
                    }
                }
            }
        }

        std::cout << "       max|d| box=" << maxBoxD << " px  conf=" << maxConfD
                  << "  kpt=" << maxKptD << " px  unmatched=" << unmatched << "\n";
        if (unmatched == 0)
        {
            std::cout << "[PASS] " << nm << " poses match the reference\n";
        }
        else
        {
            ++failures;
        }
        model->reset();
    }

    return failures;
}

// -----------------------------------------------------------------------------
//  [12] seg end to end, instance by instance, mask pixel by mask pixel
// -----------------------------------------------------------------------------
//  Reference bins (D:\Data\trt_exp\dump_seg_ref.py):
//      <stem>.seg.bin
//          int32   n
//          float   boxes[n][6]          x1, y1, x2, y2, conf, cls (source px)
//          uint8   masks[n][640*640]    network-space binary mask (0/1)
//
//  The reference is onnxruntime (SQUARE letterbox) + the official
//  nms.non_max_suppression (nc=80, so the 32 coefficient channels ride along as
//  "extra") + ops.process_mask(upsample=True) + ops.scale_boxes, with the
//  official `keep = masks.amax() > 0` filter applied -- i.e. the same steps the
//  C++ side performs, in the same order.
//
//  The mask lives in NETWORK coordinates (640x640, letterbox padding included),
//  exactly where ultralytics leaves Results.masks; the renderer is the one that
//  maps it back with the model's LetterBoxInfo.
// -----------------------------------------------------------------------------
int runSegEndToEnd(const std::string& enginePath, int batch,
                   const std::vector<std::string>& images,
                   const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [12] end-to-end segmentation (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    constexpr int   kNet      = 640;
    constexpr float kBoxTol   = 0.05f;   // pixels
    constexpr float kConfTol  = 1e-4f;   // probability
    constexpr float kMatchIou = 0.99f;   // boxes must be the same object
    // Both sides binarize the same logits at 0.0 and then crop. The only
    // disagreement left is a logit landing within 1 ULP of the threshold, so the
    // budget is tiny but not zero; the real number is printed either way.
    constexpr double kSegMaxMismatchRatio = 0.001;   // 0.1 % of all mask pixels

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(netName("seg"));
    auto* model = dynamic_cast<ISegmentor*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] yolov26_seg is not an ISegmentor\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 640;          // the engine's declared shape wins anyway
    cfg.dstH      = 640;
    cfg.extras["num_class"]   = "80";
    cfg.extras["extra"]       = "32";
    cfg.extras["conf_thresh"] = "0.25";
    cfg.extras["iou_thresh"]  = "0.7";
    cfg.extras["top_k"]       = "300";
    cfg.extras["scale"]       = "255.0";
    cfg.extras["pad_value"]   = "114.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const std::string nm = baseName(p);

        // `segmentations()` is a reference into the model and reset() clears it.
        const std::vector<std::vector<Segmentation>> all = model->segmentations();
        if (all.empty())
        {
            std::cout << "[FAIL] " << nm << ": no segmentation output slot\n";
            ++failures;
            model->reset();
            continue;
        }
        const std::vector<Segmentation> ours = all[0];

        const std::string binPath =
            locateRef(refDirs, stripExtension(nm) + ".seg.bin");
        if (binPath.empty())
        {
            std::cout << "[SKIP] " << nm << " (no seg reference)\n";
            model->reset();
            continue;
        }

        std::ifstream in(binPath, std::ios::binary | std::ios::ate);
        const std::streamsize total = in.tellg();
        in.seekg(0);
        std::vector<char> raw(static_cast<std::size_t>(total));
        in.read(raw.data(), total);

        int refN = 0;
        if (total < 4)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath
                      << " is truncated\n";
            ++failures;
            model->reset();
            continue;
        }
        std::memcpy(&refN, raw.data(), sizeof(refN));
        const std::size_t boxBytes  = static_cast<std::size_t>(refN) * 6 * sizeof(float);
        const std::size_t maskBytes = static_cast<std::size_t>(refN) * kNet * kNet;
        const std::size_t want      = 4 + boxBytes + maskBytes;
        if (refN < 0 || static_cast<std::size_t>(total) != want)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath << " has "
                      << total << " bytes, expected " << want << " (n=" << refN << ")\n";
            ++failures;
            model->reset();
            continue;
        }

        std::vector<float> refBox(static_cast<std::size_t>(refN) * 6);
        std::memcpy(refBox.data(), raw.data() + 4, boxBytes);
        const std::uint8_t* refMask =
            reinterpret_cast<const std::uint8_t*>(raw.data()) + 4 + boxBytes;

        std::cout << "       " << nm << "  " << img.cols << "x" << img.rows
                  << "  ours=" << ours.size() << "  ref=" << refN << "\n";

        if (static_cast<int>(ours.size()) != refN)
        {
            std::cout << "[FAIL] " << nm << ": instance count got " << ours.size()
                      << " want " << refN
                      << " (NMS survivor set or the all-zero-mask filter differs)\n";
            ++failures;
            model->reset();
            continue;
        }

        const auto iouXyxy = [](const float* a, const Segmentation& s)
        {
            const float x1 = std::max(a[0], s.box.left);
            const float y1 = std::max(a[1], s.box.top);
            const float x2 = std::min(a[2], s.box.right);
            const float y2 = std::min(a[3], s.box.bottom);
            const float inter = std::max(0.f, x2 - x1) * std::max(0.f, y2 - y1);
            const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
            const float areaB = (s.box.right - s.box.left) * (s.box.bottom - s.box.top);
            return inter / (areaA + areaB - inter + 1e-9f);
        };

        std::vector<char> used(ours.size(), 0);
        float  maxBoxD = 0.f, maxConfD = 0.f;
        double diffPx = 0.0, totalPx = 0.0;
        int    unmatched = 0, geomBad = 0;

        for (int r = 0; r < refN; ++r)
        {
            const float* rb = refBox.data() + static_cast<std::size_t>(r) * 6;

            int   best    = -1;
            float bestIou = -1.f;
            for (int o = 0; o < static_cast<int>(ours.size()); ++o)
            {
                if (used[static_cast<std::size_t>(o)]) { continue; }
                const float v = iouXyxy(rb, ours[static_cast<std::size_t>(o)]);
                if (v > bestIou) { bestIou = v; best = o; }
            }
            if (best < 0 || bestIou < kMatchIou)
            {
                std::cout << "[FAIL] " << nm << ": reference instance #" << r
                          << " has no counterpart (best IoU " << bestIou << ")\n";
                ++unmatched;
                continue;
            }
            used[static_cast<std::size_t>(best)] = 1;
            const Segmentation& s = ours[static_cast<std::size_t>(best)];

            const float ob[4] = {s.box.left, s.box.top, s.box.right, s.box.bottom};
            const char* bx[4] = {"x1", "y1", "x2", "y2"};
            for (int c = 0; c < 4; ++c)
            {
                const float d = std::fabs(ob[c] - rb[c]);
                if (d > maxBoxD) { maxBoxD = d; }
                if (d > kBoxTol)
                {
                    std::cout << "[FAIL] " << nm << ": instance #" << r << " box." << bx[c]
                              << " got " << ob[c] << " want " << rb[c] << "\n";
                    ++failures;
                }
            }
            const float dc = std::fabs(s.box.confidence - rb[4]);
            if (dc > maxConfD) { maxConfD = dc; }
            if (dc > kConfTol)
            {
                std::cout << "[FAIL] " << nm << ": instance #" << r << " conf got "
                          << s.box.confidence << " want " << rb[4] << "\n";
                ++failures;
            }
            if (s.box.label != static_cast<int>(rb[5]))
            {
                std::cout << "[FAIL] " << nm << ": instance #" << r << " class got "
                          << s.box.label << " want " << rb[5] << "\n";
                ++failures;
            }

            if (s.mask.width != kNet || s.mask.height != kNet ||
                s.mask.data == nullptr)
            {
                if (geomBad == 0)
                {
                    std::cout << "[FAIL] " << nm << ": mask geometry "
                              << s.mask.width << "x" << s.mask.height
                              << ", want " << kNet << "x" << kNet << "\n";
                }
                ++geomBad;
                ++failures;
                continue;
            }

            const std::uint8_t* rm =
                refMask + static_cast<std::size_t>(r) * kNet * kNet;
            const std::uint8_t* om = s.mask.data;
            double d = 0.0;
            for (std::size_t k = 0; k < static_cast<std::size_t>(kNet) * kNet; ++k)
            {
                // ours is 0/255, the reference is 0/1 -> compare the predicate.
                if ((om[k] != 0) != (rm[k] != 0)) { d += 1.0; }
            }
            diffPx  += d;
            totalPx += static_cast<double>(kNet) * kNet;
        }

        const double ratio = (totalPx > 0.0) ? diffPx / totalPx : 0.0;
        std::cout << "       max|d| box=" << maxBoxD << " px  conf=" << maxConfD
                  << "  mask mismatched=" << static_cast<long long>(diffPx)
                  << " / " << static_cast<long long>(totalPx)
                  << " (" << (ratio * 100.0) << " %)"
                  << "  unmatched=" << unmatched << "\n";

        if (unmatched == 0 && geomBad == 0 && ratio <= kSegMaxMismatchRatio)
        {
            std::cout << "[PASS] " << nm << " instance masks match the reference\n";
        }
        else
        {
            ++failures;
            std::cout << "[FAIL] " << nm << ": seg comparison failed (see above)\n";
        }
        model->reset();
    }

    return failures;
}

// -----------------------------------------------------------------------------
//  [13] obb end to end, box by box
// -----------------------------------------------------------------------------
//  Reference bins (D:\Data\trt_exp\dump_obb_ref.py):
//      <stem>.obb.f32.bin
//          int32   n
//          float   obb[n][7]   x, y, w, h, angle, conf, cls  (source pixels)
//
//  The reference runs onnxruntime (SQUARE letterbox) + nms.non_max_suppression
//  with rotated=True -- which means TorchNMS.fast_nms with batch_probiou, NOT
//  the greedy loop -- then OBBPredictor.construct_result (scale_boxes with
//  xywh=True, i.e. centre/pad and w/h/gain, with no clipping).
//
//  ORDER is part of the contract here and is compared positionally: the rotated
//  branch returns `sorted_idx[nonzero(...)]`, i.e. descending score, and the
//  kernel emits in exactly that order.  Matching by geometry instead would test
//  the thing under test.
//
//  Most of the shared sanity images produce zero detections (yolo26n-obb is a
//  DOTA aerial model), so the aerial-looking photos in the list are the ones
//  that actually exercise probiou -- 6406404.jpg has ~156 boxes.
// -----------------------------------------------------------------------------
int runObbEndToEnd(const std::string& enginePath, int batch,
                   const std::vector<std::string>& images,
                   const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [13] end-to-end oriented boxes (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    constexpr float kPosTol   = 0.05f;    // cx / cy, pixels
    constexpr float kSizeTol  = 0.05f;    // w / h, pixels
    // angle 是网络原始输出通道（弧度），TRT 与 CPU onnxruntime 的 kernel 选择
    // 不同，会让它漂 ~1e-5：实测 yolo26n-obb <= 7.7e-7，yolov11n-obb 在
    // 181 框的密集 DOTA 图上 <= 1.2e-5（框数仍完全一致）。5e-5 rad ≈ 0.003°，
    // 比任何真实的几何错误小三个数量级，用来兜住这类漂移。
    constexpr float kAngleTol = 5e-5f;    // radians
    constexpr float kConfTol  = 1e-4f;    // probability

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(netName("obb"));
    auto* model = dynamic_cast<IObbDetector*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] yolov26_obb is not an IObbDetector\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 640;          // the engine's declared shape wins anyway
    cfg.dstH      = 640;
    cfg.extras["num_class"]   = "15";
    cfg.extras["extra"]       = "1";
    cfg.extras["conf_thresh"] = "0.25";
    cfg.extras["iou_thresh"]  = "0.7";
    cfg.extras["top_k"]       = "300";
    cfg.extras["scale"]       = "255.0";
    cfg.extras["pad_value"]   = "114.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    int totalRef = 0;

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const std::string nm = baseName(p);

        const std::vector<std::vector<ObbDetection>> all = model->obbs();   // copy
        if (all.empty())
        {
            std::cout << "[FAIL] " << nm << ": no obb output slot\n";
            ++failures;
            model->reset();
            continue;
        }
        const std::vector<ObbDetection> ours = all[0];

        const std::string binPath =
            locateRef(refDirs, stripExtension(nm) + ".obb.f32.bin");
        if (binPath.empty())
        {
            std::cout << "[SKIP] " << nm << " (no obb reference)\n";
            model->reset();
            continue;
        }

        std::ifstream in(binPath, std::ios::binary | std::ios::ate);
        const std::streamsize total = in.tellg();
        in.seekg(0);
        std::vector<char> raw(static_cast<std::size_t>(total));
        in.read(raw.data(), total);

        int refN = 0;
        if (total < 4)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath
                      << " is truncated\n";
            ++failures;
            model->reset();
            continue;
        }
        std::memcpy(&refN, raw.data(), sizeof(refN));
        const std::size_t want = 4 + static_cast<std::size_t>(refN) * 7 * sizeof(float);
        if (refN < 0 || static_cast<std::size_t>(total) != want)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath << " has "
                      << total << " bytes, expected " << want << " (n=" << refN << ")\n";
            ++failures;
            model->reset();
            continue;
        }

        std::vector<float> ref(static_cast<std::size_t>(refN) * 7);
        std::memcpy(ref.data(), raw.data() + 4, ref.size() * sizeof(float));
        totalRef += refN;

        if (static_cast<int>(ours.size()) != refN)
        {
            std::cout << "[FAIL] " << nm << ": obb count got " << ours.size()
                      << " want " << refN
                      << " (Fast-NMS(probiou) survivor set differs)\n";
            ++failures;
            model->reset();
            continue;
        }

        float  maxPos = 0.f, maxSize = 0.f, maxAng = 0.f, maxConfD = 0.f;
        int    bad = 0;

        for (int i = 0; i < refN; ++i)
        {
            const float* r = ref.data() + static_cast<std::size_t>(i) * 7;
            const ObbDetection& o = ours[static_cast<std::size_t>(i)];

            const float dPos = std::max(std::fabs(o.cx - r[0]), std::fabs(o.cy - r[1]));
            const float dSize = std::max(std::fabs(o.w - r[2]), std::fabs(o.h - r[3]));
            const float dAng = std::fabs(o.angle - r[4]);
            const float dConf = std::fabs(o.confidence - r[5]);

            maxPos  = std::max(maxPos, dPos);
            maxSize = std::max(maxSize, dSize);
            maxAng  = std::max(maxAng, dAng);
            maxConfD = std::max(maxConfD, dConf);

            if (dPos > kPosTol || dSize > kSizeTol || dAng > kAngleTol ||
                dConf > kConfTol || o.label != static_cast<int>(r[6]))
            {
                if (bad < 4)
                {
                    std::cout << "[FAIL] " << nm << ": obb #" << i
                              << " ours=[" << o.cx << " " << o.cy << " " << o.w
                              << " " << o.h << " " << o.angle << " " << o.confidence
                              << " " << o.label << "]"
                              << " ref=[" << r[0] << " " << r[1] << " " << r[2]
                              << " " << r[3] << " " << r[4] << " " << r[5]
                              << " " << static_cast<int>(r[6]) << "]\n";
                }
                ++bad;
            }
        }

        std::cout << "       " << nm << "  " << img.cols << "x" << img.rows
                  << "  n=" << refN
                  << "  max|d| centre=" << maxPos << " size=" << maxSize
                  << " angle=" << maxAng << " conf=" << maxConfD << "\n";

        if (bad == 0)
        {
            std::cout << "[PASS] " << nm << " oriented boxes match the reference\n";
        }
        else
        {
            std::cout << "[FAIL] " << nm << ": " << bad << " box(es) out of tolerance\n";
            ++failures;
        }
        model->reset();
    }

    std::cout << "       (total reference boxes across the list: " << totalRef << ")\n";
    return failures;
}

// -----------------------------------------------------------------------------
//  [14] det end to end, box by box
// -----------------------------------------------------------------------------
//  Reference bins (D:\Data\trt_exp\ref_dump.py --task det):
//      <stem>.det.f32.bin
//          int32   n
//          float   boxes[n][6]     x1, y1, x2, y2, conf, cls  (source pixels)
//
//  The reference runs onnxruntime (SQUARE letterbox) + the official
//  nms.non_max_suppression (nc = num_class, max_det = 300) + ops.scale_boxes --
//  exactly the steps the C++ side performs, in the same order.  See the header
//  of dump_seg_ref.py for why a `.pt` model is not a usable reference here.
//
//  ORDER is not part of the contract (ours comes out of our own greedy rounds,
//  theirs out of TorchNMS), so the two sets are matched by IoU first and only
//  then compared value by value.  The COUNT is asserted up front: a different
//  survivor set means the NMS genuinely disagrees.
// -----------------------------------------------------------------------------
int runDetEndToEnd(const std::string& enginePath, int batch,
                   const std::vector<std::string>& images,
                   const std::vector<std::string>& refDirs)
{
    int failures = 0;
    std::cout << "\n---- [14] end-to-end detection (" << images.size()
              << " image(s), batch=" << batch << ") ----\n";

    constexpr float kBoxTol   = 0.05f;   // pixels
    constexpr float kConfTol  = 1e-4f;   // probability
    constexpr float kMatchIou = 0.99f;   // same object, not merely overlapping

    std::unique_ptr<IModel> base = ModelRegistry::instance().create(g_netPrefix);
    auto* model = dynamic_cast<IDetector*>(base.get());
    if (model == nullptr)
    {
        std::cout << "[FAIL] " << g_netPrefix << " is not an IDetector\n";
        return 1;
    }

    ModelConfig cfg;
    cfg.engine    = enginePath;
    cfg.batchSize = batch;
    cfg.dstW      = 640;          // the engine's declared shape wins anyway
    cfg.dstH      = 640;
    cfg.extras["num_class"]   = "80";
    cfg.extras["conf_thresh"] = "0.25";
    cfg.extras["iou_thresh"]  = "0.7";
    cfg.extras["top_k"]       = "300";
    cfg.extras["scale"]       = "255.0";
    cfg.extras["pad_value"]   = "114.0";

    model->init(cfg);
    std::cout << "[PASS] engine loaded: " << enginePath << "\n";

    for (const std::string& p : images)
    {
        const cv::Mat img = cv::imread(p, cv::IMREAD_COLOR);
        if (img.empty())
        {
            std::cout << "[FAIL] cannot read " << p << "\n";
            ++failures;
            continue;
        }

        model->setBatch(batchFromMat(img, batch));
        model->preprocess();
        model->infer();
        model->postprocess();

        const std::string nm = baseName(p);

        // `detections()` is a reference into the model and reset() clears it.
        const std::vector<std::vector<Detection>> all = model->detections();
        if (all.empty())
        {
            std::cout << "[FAIL] " << nm << ": no detection output slot\n";
            ++failures;
            model->reset();
            continue;
        }
        const std::vector<Detection> ours = all[0];

        const std::string binPath =
            locateRef(refDirs, stripExtension(nm) + ".det.f32.bin");
        if (binPath.empty())
        {
            std::cout << "[SKIP] " << nm << " (no det reference)\n";
            model->reset();
            continue;
        }

        std::ifstream in(binPath, std::ios::binary | std::ios::ate);
        const std::streamsize total = in.tellg();
        in.seekg(0);
        std::vector<char> raw(static_cast<std::size_t>(total));
        in.read(raw.data(), total);

        if (total < 4)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath
                      << " is truncated\n";
            ++failures;
            model->reset();
            continue;
        }
        int refN = 0;
        std::memcpy(&refN, raw.data(), sizeof(refN));
        const std::size_t want = 4 + static_cast<std::size_t>(refN) * 6 * sizeof(float);
        if (refN < 0 || static_cast<std::size_t>(total) != want)
        {
            std::cout << "[FAIL] " << nm << ": reference " << binPath << " has "
                      << total << " bytes, expected " << want
                      << " (n=" << refN << ")\n";
            ++failures;
            model->reset();
            continue;
        }

        std::vector<float> refBox(static_cast<std::size_t>(refN) * 6);
        std::memcpy(refBox.data(), raw.data() + 4, refBox.size() * sizeof(float));

        std::cout << "       " << nm << "  " << img.cols << "x" << img.rows
                  << "  ours=" << ours.size() << "  ref=" << refN << "\n";

        if (static_cast<int>(ours.size()) != refN)
        {
            std::cout << "[FAIL] " << nm << ": detection count got " << ours.size()
                      << " want " << refN
                      << " (NMS survivor set differs)\n";
            ++failures;
            model->reset();
            continue;
        }

        const auto iouXyxy = [](const float* a, const Detection& b)
        {
            const float x1 = std::max(a[0], b.left);
            const float y1 = std::max(a[1], b.top);
            const float x2 = std::min(a[2], b.right);
            const float y2 = std::min(a[3], b.bottom);
            const float iw = std::max(0.f, x2 - x1);
            const float ih = std::max(0.f, y2 - y1);
            const float inter = iw * ih;
            const float areaA = (a[2] - a[0]) * (a[3] - a[1]);
            const float areaB = (b.right - b.left) * (b.bottom - b.top);
            const float uni = areaA + areaB - inter + 1e-9f;
            return inter / uni;
        };

        std::vector<char> used(ours.size(), 0);
        float maxBoxD = 0.f, maxConfD = 0.f;
        int   unmatched = 0;

        for (int r = 0; r < refN; ++r)
        {
            const float* rb = refBox.data() + static_cast<std::size_t>(r) * 6;

            int   best = -1;
            float bestIou = 0.f;
            for (std::size_t i = 0; i < ours.size(); ++i)
            {
                if (used[i] != 0) { continue; }
                const float v = iouXyxy(rb, ours[i]);
                if (v > bestIou) { bestIou = v; best = static_cast<int>(i); }
            }
            if (best < 0 || bestIou < kMatchIou)
            {
                if (unmatched < 4)
                {
                    std::cout << "[FAIL] " << nm << ": no match for ref box #" << r
                              << " (best IoU " << bestIou << ")\n";
                }
                ++unmatched;
                continue;
            }
            used[static_cast<std::size_t>(best)] = 1;

            const Detection& o = ours[static_cast<std::size_t>(best)];
            const float dBox = std::max(
                std::max(std::fabs(o.left - rb[0]), std::fabs(o.top - rb[1])),
                std::max(std::fabs(o.right - rb[2]), std::fabs(o.bottom - rb[3])));
            const float dConf = std::fabs(o.confidence - rb[4]);
            maxBoxD  = std::max(maxBoxD, dBox);
            maxConfD = std::max(maxConfD, dConf);

            if (dBox > kBoxTol || dConf > kConfTol ||
                o.label != static_cast<int>(rb[5]))
            {
                if (unmatched < 4)
                {
                    std::cout << "[FAIL] " << nm << ": box #" << r
                              << " ours=[" << o.left << " " << o.top << " "
                              << o.right << " " << o.bottom << " "
                              << o.confidence << " " << o.label << "]"
                              << " ref=[" << rb[0] << " " << rb[1] << " "
                              << rb[2] << " " << rb[3] << " " << rb[4] << " "
                              << static_cast<int>(rb[5]) << "]\n";
                }
                ++unmatched;
            }
        }

        std::cout << "       " << nm << "  n=" << refN
                  << "  max|d| box=" << maxBoxD << " conf=" << maxConfD << "\n";

        if (unmatched == 0)
        {
            std::cout << "[PASS] " << nm << " detections match the reference\n";
        }
        else
        {
            std::cout << "[FAIL] " << nm << ": " << unmatched
                      << " box(es) unmatched or out of tolerance\n";
            ++failures;
        }
        model->reset();
    }

    return failures;
}

}  // namespace

// =============================================================================
//  Command line
// -----------------------------------------------------------------------------
//  Two accepted forms:
//
//    (legacy)  test_yolov26_tasks <cls.trt> <image> [image...]
//              -> equivalent to --cls <cls.trt> plus the trailing images
//
//    (flags)   test_yolov26_tasks [--det <e.trt>] [--cls <e.trt>]
//                                 [--depth <e.trt>] [--sem <e.trt>]
//                                 [--pose <e.trt>] [--seg <e.trt>]
//                                 [--obb <e.trt>]
//                                 [--refdir <dir>[,<dir>...]] <image> [image...]
//
//  --net <prefix> selects the model family: "yolov26" (default), "yolov11",
//  "yolov8", "yolov9" or "yolov10".  All of them share one implementation
//  (every family derives from YOLOV26), so the same assertions cover all; only
//  the registered names, the task set and the engines differ.
//
//  --refdir is repeatable / comma-separated; each optional section picks the
//  first directory that actually holds its file.
//
//  Images accumulate into one shared list: every engine that was named runs
//  over all of them.  So this is valid:
//      --cls a.trt --depth b.trt --sem c.trt --pose d.trt --seg e.trt --obb f.trt x.jpg y.jpg
//
//  With no image at all every optional section is skipped and the test still
//  checks sections [1]..[7].
// =============================================================================
int main(int argc, char** argv)
{
    std::cout << "============ test_yolov26_tasks ============\n";

    std::string clsEngine, depthEngine, semEngine, poseEngine, segEngine, obbEngine;
    std::string detEngine;
    std::vector<std::string> refDirs;
    std::vector<std::string> images;

    for (int i = 1; i < argc; ++i)
    {
        const std::string a = argv[i];
        if (a == "--det" || a == "--cls" || a == "--depth" || a == "--sem" ||
            a == "--pose" || a == "--seg" || a == "--obb")
        {
            if (i + 1 >= argc)
            {
                std::cout << "[FAIL] " << a << " needs an engine path\n";
                ++g_failures;
                break;
            }
            if (a == "--det")   { detEngine   = argv[++i]; }
            if (a == "--depth") { depthEngine = argv[++i]; }
            if (a == "--sem")   { semEngine   = argv[++i]; }
            if (a == "--pose")  { poseEngine  = argv[++i]; }
            if (a == "--seg")   { segEngine   = argv[++i]; }
            if (a == "--obb")   { obbEngine   = argv[++i]; }
        }
        else if (a == "--net")
        {
            if (i + 1 >= argc)
            {
                std::cout << "[FAIL] --net needs a model-name prefix "
                             "(yolov26 / yolov11 / yolov8 / yolov9 / yolov10)\n";
                ++g_failures;
                break;
            }
            g_netPrefix = argv[++i];
        }
        else if (a == "--refdir")
        {
            if (i + 1 >= argc)
            {
                std::cout << "[FAIL] --refdir needs a directory\n";
                ++g_failures;
                break;
            }
            // Repeatable, and a single value may also be comma-separated:
            // different tasks are dumped into different directories.
            std::string v = argv[++i];
            std::size_t pos = 0;
            while (true)
            {
                const std::size_t comma = v.find(',', pos);
                std::string one = v.substr(pos, comma == std::string::npos
                                                    ? std::string::npos
                                                    : comma - pos);
                // A trailing '/' is harmless but makes the joined paths ugly.
                while (!one.empty() && (one.back() == '/' || one.back() == '\\'))
                {
                    one.pop_back();
                }
                if (!one.empty()) { refDirs.push_back(one); }
                if (comma == std::string::npos) { break; }
                pos = comma + 1;
            }
        }
        else if (a.size() > 1 && a[0] == '-')
        {
            std::cout << "[FAIL] unknown option " << a << "\n";
            ++g_failures;
            break;
        }
        else if (images.empty() && a.find(".trt") != std::string::npos &&
                 clsEngine.empty())
        {
            // Legacy positional form: the first bare *.trt argument is cls.
            clsEngine = a;
        }
        else
        {
            images.push_back(a);
        }
    }

    // ---- [1] 注册中心 ----
    std::cout << "\n---- [1] registry (prefix '" << g_netPrefix << "') ----\n";
    // Each family registers only the tasks it actually implements.  det carries
    // no task suffix, so the prefix itself is a registered name:
    //   yolov26  det + cls / depth / obb / pose / seg / sem
    //   yolov11  det + cls / obb / pose / seg        (no depth / sem)
    //   yolov8   det + cls / obb / pose / seg
    //   yolov9   det + seg                           (no cls / obb / pose)
    //   yolov10  det only
    std::vector<std::string> names;
    names.push_back(g_netPrefix);
    if (g_netPrefix == "yolov26")
    {
        for (const char* t : {"cls", "depth", "obb", "pose", "seg", "sem"})
        { names.push_back(netName(t)); }
    }
    else if (g_netPrefix == "yolov11" || g_netPrefix == "yolov8")
    {
        for (const char* t : {"cls", "obb", "pose", "seg"})
        { names.push_back(netName(t)); }
    }
    else if (g_netPrefix == "yolov9")
    {
        names.push_back(netName("seg"));
    }
    else if (g_netPrefix != "yolov10")
    {
        std::cout << "[FAIL] unknown --net prefix '" << g_netPrefix << "'\n";
        ++g_failures;
    }
    {
        const std::vector<std::string> all = ModelRegistry::instance().names();
        for (const std::string& n : names)
        {
            const bool ok =
                std::find(all.begin(), all.end(), n) != all.end();
            std::cout << (ok ? "[PASS] " : "[FAIL] ") << "registered: " << n << "\n";
            if (!ok) { ++g_failures; }
        }
    }

    // ---- [2] 引擎缺失必须显式抛 ----
    std::cout << "\n---- [2] init must fail loudly when the engine is missing ----\n";
    for (const std::string& n : names)
    {
        std::unique_ptr<IModel> m = ModelRegistry::instance().create(n);
        if (!m)
        {
            std::cout << "[FAIL] create failed for " << n << "\n";
            ++g_failures;
            continue;
        }
        ModelConfig cfg;
        cfg.engine   = "/definitely/not/exist/yolov26_12345.trt";
        cfg.batchSize = 1;
        cfg.dstH = 640;
        cfg.dstW = 640;
        bool threw = false;
        try { m->init(cfg); }
        catch (const std::exception& e)
        {
            threw = true;
            std::cout << "       " << n << " -> " << e.what() << "\n";
        }
        std::cout << (threw ? "[PASS] " : "[FAIL] ") << n
                  << " threw on a missing engine\n";
        if (!threw) { ++g_failures; }
    }

    testTopK();
    testScaleMaskGeometry();
    testNearestIndices();
    testClsGeometry();

    // ---- [8] 可选：端到端分类 ----
    if (!clsEngine.empty() && !images.empty())
    {
        g_failures += runClsEndToEnd(clsEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [8] end-to-end skipped"
                     " (needs --cls <cls.trt> <image...>)\n";
    }

    // ---- [9] 可选：端到端深度 ----
    if (!depthEngine.empty() && !images.empty())
    {
        g_failures += runDepthEndToEnd(depthEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [9] end-to-end skipped"
                     " (needs --depth <depth.trt> <image...>)\n";
    }

    // ---- [10] 可选：端到端语义分割 ----
    if (!semEngine.empty() && !images.empty())
    {
        g_failures += runSemEndToEnd(semEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [10] end-to-end skipped"
                     " (needs --sem <sem.trt> <image...>)\n";
    }

    // ---- [11] 可选：端到端人体关键点 ----
    if (!poseEngine.empty() && !images.empty())
    {
        g_failures += runPoseEndToEnd(poseEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [11] end-to-end skipped"
                     " (needs --pose <pose.trt> <image...>)\n";
    }

    // ---- [12] 可选：端到端实例分割 ----
    if (!segEngine.empty() && !images.empty())
    {
        g_failures += runSegEndToEnd(segEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [12] end-to-end skipped"
                     " (needs --seg <seg.trt> <image...>)\n";
    }

    // ---- [13] 可选：端到端旋转框 ----
    if (!obbEngine.empty() && !images.empty())
    {
        g_failures += runObbEndToEnd(obbEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [13] end-to-end skipped"
                     " (needs --obb <obb.trt> <image...>)\n";
    }

    // ---- [14] 可选：端到端目标检测 ----
    if (!detEngine.empty() && !images.empty())
    {
        g_failures += runDetEndToEnd(detEngine, 2, images, refDirs);
    }
    else
    {
        std::cout << "\n---- [14] end-to-end detection skipped"
                     " (needs --det <det.trt> <image...>)\n";
    }

    std::cout << "\n============ summary ============\n";
    if (g_failures == 0) { std::cout << "  ALL PASS\n"; return 0; }
    std::cout << "  FAILURES: " << g_failures << "\n";
    return 1;
}

