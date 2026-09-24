// =============================================================================
//  test/test_kernels/test_kernels.cpp
// -----------------------------------------------------------------------------
//  kernels 测试（不需 engine，只用 CUDA）：
//    [1] resizeLetterbox：1x1x1 src -> 2x2x1 dst，全零仿射，检查 padding + 采样
//    [2] bgrToNchwNormalized：BGR->RGB + 归一化 + NCHW 布局
//    [3] transposeAnchors：[B, C, N] -> [B, N, C]
//    [4] decodeYoloV8Head：构造 1 个高置信 anchor，检查输出
//    [5] nmsFast：两个重叠框，检查 keep 标志
// =============================================================================
#include "trt_alpha/core/device_buffer.hpp"
#include "trt_alpha/core/pinned_buffer.hpp"
#include "trt_alpha/core/cuda_stream.hpp"
#include "trt_alpha/kernels/postprocess.hpp"
#include "trt_alpha/kernels/preprocess.hpp"

#include <cmath>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <vector>

using trt_alpha::core::CudaStream;
using trt_alpha::core::DeviceBuffer;
using trt_alpha::core::PinnedBuffer;
using trt_alpha::kernels::AffineMat;
using trt_alpha::kernels::YoloDecodeParams;
using trt_alpha::kernels::kObjectWidth;

namespace {

int g_failures = 0;

void check(bool cond, const char* what)
{
    if (cond) { std::cout << "[PASS] " << what << "\n"; }
    else      { std::cout << "[FAIL] " << what << "\n"; ++g_failures; }
}

}  // namespace

int main()
{
    std::cout << "=== kernels tests ===\n";
    CudaStream stream;

    // ---------------------------------------------------------------
    // [1] resizeLetterbox：1x1 -> 2x2
    // ---------------------------------------------------------------
    {
        // src: 1x1 uint8 BGR = [10, 20, 30]
        // dst: 2x2 float BGR，仿射为恒等（放大 1:1，无 padding）
        constexpr int SRC_W = 2, SRC_H = 2;
        constexpr int DST_W = 4, DST_H = 4;

        std::vector<std::uint8_t> srcHost(SRC_W * SRC_H * 3);
        for (int i = 0; i < SRC_W * SRC_H * 3; ++i)
        {
            srcHost[i] = static_cast<std::uint8_t>(i);
        }

        DeviceBuffer srcDev(srcHost.size());
        DeviceBuffer dstDev(DST_W * DST_H * 3 * sizeof(float));

        cudaMemcpy(srcDev.data(), srcHost.data(), srcHost.size(),
                   cudaMemcpyHostToDevice);

        // 仿射：dst(4x4) -> src(2x2)，scale = 0.5
        AffineMat m;
        m.v0 = 0.5f; m.v1 = 0.f; m.v2 = 0.f;
        m.v3 = 0.f;  m.v4 = 0.5f; m.v5 = 0.f;

        trt_alpha::kernels::resizeLetterbox(
            stream.get(), 1,
            static_cast<const std::uint8_t*>(srcDev.data()), SRC_W, SRC_H,
            static_cast<float*>(dstDev.data()), DST_W, DST_H,
            114.f, m);
        stream.synchronize();

        std::vector<float> dstHost(DST_W * DST_H * 3);
        cudaMemcpy(dstHost.data(), dstDev.data(), dstHost.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // 左上角 (0,0) 采样 src(0,0) = [0,1,2]
        check(dstHost[0] == 0.f && dstHost[1] == 1.f && dstHost[2] == 2.f,
              "[1] top-left pixel sampled correctly");
    }

    // ---------------------------------------------------------------
    // [2] bgrToNchwNormalized
    // ---------------------------------------------------------------
    {
        constexpr int W = 2, H = 2;
        // src: 1x(2x2x3) float BGR HWC
        const float srcHWC[12] = {
            // pixel (0,0): B=1, G=2, R=3
            1.f, 2.f, 3.f,
            // pixel (1,0)
            4.f, 5.f, 6.f,
            // pixel (0,1)
            7.f, 8.f, 9.f,
            // pixel (1,1)
            10.f, 11.f, 12.f,
        };
        DeviceBuffer srcDev(sizeof(srcHWC));
        DeviceBuffer dstDev(sizeof(srcHWC));
        cudaMemcpy(srcDev.data(), srcHWC, sizeof(srcHWC), cudaMemcpyHostToDevice);

        const float scale = 1.f;   // 不做 /255
        const float mean[3] = {0.f, 0.f, 0.f};
        const float std_[3] = {1.f, 1.f, 1.f};

        trt_alpha::kernels::bgrToNchwNormalized(
            stream.get(), 1,
            static_cast<const float*>(srcDev.data()),
            static_cast<float*>(dstDev.data()),
            W, H, scale, mean, std_);
        stream.synchronize();

        std::vector<float> dstHost(sizeof(srcHWC) / sizeof(float));
        cudaMemcpy(dstHost.data(), dstDev.data(), sizeof(srcHWC),
                   cudaMemcpyDeviceToHost);

        // NCHW 输出：channel 0 = R, channel 1 = G, channel 2 = B
        // pixel (0,0) 的 R = 3，在 channel 0 的第一个位置 = dst[0]
        // pixel (0,0) 的 B = 1，在 channel 2 的第一个位置 = dst[2*4+0] = dst[8]
        check(dstHost[0] == 3.f,   "[2] channel 0 (R) first pixel == 3");
        check(dstHost[4] == 2.f,   "[2] channel 1 (G) first pixel == 2");
        check(dstHost[8] == 1.f,   "[2] channel 2 (B) first pixel == 1");
    }

    // ---------------------------------------------------------------
    // [3] transposeAnchors：[B=1, srcRow=2, N=3] -> [B=1, N=3, srcRow=2]
    // ---------------------------------------------------------------
    {
        const float src[6] = {1.f, 2.f, 3.f,  10.f, 20.f, 30.f};   // 2 行 x 3 anchors
        DeviceBuffer srcDev(sizeof(src));
        DeviceBuffer dstDev(sizeof(src));
        cudaMemcpy(srcDev.data(), src, sizeof(src), cudaMemcpyHostToDevice);

        trt_alpha::kernels::transposeAnchors(
            stream.get(), 1,
            static_cast<const float*>(srcDev.data()), 2, 3,
            static_cast<float*>(dstDev.data()));
        stream.synchronize();

        float dst[6];
        cudaMemcpy(dst, dstDev.data(), sizeof(dst), cudaMemcpyDeviceToHost);

        // dst = [1, 10, 2, 20, 3, 30]
        check(dst[0] == 1.f && dst[1] == 10.f, "[3] row 0");
        check(dst[2] == 2.f && dst[3] == 20.f, "[3] row 1");
        check(dst[4] == 3.f && dst[5] == 30.f, "[3] row 2");
    }

    // ---------------------------------------------------------------
    // [4] decodeYoloV8Head：一个高置信 anchor
    // ---------------------------------------------------------------
    {
        // srcRow = 4 + 2 类 = 6。1 个 anchor。cx=10, cy=20, w=4, h=8, [cls0=0.1, cls1=0.9]
        const float src[6] = {10.f, 20.f, 4.f, 8.f, 0.1f, 0.9f};
        DeviceBuffer srcDev(sizeof(src));
        cudaMemcpy(srcDev.data(), src, sizeof(src), cudaMemcpyHostToDevice);

        const int topK = 10;
        const int dstRow = kObjectWidth;
        const int dstElems = 1 + topK * dstRow;   // 1 batch, count + slots
        DeviceBuffer dstDev(dstElems * sizeof(float));
        cudaMemset(dstDev.data(), 0, dstElems * sizeof(float));

        YoloDecodeParams p;
        p.batch = 1;
        p.numClasses = 2;
        p.topK = topK;
        p.confThreshold = 0.25f;
        p.iouThreshold = 0.45f;

        trt_alpha::kernels::decodeYoloV8Head(
            stream.get(), p,
            static_cast<const float*>(srcDev.data()), 1,
            static_cast<float*>(dstDev.data()));
        stream.synchronize();

        std::vector<float> dst(dstElems);
        cudaMemcpy(dst.data(), dstDev.data(), dstElems * sizeof(float),
                   cudaMemcpyDeviceToHost);

        check(dst[0] == 1.f,          "[4] count == 1");
        check(dst[1] == 8.f,          "[4] left == cx - w/2 == 8");
        check(dst[2] == 16.f,         "[4] top  == cy - h/2 == 16");
        check(dst[3] == 12.f,         "[4] right == cx + w/2 == 12");
        check(dst[4] == 24.f,         "[4] bottom == cy + h/2 == 24");
        check(std::fabs(dst[5] - 0.9f) < 1e-5f, "[4] confidence == 0.9");
        check(dst[6] == 1.f,          "[4] label == 1");
        check(dst[7] == 1.f,          "[4] keep == 1");
    }

    // ---------------------------------------------------------------
    // [5] nmsFast：两个重叠框（IoU 高）-> 只保留高分的
    // ---------------------------------------------------------------
    {
        const int topK = 4;
        const int dstRow = kObjectWidth;
        const int dstElems = 1 + topK * dstRow;

        std::vector<float> host(dstElems, 0.f);
        host[0] = 2.f;   // count = 2
        // 框 0：conf 0.9，label 0，与框 1 高度重叠
        host[1] = 10.f; host[2] = 10.f; host[3] = 30.f; host[4] = 30.f;
        host[5] = 0.9f; host[6] = 0.f;  host[7] = 1.f;
        // 框 1：conf 0.8，label 0，与框 0 重叠
        host[8]  = 11.f; host[9]  = 11.f; host[10] = 31.f; host[11] = 31.f;
        host[12] = 0.8f; host[13] = 0.f;  host[14] = 1.f;

        DeviceBuffer dev(dstElems * sizeof(float));
        cudaMemcpy(dev.data(), host.data(), host.size() * sizeof(float),
                   cudaMemcpyHostToDevice);

        YoloDecodeParams p;
        p.batch = 1;
        p.topK = topK;
        p.iouThreshold = 0.45f;

        trt_alpha::kernels::nmsFast(stream.get(), p,
                                    static_cast<float*>(dev.data()), dstRow);
        stream.synchronize();

        std::vector<float> out(dstElems);
        cudaMemcpy(out.data(), dev.data(), out.size() * sizeof(float),
                   cudaMemcpyDeviceToHost);

        // 框 0（高分）应保留 keep=1；框 1（低分且重叠）应被淘汰 keep=0
        check(out[7] == 1.f,  "[5] high-conf box kept (keep=1)");
        check(out[14] == 0.f, "[5] low-conf overlapping box suppressed (keep=0)");
    }

    std::cout << "====================\n";
    if (g_failures == 0) { std::cout << "ALL PASS\n"; return 0; }
    std::cout << g_failures << " FAILED\n";
    return 1;
}