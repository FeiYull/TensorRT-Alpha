<p align="center">
  <img src=".github/trt_alpha_pipeline.svg" alt="TensorRT-Alpha inference pipeline" width="100%">
</p>


<p align="center">
  <img alt="Windows" src="https://img.shields.io/badge/Windows-0078D6?logo=data:image/svg+xml;base64,PHN2ZyB4bWxucz0iaHR0cDovL3d3dy53My5vcmcvMjAwMC9zdmciIHZpZXdCb3g9IjAgMCAyNCAyNCI%2BPHBhdGggZmlsbD0iI2ZmZmZmZiIgZD0iTTAgMy40NDkgOS43NSAyLjF2OS40NTFIMG0xMC45NDktOS42MDJMMjQgMHYxMS40SDEwLjk0OU0wIDEyLjZoOS43NXY5LjQ1MUwwIDIwLjY5OU0xMC45NDkgMTIuNkgyNFYyNGwtMTIuOS0xLjgwMSIvPjwvc3ZnPg%3D%3D&logoColor=white">
  <img alt="Linux" src="https://img.shields.io/badge/Linux-FCC624?logo=linux&logoColor=black">
  <img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-10.x-76B900">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.x-76B900">
  <img alt="C++" src="https://img.shields.io/badge/C++-17-00599C">
</p>

<p align="center">
  <a href="README_en.md">English</a> | 简体中文
</p>

## 简介

<details open>
<summary><b>特色简介</b></summary>

C++ 推理仓库，基于 **TensorRT + CUDA**，**手写全部 CUDA C 算子**：

- **端到端 GPU 推理**：数据源 → 组批 → GPU 前处理 → 推理 → GPU 后处理 → 渲染 / 保存，全程零 CPU 往返。
- **多 batch 推理**：`1 engine + N context` 并发，每个 worker 独占一条 `cudaStream`。
- **无损复刻原始 PyTorch 推理**：对齐 ultralytics 官方实现，见 [严格精度对齐](#严格精度对齐)。
- **多种输入源**：图片、图片目录、视频流（RTSP / RTMP / HTTP(S) / UDP）、webcam、URL —— 一个 `--source` 自动判别。

</details>

<details>
<summary><b>近期计划</b></summary>



- **TODO**：Nvidia Nano 等多款边缘设备（计划测试中）
- **TODO**：支持 GStreamer —— 可拆卸 OpenCV，替换为 GStreamer 数据源
- **TODO**：`trt_alpha build` —— ONNX → engine 一键序列化
- **TODO**：展示更细粒度的性能测试 —— 用 **Nsight Systems** 剖析整条流水线的时间线（H2D / 推理 / D2H / 后处理 / 渲染的排布与空泡），用 **Nsight Compute** 逐个 CUDA kernel 剖析（占用率、访存、瓶颈定位）
- **TODO**：更多模型系列

</details>

<details>
<summary><b>更新日志</b></summary>

- 2026.10.18 🚀 YOLOV11 / YOLOV8 / YOLOV9 / YOLOV10 接入，全部走 YOLO26 的 fused 前后处理：yolov11 五任务；yolov8 重写五任务（旧的动态引擎保留为 `yolov8*_old` 供 A/B 对拍）；yolov9 det + seg；yolov10 det（仅 one2many）。与 python 逐框对拍实测：框 ≤3.4e-4 px、conf ≤2e-6、mask 逐像素 0 差异
- 2026.10.16 💖 release v1.0
- 2026.10.15 🐧 Linux 支持：GCC 13 移植、行尾 / 编辑器策略归一、引擎序列化脚本；Windows 运行时 DLL 自动部署；参数解析与源探测硬化 + `run` 全参数矩阵测试脚本
- 2026.10.14 🧠 YOLO26 全任务族补齐：cls / depth / obb / pose / seg / sem；首页架构图
- 2026.10.12 🎯 YOLO26-det 接入：后处理算子族（decode / NMS / 融合 / 向量化），命名统一 `yolo26` → `yolov26`
- 2026.10.11 ⚡ 逐位对齐 OpenCV 的 CUDA 预处理算子族（resize / copyMakeBorder / cvtColor / hwc2chw / normalize + 融合核）
- 2026.10.10 📦 legacy CUDA 核归档到 `kernels/legacy/`，公开头注释全面英译
- 2026.10.07 🔍 代码评审硬化：batch 上限守卫、丢帧策略、覆盖输入守卫；CTest 入场（32 项）
- 2026.10.06 🛡️ 运行期配置透视（`logConfigBox` + 死键 `[unused]` 判定）；URL 流（rtsp / rtmp / http(s) / udp）+ 目录源 + 输出目录规则
- 2026.10.05 🔢 输入布局抽象（轴字母串 `Layout`）；静态 / 动态 batch 语义统一，以引擎 profile 为唯一真相源
- 2026.10.04 📄 两级配置：新增 `configs/base.ini`，模型 ini 只保留特有字段
- 2026.10.03 ⚙️ 高层 `Infer` API（`InferParams` / `Infer` / `Stream` / `Result`）+ renderer 可插拔
- 2026.10.01 🚀 `1 engine + N context`：共享 `ICudaEngine`，4 worker 显存 ~356 → 309 MiB
- 2026.09.30 🧩 目录按任务拆分 `det` / `seg` / `kpt`；CLI `--net` 支持全部模型并自动推导 `configs/<net>.ini`
- 2026.09.29 🎯 任务类型扩展：YOLOv8-seg / U2Net（seg）、YOLOv8-pose（kpt）
- 2026.09.28 🔥 YOLOv3 / YOLOv4 / YOLO-NAS / EfficientDet / YuNet 接入（det 共 11 个）
- 2026.09.27 ⚡ 新增 `trt_alpha bench`（分步耗时 + p50/p90/p99 + FPS）；YOLOv5 / v6 / v7 / YOLOX / YOLOR 接入
- 2026.09.26 🖥️ CLI `list` / `run` 端到端推理；日志模块与内存池收尾
- 2026.09.25 🔧 三级流水线骨架（source → pool → render）完成
- 2026.09.24 🎉 全面重构：新框架 `core` + `kernels`，第一个模型 YOLOv8 跑通
- 2026.09.20 🔥 重构框架，实现更高性能，解决历史遗留 bug
- ...
- 2023.01.01 🔥 更新 yolov3, yolov4, yolov5, yolov6
- 2023.01.04 🍅 更新 yolov7, yolox, yolor
- 2023.01.05 🎉 更新 u2net, libfacedetection
- 2023.01.08 🚀 全网最快支持 yolov8 的 tensorrt 部署
- 2023.01.20 🍏 更新 efficientdet, pphumanseg
- 2023.12.09 🍁 更新 yolov8-pose
- 2023.12.19 🍉 更新 yolov8-seg
- 2023.12.27 💖 更新 yolonas

</details>

## 严格精度对齐

<details open>
<summary><b>example:yolov26</b></summary>

<p align="center">
  <img src=".github/yolov26_compare_w070.webp" alt="YOLO26 task family: this repository (left) vs PyTorch (right)">
</p>

<p align="center"><sub> 部分对齐效果展示 —— 对齐了<b>所有模型</b>的精度</sub></p>

更多参考：[对齐时，关闭矩形推理](docs/rect_false.md)

</details>

## 部分性能展示

<p align="center">
  <img src=".github/stage_compare_linear.png" alt="YOLO26n per-frame stage cost: TensorRT GPU vs ultralytics CPU" width="100%">
</p>

<p align="center"><sub>YOLO26n 单帧三段耗时　·　1080p / imgsz 640 / batch=2 折算到每帧　·　不含 H2D 上传与后处理 D2H</sub></p>

## 安装依赖

<details open>
<summary><small>Linux</small></summary>

| 依赖 | 版本 | 备注 |
|---|---|---|
| gcc | 13.3.0 | |
| C++ | C++17 | |
| [OpenCV](https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.1) | >= 4.5 | 未来会因性能问题剔除 |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 | |
| CUDA | 12.9 | |
| [FFmpeg](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz) | 9.0 | 🚀 [可选](#4使用-ffmpeg-解码视频实现更高性能)，更高性能 |

具体参考：[Ubuntu 24.04](docs/install_for_ubuntu24.04.md)　or　[Docker](docs/install_with_docker.md)

</details>

<details open>
<summary><small>Windows</small></summary>

| 依赖 | 版本 | 备注 |
|---|---|---|
| Visual Studio | 2017 / 2019 / 2022 | |
| C++ | C++17 | |
| [OpenCV](https://release-assets.githubusercontent.com/github-production-release-asset/5108051/2d11b48c-60e6-4f11-acc4-158da7fcb799?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-10-16T02%3A28%3A08Z&rscd=attachment%3B+filename%3Dopencv-4.8.1-windows.exe&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-10-16T01%3A27%3A29Z&ske=2026-10-16T02%3A28%3A08Z&sks=b&skv=2018-11-09&sig=OcWdIQ0t5iJThr4cjQEp%2BbsEw865e73NiQWS4Fsz0V8%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc5MjExODU1OCwibmJmIjoxNzkyMTE0OTU4LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.THYDLnoAwzXO-Ha4OP4HwG83NRh5N-X5_1xaqFRjxFA&response-content-disposition=attachment%3B%20filename%3Dopencv-4.8.1-windows.exe&response-content-type=application%2Foctet-stream) | >= 4.5 | 未来会因性能问题剔除 |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/zip/TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 | |
| CUDA | 12.9 | |
| [FFmpeg](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip) | 9.0 | 🚀 [可选](#4使用-ffmpeg-解码视频实现更高性能)，更高性能 |

具体参考：[Windows](docs/install_for_windows.md)

</details>

## 快速开始

<details>
<summary><b>1. 拉代码与编译</b></summary>

```bash
git clone -b v2-dev https://github.com/FeiYull/TensorRT-Alpha.git
cd TensorRT-Alpha
```

<details>
<summary><small>Windows（PowerShell）下编译</small></summary>

```powershell
cmake -S D:\VS2022_Project\TensorRT-Alpha `
      -B D:\VS2022_Project\TensorRT-Alpha\build `
      -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11

cmake --build build --config Debug   -j 24
cmake --build build --config Release -j 24

# 验证
./build/bin/Debug/trt_alpha.exe list

```


</details>

<details>
<summary><small>Ubuntu 24.04环境下编译</small></summary>

```bash
cmake -S ~/workspace/TensorRT-Alpha -B ~/workspace/TensorRT-Alpha/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/home/feiyull/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/home/feiyull/thirdparty/TensorRT-10.16.1.11

cmake --build build -j24
# 验证
./build/bin/Release/trt_alpha list
```

产物：`build/bin/Release/trt_alpha`。要 Debug 就换成 `-DCMAKE_BUILD_TYPE=Debug` 重配一次，产物进 `build/bin/Debug/`。

</details>

</details>

<details>
<summary><b>2. 序列化模型</b></summary>

```bash
# ONNX：用 ultralytics 官方命令行导出
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640 batch=2 opset=17 simplify=True

# 序列化：ONNX -> engine
trtexec --onnx=yolo26n.onnx --saveEngine=engines/yolov26n.trt --noTF32
```

更多参考：[导出与序列化](docs/export_and_serialize.md)

</details>

<details open>
<summary><b>3. 执行</b></summary>

```bash
./trt_alpha run --net yolov26 --source data/bus.jpg --show
./trt_alpha run --net yolov26 --source data/xxx.mp4 --config configs/yolov26.ini --save
./trt_alpha run --net yolov26 --source rtsp:xxxxx --engine yolov26.trt --save output/det/
# 加载量化模型
./trt_alpha run --net yolov26 --source data/xxx.mp4 --engine yolov26_fp16.trt
```

更多参考：[CLI 完整用法](docs/cli.md)

源码里直接调用：

```cpp
#include "trt_alpha/infer/infer.hpp"

int main(){
    trt_alpha::InferParams p;     
    p.model_type  = trt_alpha::ModelType::yolov26; // det, pose, cls, obb, depth
    p.source = "xxx.jpg";                          // xx.jpg, xx.mp4, camera_id, url, images' path
    // p.engine = "engines/yolov26.trt" p.config_path = "configs/yolov26.ini"
    trt_alpha::Infer  model(p);   trt_alpha::Stream stream = model.async();    trt_alpha::Result r;
    while (stream.get(r)){
        // here:get the results of inference
        // ......
    }
    stream.stop();
    return 1;
}
```

更多参考：[`samples/sample_infer/sample_infer.cpp`](samples/sample_infer/sample_infer.cpp)

</details>

## 模型支持

<details open>
<summary><b>模型矩阵</b></summary>

`✅` 已跑通并实测 · `🚧` 规划中 · `—` 未提供 · `📥` 点击下载 onnx

<div align="center">

| model | fp32 | fp16 | int8 | 下载onnx |
|---|:---:|:---:|:---:|:---:|
| **YOLO26** | | | | |
| yolo26-det | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n.onnx) |
| yolo26-seg | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-seg.onnx) |
| yolo26-obb | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-obb.onnx) |
| yolo26-cls | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-cls.onnx) |
| yolo26-depth | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-depth.onnx) |
| yolo26-sem | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-sem.onnx) |
| yolo26-pose | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-pose.onnx) |
| yolo26-tracking | 🚧 | 🚧 | 🚧 | —  
| **YOLOV11** | | | | |
| yolov11_det | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo11n.onnx) |
| yolov11_seg | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov11n_seg.onnx) |
| yolov11_obb | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov11n_obb.onnx) |
| yolov11_cls | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov11n_cls.onnx) |
| yolov11_pose | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov11n_pose.onnx) |
| **YOLOV9** | | | | |
| yolov9_det | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov9t.onnx) |
| yolov9_seg | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov9c_seg.onnx) |
| **YOLOV10** | | | | |
| yolov10_det | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov10n.onnx) |

</div>



**其他（沿用 legacy 资产）**

<div align="center">

| model | fp32 | fp16 | int8 | 下载onnx |
|---|:---:|:---:|:---:|:---:|
| yolov3 | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/alpha_yolov3-spp.onnx) |
| yolov4 | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/alpha_yolov4_-1_3_608_608_dynamic.onnx) |
| yolov5 | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov5n.onnx) |
| yolov6 | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov6n.onnx) |
| yolov7 | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov7-tiny.onnx) |
| yolov8_det | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov8n.onnx) |
| yolov8_seg | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov8n_seg.onnx) |
| yolov8_cls | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov8n_cls.onnx) |
| yolov8_pose | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov8n_pose.onnx) |
| yolov8_obb | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolov8n_obb.onnx) |
| yolox | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolox_tiny.onnx) |
| yolor | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolor_csp.onnx) |
| yolo_nas | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolonas_s.onnx) |
| efficientdet | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/efficientdet0.onnx) |
| yunet | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/alpha_yunet_yunet_final_dynamic_simplify.onnx) |
| u2net | ✅ | — | — | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/u2netp.onnx) |

</div>

</details>

## 目录结构

<details open>
<summary><b>目录树</b></summary>

```
app/                 CLI（run / bench / list / build）
cmake/               OpenCV / TensorRT / CUDA 探测 + 运行时部署
configs/             模型 INI（base.ini + <model>.ini 两级）
data/                图片 / 视频 / classes（需自行放入）
include/trt_alpha/   公共头
  ├── core/          引擎 / 内存池 / 队列 / 池 / 模型注册（零 OpenCV）
  ├── datasource/    数据源接口（OpenCV 实现，可替换为 GStreamer）
  ├── pipeline/      三级流水线调度
  ├── renderer/      渲染接口（OpenCV 实现）
  ├── kernels/       CUDA 算子（ops/ = 预处理 / 后处理算子族，legacy/ = 旧实现归档）
  └── det|seg|kpt|cls|depth|sem|obb/  任务类型与模型
src/                 实现
samples/             使用示例（含 sample_infer 高层 API 示例）
test/                单元测试（CTest）
scripts/             平台脚本（linux/ = 引擎序列化、参数矩阵测试）
```

</details>

## 其他

#### [1.配置文件](docs/config.md)
#### [2.量化](docs/quantization.md)

#### [3.部署自己训练的模型](docs/train_and_deploy_custom.md)

#### [4.使用 FFmpeg 解码视频，实现更高性能](docs/install_ffmpeg_zh.md)

## Citation
```bibtex
@misc{FeiYull_TensorRT-Alpha,
  author = {FeiYull},
  title = {TensorRT-Alpha},
  year = {2023},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {https://github.com/FeiYull/tensorrt-alpha}
}
```


