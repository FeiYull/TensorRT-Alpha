![TensorRT-Alpha inference pipeline](.github/trt_alpha_pipeline.svg)

<h3 align="center">TensorRT-Alpha</h3>

<p align="center">
把 CV 模型从 PyTorch / ONNX 一路加速到 TensorRT 的 <b>工业级 C++ 推理框架</b><br/>
数据源 → 组批 → GPU 前处理 → 推理 → GPU 后处理 → 渲染 / 落盘，全程零 CPU 往返
</p>

<p align="center">
  <img alt="CUDA" src="https://img.shields.io/badge/CUDA-12.x-76B900">
  <img alt="TensorRT" src="https://img.shields.io/badge/TensorRT-10.x-76B900">
  <img alt="C++" src="https://img.shields.io/badge/C++-17-00599C">
  <img alt="platform" src="https://img.shields.io/badge/platform-Windows%20%7C%20Linux-555">
</p>

---

## 优势

- **GPU 端到端**：预处理（resize / letterbox / cvtColor / hwc2chw / normalize）、推理、后处理（decode / NMS / scale_boxes）全部自己写的 CUDA 算子，不落 CPU。
- **逐位对齐 OpenCV**：`INTER_LINEAR` / `copyMakeBorder` / `cvtColor` 复刻 OpenCV 4.8.1，1000 张 VOC 共 12.288 亿元素 `max|d| = 0`。换掉 `cv::resize` 不影响任何一张图的结果。
- **多 Batch + 多 Context 并发**：`1 engine + N context`，每个 worker 独占一条 `cudaStream`，`H2D → preprocess → enqueueV3 → postprocess → D2H` 串成一条流水线。
- **工业级内存池**：显存 / 页锁定（pinned）内存池化复用，稳态下 `cudaMalloc` 近乎为零。
- **模块解耦**：`core` 层零 OpenCV、零 nvonnxparser；数据源与渲染器是接口，可换成 FFmpeg / GStreamer / Qt / DeepStream。跨层只靠 `BatchResult` 一个契约。
- **配置即意图，引擎是唯一真相源**：INI 里写的 batch / shape 与引擎 profile 不符时**显式报错退出**，绝不静默改值 —— 线上不会出现“跑得到但结果不对”。
- **易扩展**：新增一个模型 = 1 个头 + 1 个源文件 + 1 行注册 + 1 个 INI。

## 未来计划

- **GStreamer 数据源**：硬解 + 多路 RTSP 接入，与现有 `OpenCVSource` 并存、可切换（规划中）。
- YOLO26 其余任务头：seg / obb / cls / depth / tracking。
- 更多模型系列：YOLO11 等。
- `trt_alpha build`：ONNX → engine 一键序列化。

---

## 环境要求

| 依赖 | 版本 | 说明 |
|---|---|---|
| OS | Windows 10/11 · Linux | 已实测 |
| C++ | C++17 | |
| CMake | ≥ 3.25 | |
| CUDA | 12.x | 需 `nvcc` + `cudart` |
| TensorRT | 10.x（兼容 8.x） | 需 `nvinfer` + `nvonnxparser` |
| OpenCV | ≥ 4.5 | 仅数据源 / 渲染器用到 |
| 显卡 | 计算能力 7.5+ | `CMakeLists.txt` 会自动探测本机架构 |

> 基准环境：RTX 5060 Ti · CUDA 12.9 · TensorRT 10.16.1.11 · OpenCV 4.8.1 · VS2022。

## 编译

```bash
# 当前开发分支是 v2-dev（main 是 2023 年的老架构，已归档）
git clone -b v2-dev https://github.com/FeiYull/TensorRT-Alpha.git
cd TensorRT-Alpha
```

**Linux**

```bash
sudo apt install build-essential cmake libopencv-dev
# TensorRT 用官方 tar 包解压到 /usr/local/TensorRT（或任意目录，下面用 -DTensorRT_ROOT 指路）

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DTensorRT_ROOT=/usr/local/TensorRT
cmake --build build -j
```

**Windows**

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11
cmake --build build --config Release -j 16
```

产物：

| 平台 | 可执行文件 |
|---|---|
| Windows | `build\bin\Release\trt_alpha.exe`（另有各 `sample_*` / `test_*`） |
| Linux | `build/bin/trt_alpha` |

跑单测：`ctest --test-dir build -C Release --output-on-failure`

> **不在仓库里的东西**（`.gitignore` 已忽略，需自行准备）：
> ① `.trt` / `.onnx` 引擎与权重；② `data/` 下的图片、视频与 `data/classes/*.txt` 类别文件。
> 引擎可以用 `trtexec` 自己导，或用[老仓库网盘](https://share.weiyun.com/3T3mZKBm)里现成的 ONNX。

---

## 支持列表

`✅` 已跑通并实测 · `🚧` 规划中 · `—` 未提供

| model | fp32 | fp16 | int8 | onnx |
|---|:---:|:---:|:---:|:---:|
| **YOLO26** | | | | |
| yolo26-det | ✅ | ✅ | ✅ ¹ | ✅ |
| yolo26-seg | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo26-obb | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo26-cls | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo26-depth | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo26-tracking | 🚧 | 🚧 | 🚧 | 🚧 |
| **YOLO11** | | | | |
| yolo11-det | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-seg | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-obb | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-cls | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-pose | 🚧 | 🚧 | 🚧 | 🚧 |
| **其他（沿用 v1 资产）** | | | | |
| yolov3 / yolov4 / yolov5 / yolov6 / yolov7 / yolov8 | ✅ | — | — | ✅ |
| yolov8-seg / yolov8-pose | ✅ | — | — | ✅ |
| yolox / yolor | ✅ | — | — | ✅ |
| yolo_nas / efficientdet | ✅ | — | — | ✅ |
| yunet / u2net | ✅ | — | — | ✅ |

- `fp32` / `fp16` / `int8` = 已实测跑通的引擎精度；`onnx` = 已提供导出流程或现成下载。
- ¹ YOLO26 的 INT8 由 Python 端 PTQ（Q/DQ 显式量化）产出，只走 ONNX 全流程；TensorRT 的 `IInt8Calibrator` 自 **10.13 起废弃、11.0 移除**，**不要指望 `trtexec --int8` 帮你校准**。详见 [已知问题](#已知问题)。

---

## 快速使用

```powershell
# 列出全部已注册模型
.\build\bin\Release\trt_alpha.exe list

# 输入源统一走 --source，类型自动判别（图片 / 目录 / 视频 / 流 / 摄像头 id）
.\build\bin\Release\trt_alpha.exe run --source data/bus.jpg --net yolov26 --show          # 单张图片
.\build\bin\Release\trt_alpha.exe run --source data/ --net yolov26 --save                # 图片目录（每次推理只处理一张；混尺寸也安全）
.\build\bin\Release\trt_alpha.exe run --source data/people.mp4 --net yolov26 --show      # 本地视频
.\build\bin\Release\trt_alpha.exe run --source 0 --net yolov26 --show                    # 摄像头 0
.\build\bin\Release\trt_alpha.exe run --source rtsp://192.168.1.10:554/stream1 --net yolov8 --show   # RTSP / RTMP / HTTP(S) / UDP 流

# 换引擎（默认读 configs/<net>.ini 里的 engine=）
.\build\bin\Release\trt_alpha.exe run --source data/ --net yolov26 --engine engines/yolov26n_fp16.trt --save

# 性能测量：延迟 / 吞吐 / 分步耗时
.\build\bin\Release\trt_alpha.exe bench --net yolov26 --iters 100 --warmup 10
```

Linux 把前缀换成 `./build/bin/` 即可。常用参数：`--net` `--source <path|url|id>` `--engine` `--config` `--batch` `--workers` `--save [dir]` `--show`。

> `--source` 的判别规则（与高层 `InferParams.source` 同一套）：URL → 流；纯数字 → 摄像头 id；视频后缀 → 本地视频；其余 → 图片（目录会自动按目录处理）。所以命令行不再有 `--image` / `--images` / `--video` / `--url` / `--camera`。

## 代码里怎么用

最简路径 —— 高层 `Infer` API（完整版见 [`samples/sample_infer/sample_infer.cpp`](samples/sample_infer/sample_infer.cpp)）：

```cpp
#include "trt_alpha/infer/infer.hpp"

trt_alpha::InferParams p;
p.model_type  = trt_alpha::ModelType::yolov8;   // 另有 yolov8_seg / yolov8_pose / yolov26 全任务族（yolov26_cls / yolov26_seg / yolov26_pose / yolov26_obb / yolov26_depth / yolov26_sem）/ u2net 等
p.config_path = "configs/yolov8.ini";
p.source      = "data/bus.jpg";

trt_alpha::Infer model(p);
auto stream = model.async();

trt_alpha::Result r;
while (stream.get(r))                       // 一批一批取
    for (auto& f : r.frames())              // 一帧一帧看
    {
        auto raw = f.image();               // 原图：原始像素，不带框
        auto ann = f.rendered();            // 渲染图：框 / 掩码 / 关键点已画好（p.draw = false 时为空）
        for (auto& b : f.boxes())           // 结果：坐标 / 掩码都已是【原图尺寸】
            std::cout << b.label << " " << b.confidence
                      << " [" << b.left << "," << b.top << "," << b.right << "," << b.bottom << "]\n";
    }
stream.stop();
```

想手工组装组件（数据源 / 推理池 / 渲染器 / 流水线），看 [`samples/sample_yolov26/sample_yolov26.cpp`](samples/sample_yolov26/sample_yolov26.cpp)。

> YOLO26 三种入口都可用：CLI（`--net yolov26`）、高层 `Infer`（`ModelType::yolov26`，见 [`sample_infer`](samples/sample_infer/sample_infer.cpp)）、手工组装（`sample_yolov26`）。
> 高层 `Infer` 还登记了 YOLO26 全任务族（`yolov26_cls / _seg / _pose / _obb / _depth / _sem`）：只换 `model_type` + `config_path`，取数代码不变。
> 默认一次给齐三样：`frame.image()` **原图**、`frame.rendered()` **渲染图**、以及各任务结果 —— `frame.boxes()/keypoints()/masks()/obbs()/classifications()`，`frame.depth()`（float 米）/ `frame.semantic()`（uint8 类别图）；**坐标与掩码统一为原图尺寸**（seg 的 `mask` 是网络尺寸、`maskOrig` 是原图尺寸）。
> 渲染默认开（`p.draw = true`）但**不落盘、不显示**；要存盘 / 弹窗再开 `p.save` / `p.show`，只要数据可 `p.draw = false` 省掉每帧绘制 + 拷贝。

---

## 性能

YOLO26n，640×640，batch = 2，trtexec `--loadEngine` 200 次迭代：

| 引擎 | GPU 延迟 | 吞吐 | 体积 |
|---|---|---|---|
| `yolov26n.trt`（FP32） | 2.555 ms | 304 qps | 13.0 MB |
| **`yolov26n_fp16.trt`（FP16）** | **1.736 ms** | **375 qps** | 8.32 MB |
| `yolov26n_int8.trt`（INT8） | 1.885 ms | 355 qps | 9.72 MB |

小模型上 **FP16 是最优解**：INT8 反而比它慢 8.6%（head 保持浮点 + Q/DQ 反量化开销），价值只在体积。
三个引擎的 I/O 恒为 `fp32 [2,3,640,640]` / `fp32 [2,84,8400]`，**换精度不需要改一行 C++**。

## 已知问题

- **INT8 检测头被量化**：检测头最后一层输出卷积也插了 Q/DQ，logits 被标定量程封顶，置信度上限偏低（FP32 下 `0.8606` → INT8 下 `0.8592`），偶尔出现重复伪框。生产建议用 **FP32 / FP16**；修复方向（导出时把该卷积排除量化）在计划内。

---

## 目录结构

```
app/                 CLI（run / bench / list）
cmake/               OpenCV / TensorRT / CUDA 探测
configs/             模型 INI（base.ini + <model>.ini 两级）
data/                图片 / 视频 / classes（需自行放入）
include/trt_alpha/   公共头
  ├── core/          引擎 / 内存池 / 队列 / 池 / 模型注册（零 OpenCV）
  ├── datasource/    数据源接口（OpenCV 实现）
  ├── pipeline/      三级流水线调度
  ├── renderer/      渲染接口（OpenCV 实现）
  ├── kernels/       CUDA 算子（ops/ = 预处理 / 后处理算子族，legacy/ = 旧实现归档）
  └── det|seg|kpt|cls/  任务类型与模型
src/                 实现
samples/             使用示例
test/                单元测试（CTest）
```

---

## 更新日志

### v1（2023，老架构，见 `main` 分支）

- 2023.01.01 🔥 yolov3 / yolov4 / yolov5 / yolov6
- 2023.01.04 🍅 yolov7 / yolox / yolor
- 2023.01.05 🎉 u2net / libfacedetection
- 2023.01.08 🚀 全网最快支持 yolov8 的 TensorRT 部署
- 2023.01.20 🍏 efficientdet / pphumanseg
- 2023.12.09 🍁 yolov8-pose
- 2023.12.19 🍉 yolov8-seg
- 2023.12.27 💖 yolo-nas

### v2（2026，全面重构，`v2-dev` 分支）

- 2026.09.24 🎉 架构推倒重写：`core` + `kernels`，第一个模型 YoloV8 跑通
- 2026.09.25–27 三级流水线骨架、工业级内存池、日志系统、CLI（`list` / `run` / `bench`）
- 2026.09.28–30 v1 模型全量移植：yolov3 ~ yolov8、yolox、yolor、yolo-nas、efficientdet、yunet、u2net、yolov8-pose、yolov8-seg
- 2026.10.01 🔥 `1 engine + N context` 并发，每个 worker 独占 context + stream
- 2026.10.03–05 高层 `Infer` API、输入布局抽象（轴字母串）、batch 语义统一到引擎 profile、两级配置 `base.ini`
- 2026.10.07 🛡️ 代码评审硬化（丢帧策略、batch 上限校验、内存池双重归还、CTest 入场）
- 2026.10.10 📦 legacy CUDA 核归档到 `kernels/legacy/`，公开头注释全面英译
- 2026.10.11 ⚡ 逐位对齐 OpenCV 的 CUDA 预处理算子族（resize / copyMakeBorder / cvtColor / hwc2chw / normalize + 融合核）+ 对拍样例
- 2026.10.12 🚀 **YOLO26-det 接入**：后处理算子族（decode / NMS / 融合 / 向量化）、`extra` 通道预留、静态 batch 与图片源语义收口
- 2026.10.12 🧪 Python 端 PTQ 导出 FP16 / INT8 引擎，三档精度实测
- 2026.10.13 📖 首页架构示意图 + README

---

## 引用

```
@misc{FeiYull_TensorRT-Alpha,
  author = {FeiYull},
  title  = {TensorRT-Alpha},
  year   = {2023},
  publisher = {GitHub},
  howpublished = {https://github.com/FeiYull/TensorRT-Alpha}
}
```
