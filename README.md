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


- **TODO**：support for docker
- **TODO**：Nvidia Nano 等多款边缘设备（计划测试中）
- **TODO**：支持 GStreamer —— 可拆卸 OpenCV，替换为 GStreamer 数据源
- **TODO**：`trt_alpha build` —— ONNX → engine 一键序列化
- **TODO**：更细粒度的性能测试 —— 用 **Nsight Systems** 剖析整条流水线的时间线（H2D / 推理 / D2H / 后处理 / 渲染的排布与空泡），用 **Nsight Compute** 逐个 CUDA kernel 剖析（占用率、访存、瓶颈定位）
- **TODO**：YOLO11 等更多模型系列

</details>

<details>
<summary><b>更新日志</b></summary>

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

## 安装依赖


<details open>
<summary><small>Linux</small></summary>

| 依赖 | 版本 |
|---|---|
| gcc | 13.3.0 |
| C++ | C++17 |
| [OpenCV](https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.1) | >= 4.5 |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 |
| CUDA | 12.9 |

</details>

<details open>
<summary><small>Windows</small></summary>

| 依赖 | 版本 |
|---|---|
| Visual Studio | 2017 / 2019 / 2022 |
| C++ | C++17 |
| [OpenCV](https://release-assets.githubusercontent.com/github-production-release-asset/5108051/2d11b48c-60e6-4f11-acc4-158da7fcb799?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-10-16T02%3A28%3A08Z&rscd=attachment%3B+filename%3Dopencv-4.8.1-windows.exe&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-10-16T01%3A27%3A29Z&ske=2026-10-16T02%3A28%3A08Z&sks=b&skv=2018-11-09&sig=OcWdIQ0t5iJThr4cjQEp%2BbsEw865e73NiQWS4Fsz0V8%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc5MjExODU1OCwibmJmIjoxNzkyMTE0OTU4LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.THYDLnoAwzXO-Ha4OP4HwG83NRh5N-X5_1xaqFRjxFA&response-content-disposition=attachment%3B%20filename%3Dopencv-4.8.1-windows.exe&response-content-type=application%2Foctet-stream) | >= 4.5 |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/zip/TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 |
| CUDA | 12.9 |

</details>

## 快速开始

<details>
<summary><b>1. 拉代码</b></summary>

```bash
git clone -b v2-dev https://github.com/FeiYull/TensorRT-Alpha.git
cd TensorRT-Alpha
```

</details>

<details>
<summary><b>2. 编译</b></summary>

<details>
<summary><small>Windows（PowerShell）</small></summary>

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
<summary><small>Ubuntu 24.04</small></summary>

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
<summary><b>3. 导出 ONNX</b></summary>

ONNX 由 [ultralytics](https://github.com/ultralytics/ultralytics) 官方仓库导出，先把仓库拉下来：

```bash
git clone https://github.com/ultralytics/ultralytics
cd ultralytics
pip install -e .
```

```python
from ultralytics import YOLO

# model = YOLO("yolo26n-cls.pt")
model = YOLO("yolo26n-obb.pt")
# model = YOLO("yolo26n-pose.pt")
# model = YOLO("yolo26n-seg.pt")
# model = YOLO("yolo26n-depth.pt")

model.export(
    format="onnx",
    batch=2,
    imgsz=640,
    dynamic=False,
    simplify=True,
    opset=17,
    nms=None,
)
```

> 注：各任务的 ONNX **按模型原生分辨率导出**（det / pose / seg = 640，depth = 768，obb / sem = 1024，cls = 224），与仓库 `engines/` 里现成的引擎一致；换分辨率要用同尺寸重建引擎。

</details>

<details>
<summary><b>4. 用 trtexec 编译模型</b></summary>

```bash
# FP32：本仓库用于验证精度的指令（关掉 TF32，避免 fp32 精度漂移）
trtexec --onnx=sample.onnx --saveEngine=sample.trt --noTF32

# FP16：必须显式加 --fp16，否则 TensorRT 会按 FP32 计算
trtexec --onnx=yolo26n_fp16.onnx --saveEngine=yolov26n_fp16.trt --fp16

# INT8：用 Python 端 PTQ 导出的 Q/DQ 量化 ONNX，再配 --int8
trtexec --onnx=yolo26n_int8.onnx --saveEngine=yolov26n_int8.trt --int8

# 让 TensorRT 自行挑最优精度组合
trtexec --onnx=sample.onnx --saveEngine=sample.trt --best
```

> 注：`--noTF32` **只在 FP32 引擎上用于精度对齐**；FP16 / INT8 引擎别加（那是关掉 TF32 张量核的开关，对已量化的图只会白白牺牲速度）。`--fp16` 与 `--int8` 也**不要同时给**，TensorRT 10.16 对 weakly-typed 量化网络会直接 build 失败。

</details>

<details>
<summary><b>5. CLI 执行</b></summary>

```bash
BIN=./build/bin/Release/trt_alpha          # Windows：.\build\bin\Release\trt_alpha.exe

# 列出所有已注册模型
$BIN list

# ---------- YOLO26 全任务族 ----------
# 目标检测（默认读 configs/yolov26.ini -> engines/yolov26n.trt）
$BIN run --net yolov26       --source data/bus.jpg    --show
# 实例分割
$BIN run --net yolov26_seg   --source data/bus.jpg    --save
# 姿态估计
$BIN run --net yolov26_pose  --source data/zidane.jpg --show
# 旋转框检测（DOTA）
$BIN run --net yolov26_obb   --source data/boats.mp4  --save
# 图像分类
$BIN run --net yolov26_cls   --source data/dog.jpg    --show
# 深度估计
$BIN run --net yolov26_depth --source data/road0.jpg  --save
# 语义分割
$BIN run --net yolov26_sem   --source data/road1.jpg  --save
$BIN run --net yolov26_sem   --source data/road1.jpg  --save MY_SAVE_PATH

# ---------- 输入源自动判别 ----------
$BIN run --net yolov26 --source data/                            # 图片目录（每张独立推理）
$BIN run --net yolov26 --source data/people.mp4 --show           # 本地视频
$BIN run --net yolov26 --source 0 --show                         # 摄像头 0
$BIN run --net yolov26 --source rtsp://192.168.1.10:554/stream1  # RTSP / RTMP / HTTP(S) / UDP

# ---------- 换引擎 / 调 batch / 多 worker ----------
$BIN run --net yolov26 --source data/ --engine engines/yolov26n_fp16.trt --save
$BIN run --net yolov26 --source data/ --engine engines/yolov26n_int8.trt --save
$BIN run --net yolov26 --source data/ --batch 2 --workers 4 --save   # yolov26 引擎静态 batch=2

# ---------- 性能测量 ----------
$BIN bench --net yolov26 --iters 100 --warmup 10
$BIN bench --engine engines/yolov26n_fp16.trt --batch 2 --iters 200 --warmup 20
```

> 注：`--source data/` 表示传进去一个路径，默认只读取图片，不管模型支持 batch 为多少，实际推理的时候自动退化为 batch=1。

</details>

<details open>
<summary><b>6. 快速使用源码</b></summary>

更多和具体使用参考：[`samples/sample_infer/sample_infer.cpp`](samples/sample_infer/sample_infer.cpp)

```cpp
#include "trt_alpha/infer/infer.hpp"

int main(){
    trt_alpha::InferParams p;     
    p.model_type  = trt_alpha::ModelType::yolov26; // det, pose, cls, obb, depth
    p.source = "xxx.jpg";                          // xx.jpg, xx.mp4, camera_id, url, images' path
    trt_alpha::Infer  model(p);   trt_alpha::Stream stream = model.async();    trt_alpha::Result r;
    while (stream.get(r)){
        // here:get the results of inference
        // ......
    }
    stream.stop();
    return 1;
}
```

</details>

## 模型支持

<details open>
<summary><b>模型矩阵</b></summary>

`✅` 已跑通并实测 · `🚧` 规划中 · `—` 未提供

<div align="center">

| model | fp32 | fp16 | int8 | onnx |
|---|:---:|:---:|:---:|:---:|
| **YOLO26** | | | | |
| yolo26-det | ✅ | ✅ | ✅ | ✅ |
| yolo26-seg | ✅ | ✅ | ✅ | ✅ |
| yolo26-obb | ✅ | ✅ | ✅ | ✅ |
| yolo26-cls | ✅ | ✅ | ✅ | ✅ |
| yolo26-depth | ✅ | ✅ | ✅ | ✅ |
| yolo26-sem | ✅ | ✅ | ✅ | ✅ |
| yolo26-pose | ✅ | ✅ | ✅ | ✅ |
| yolo26-tracking | 🚧 | 🚧 | 🚧 | 🚧 |
| **YOLO11** | | | | |
| yolo11-det | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-seg | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-obb | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-cls | 🚧 | 🚧 | 🚧 | 🚧 |
| yolo11-pose | 🚧 | 🚧 | 🚧 | 🚧 |

</div>

**其他（沿用 legacy 资产）**

<div align="center">

| model | fp32 | fp16 | int8 | onnx |
|---|:---:|:---:|:---:|:---:|
| yolov3 | ✅ | — | — | ✅ |
| yolov4 | ✅ | — | — | ✅ |
| yolov5 | ✅ | — | — | ✅ |
| yolov6 | ✅ | — | — | ✅ |
| yolov7 | ✅ | — | — | ✅ |
| yolov8 | ✅ | — | — | ✅ |
| yolov8-pose | ✅ | — | — | ✅ |
| yolov8-seg | ✅ | — | — | ✅ |
| yolox | ✅ | — | — | ✅ |
| yolor | ✅ | — | — | ✅ |
| yolo_nas | ✅ | — | — | ✅ |
| efficientdet | ✅ | — | — | ✅ |
| yunet | ✅ | — | — | ✅ |
| u2net | ✅ | — | — | ✅ |

</div>

</details>

## 严格精度对齐

<details open>
<summary><b>example:yolov26</b></summary>

<p align="center">
  <img src=".github/yolov26-Offical(left)vsOurs(right).jpg" alt="yolov26 official (left) vs ours (right)">
</p>

<p align="center"><sub>左：ultralytics 官方 PyTorch 推理　·　右：本仓库 TensorRT 推理 —— 对齐了<b>所有模型</b>的精度</sub></p>

<details>
<summary><small>注：为什么要关闭矩形推理（rect=False）</small></summary>

在使用仓库代码部署模型的时候，兼顾高效和精度，yolov26 等模型都关闭了**矩形推理**：

```python
from ultralytics import YOLO

# model = YOLO("yolo26n-cls.pt")
model = YOLO("yolo26n-obb.pt")
# model = YOLO("yolo26n-pose.pt")
# model = YOLO("yolo26n-seg.pt")
# model = YOLO("yolo26n-depth.pt")

results = model(
    r"Data/boats.jpg",
    rect=False,     # close rectangle inference!
    save=True,
    project="./save",
    name="output",
    exist_ok=True,
)
```

静态引擎走**正方形 letterbox**，与 `rect=False` 完全一致；开着矩形推理去对拍，坐标会整体偏移。

</details>

</details>

## 配置文件

<details>
<summary><b>ini格式的配置文件说明</b></summary>

配置分两级，**优先级从高到低**：

```
调用 API / 命令行参数   >   configs/<model>.ini   >   configs/base.ini
```

- **`configs/base.ini`**：所有模型共享的默认值（归一化参数、后处理阈值、线程池、输出目录等）。
- **`configs/<model>.ini`**：单个模型的覆盖项（引擎路径、类别文件、batch、该模型特有的阈值）。
- **调用 API / 命令行参数**：**最高优先级**，运行期直接覆盖上面两层。例如 `--engine` 覆盖 INI 里的 `engine=`、`--batch` 覆盖 `batch_size`、`--workers` 覆盖 `[pool].workers`、`--config` 换掉整个模型 INI；高层 `InferParams` 里的字段同理。



`configs/yolov26.ini` ：

```ini
[model]
engine = engines/yolov26n.trt
class_names_file = data/classes/coco80.txt
input_output_names = images,output0
num_class = 80
# extra: 每 anchor 的额外通道（seg = 32 / pose = 51）。det 用 0。
# extra = 0

[input]
batch_size = 2

[postprocess]
conf_thresh = 0.25
iou_thresh = 0.7
top_k = 300
```

</details>

## 性能

<details open>
<summary><b>展开查看性能数据</b></summary>

YOLO26n，batch = 2，Ubuntu 24.04 + RTX 5060 Ti，trtexec `--loadEngine` 200 次迭代（100 次平均）：

**yolo26-det（640×640）精度对比**

| 引擎 | 序列化参数 | GPU 延迟 | 吞吐 | 体积 |
|---|---|---|---|---|
| `yolov26n.trt`（FP32） | `--noTF32` | 2.401 ms | 302.4 qps | 13.03 MB |
| **`yolov26n_fp16.trt`（FP16）** | **`--fp16`** | **1.708 ms** | **383.4 qps** | 8.20 MB  |



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


