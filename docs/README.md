
```markdown
# TensorRT-Alpha

> 工业级 C++ TensorRT 推理框架 —— 模块解耦、零拷贝、多 Batch、多 Context 并发。

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)]()
[![CUDA](https://img.shields.io/badge/CUDA-12.x-green.svg)]()
[![TensorRT](https://img.shields.io/badge/TensorRT-10.16-green.svg)]()
[![License](https://img.shields.io/badge/license-MIT-blue.svg)]()

---

## 目录

- [特性](#特性)
- [架构](#架构)
- [快速开始](#快速开始)
- [构建](#构建)
- [CLI](#cli)
- [使用示例](#使用示例)
- [性能](#性能)
- [目录结构](#目录结构)
- [路线图](#路线图)

---

## 特性

- ⚡ **GPU 端到端推理**：预处理 → 推理 → 后处理全程 CUDA，零 CPU 往返
- 🎯 **多 Batch 推理**：一次 `enqueueV3` 处理整批图像
- 🔥 **多 Context 并发**：N 个 Worker 各持独立 context + stream，满血榨干 GPU
- 🧠 **工业级内存池**：显存 / 页锁定内存池化复用，`cudaMalloc` 调用降低 **98.9%**
- 🧩 **模块解耦**：数据源 / 推理 / 渲染 / 调度彻底分离，**核心层零 OpenCV 依赖**
- 🔌 **可替换**：数据源、渲染器可无缝切换到 FFmpeg / Qt / Skia / DeepStream
- 🎨 **易扩展**：新增模型 = 1 个 `.cpp` + 1 行注册
- 🛡️ **安全**：RAII、Debug 双重归还检测、完整错误上下文

---

## 架构

```
APP / CLI
    │
    ├──▶ IDataSource（OpenCV 实现，可换）
    ├──▶ InferencePool（N workers，各持独立 context）
    └──▶ IRenderer（OpenCV 实现，可换）
              │
              ▼
        BatchResult（数据契约，无 OpenCV）
              │
              ▼
        core（底座，无 OpenCV）
          TrtEngine / MemoryPool / Logger / Paths
          DeviceBuffer / PinnedBuffer / CudaStream
          BoundedQueue / IModel / ModelRegistry
          BufferView / Batch / BatchResult
```

### 三级流水线

```
[数据源线程1] ── batch ──┐
[数据源线程2] ── batch ──┼──▶ [推理池 N worker] ──▶ [结果队列] ──▶ [渲染线程]
[数据源线程3] ── batch ──┘
```

- **数据源线程**：每源一个，读帧 → `pool.submit()` → 推结果队列
- **推理池 worker**：N 个（可配），跑 `IModel` 四步
- **渲染线程**：1 个，从结果队列取 → 画 → 存 / 显

---

## 快速开始

### 依赖

| 依赖 | 版本 | 必填 |
|---|---|---|
| OS | Windows 10 / Linux | ✅ |
| C++ | C++17 | ✅ |
| CMake | >= 3.25 | ✅ |
| CUDA | 12.x | ✅ |
| TensorRT | 10.16 | ✅ |
| OpenCV | >= 4.5 | 可选（数据源 / 渲染） |
| nvonnxparser | 随 TRT | 可选（`build` 命令） |

### 准备

- **engine**：用 `trtexec` 或 `trt_alpha build` 把 ONNX 转成 `.trt`；
- **类别文件**：`data/classes/coco80.txt`（格式：`名字 R G B`，行号 = label）；
- **模型 INI**：`configs/yolov8.ini`。

---

## 构建

### 首次配置（只跑一次）

```powershell
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11
```

**注意**：不加 `-DCMAKE_CUDA_ARCHITECTURES`（`CMakeLists.txt` 自动探测 GPU 架构）。

### Debug 构建

```powershell
cmake --build build --config Debug -j 16
```

### Release 构建

```powershell
cmake --build build --config Release -j 16
```

---

## CLI

### `trt_alpha list` —— 列出已注册模型

```powershell
trt_alpha list
```

### `trt_alpha run` —— 推理

**图片**：

```powershell
trt_alpha run --image data/bus.jpg --config configs/yolov8.ini --save
```

**图片目录**：

```powershell
trt_alpha run --images data --config configs/yolov8.ini --save
```

**视频**：

```powershell
trt_alpha run --video data/people.mp4 --config configs/yolov8.ini --show
```

**摄像头**：

```powershell
trt_alpha run --camera 0 --config configs/yolov8.ini --show
```

**命令行覆盖 INI**：

```powershell
trt_alpha run --video data/people.mp4 --config configs/yolov8.ini --batch 2 --show
```

### `trt_alpha bench` —— 性能测量

```powershell
trt_alpha bench --engine D:/ThirdParty/TensorRT-10.16.1.11/bin/yolov8n.trt `
                --config configs/yolov8.ini `
                --iters 100 --warmup 10
```

**输出**：

```
Latency (ms):
  mean : 1.87
  p50  : 1.86
  p90  : 1.95
  p99  : 2.04

Throughput:
  FPS  : 534.2

Per-step (mean, ms):
  setBatch    : 0.10
  preprocess  : 0.008
  infer       : 0.76
  postprocess : 1.01
```

### `trt_alpha build` —— ONNX → engine（TODO）

---

## 使用示例

**`samples/sample_yolov8/sample_yolov8.cpp`** —— 手工组合组件：

```cpp
// 1. ModelConfig
core::ModelConfig cfg;
cfg.engine = enginePath;
cfg.batchSize = 1;
cfg.dstH = 640;
cfg.dstW = 640;
cfg.classNamesFile = "data/classes/coco80.txt";
cfg.extras["num_class"] = "80";
cfg.extras["conf_thresh"] = "0.25";
cfg.classNames = core::loadClassNamesFile(cfg.classNamesFile);

// 2. InferencePool
core::InferencePool pool(cfg,
    []() -> std::unique_ptr<IModel> {
        return ModelRegistry::instance().create("yolov8");
    },
    /*workers=*/1);

// 3. DataSource
datasource::SourceConfig srcCfg;
srcCfg.type = datasource::SourceType::Image;
srcCfg.path = "data/bus.jpg";
srcCfg.batchSize = cfg.batchSize;
std::vector<std::unique_ptr<datasource::IDataSource>> sources;
sources.push_back(std::make_unique<datasource::OpenCVSource>(srcCfg));

// 4. Renderer
renderer::OpenCVRenderer renderer;

// 5. Pipeline
pipeline::PipelineConfig pcfg;
pcfg.sources = std::move(sources);
pcfg.pools = { &pool };
pcfg.renderer = &renderer;
pcfg.classNames = cfg.classNames;
pcfg.saveEnabled = true;
pcfg.saveDir = "save";

pipeline::Pipeline p(std::move(pcfg));
p.start();
p.waitForCompletion();
```

---

## 性能

**环境**：RTX 5060 Ti / CUDA 12.9 / TensorRT 10.16 / YOLOv8n 640×640 / batch=1 / Release。

| 指标 | 值 |
|---|---|
| **端到端延迟** | mean **1.87 ms** / p50 1.86 ms / p99 2.04 ms |
| **吞吐** | **534 FPS** |
| `setBatch` | 0.10 ms |
| `preprocess` | 0.008 ms |
| `infer` | 0.76 ms |
| `postprocess` | 1.01 ms |

**瓶颈**：**后处理（decode + NMS + D2H）占 54%**，v1.1 优化重点。

---

## 目录结构

```
trt_alpha/
├── CMakeLists.txt
├── app/                    CLI（run / bench / list / build）
├── cmake/                  OpenCV / TensorRT / CUDA 探测
├── configs/                模型 INI
├── data/                   测试图片 / 视频 / 类别文件
├── include/trt_alpha/      公共头文件
│   ├── core/               核心（无 OpenCV）
│   ├── datasource/         数据源接口
│   ├── det/ seg/ cls/      任务类型
│   ├── kernels/            CUDA 工具库
│   ├── pipeline/           调度器
│   └── renderer/           渲染接口
├── src/                    实现
├── samples/                使用示例
├── test/                   单元测试
└── docs/                   文档
```

---

## 路线图

### v1.0（2026-10-07 之后）

- ✅ 端到端推理（det）
- ✅ CLI（run / bench / list）
- ✅ 三级流水线
- ✅ 工业级内存池
- ✅ 日志系统

### v1.1

- **更多 det 模型**：YoloV5 / V6 / V7 / V9；
- **YoloV8-seg**（seg 第一个）；
- **后处理优化**（decode + NMS）。

### v1.2

- **分类**（ResNet / MobileNet）；
- **分割**（U2Net / PPHumanSeg）；
- **姿态**（YoloV8-pose）。

### v1.3+

- **RT-DETR** 等新架构；
- **视频流 / RTSP**；
- **服务化**（FastAPI + gRPC）。

---

## 联系

- **GitHub**：https://github.com/FeiYull/TensorRT-Alpha
- **作者**：FeiYull

---

## License

MIT
```

---

## 第三步：Release 全测试

```powershell
cd D:\VS2022_Project\TensorRT-Alpha

# 快速回归
.\build\bin\Release\test_logger.exe
.\build\bin\Release\test_bounded_queue.exe
.\build\bin\Release\test_buffer.exe
.\build\bin\Release\test_batch.exe
.\build\bin\Release\test_memory_pool.exe
.\build\bin\Release\test_datasource.exe
.\build\bin\Release\test_renderer.exe
.\build\bin\Release\test_kernels.exe
.\build\bin\Release\test_config.exe
.\build\bin\Release\test_paths.exe
.\build\bin\Release\test_ini_parser.exe
.\build\bin\Release\test_engine.exe
.\build\bin\Release\test_task_types.exe
.\build\bin\Release\test_model_registry.exe
.\build\bin\Release\test_batch_result.exe
.\build\bin\Release\test_buffer_view.exe
```

**期望**：全 `ALL PASS`。

**真推理**：

```powershell
.\build\bin\Release\test_yolov8.exe `
    D:\ThirdParty\TensorRT-10.16.1.11\bin\yolov8n.trt `
    D:\VS2022_Project\TensorRT-Alpha\data\bus.jpg

.\build\bin\Release\test_pipeline.exe `
    D:\ThirdParty\TensorRT-10.16.1.11\bin\yolov8n.trt `
    D:\VS2022_Project\TensorRT-Alpha\data\bus.jpg
```

**期望**：`ALL PASS`。

**CLI**：

```powershell
.\build\bin\Release\trt_alpha.exe list
.\build\bin\Release\trt_alpha.exe run --image data/bus.jpg --config configs/yolov8.ini --save
.\build\bin\Release\trt_alpha.exe bench --engine D:/ThirdParty/TensorRT-10.16.1.11/bin/yolov8n.trt --iters 50 --warmup 5
```
