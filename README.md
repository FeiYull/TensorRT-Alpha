# 🚀 TensorRT-Alpha

### 工业级 C++ TensorRT 推理框架 · 正在重构

---

## 🚧 Refactoring in Progress

本仓库正在进行 **工业级重构**。

- **`main`** —— 旧版（v0.9），稳定，已归档
- **`v2`** —— 新版（v1.0），开发中，API 未冻结

**预计 2026-10-07 之后发布第一个 Release。**

---

## ✨ 核心特性

- ⚡ **GPU 端到端推理**：预处理 → 推理 → 后处理全程 CUDA，零 CPU 往返
- 🎯 **多 Batch 推理**：一次 `enqueueV3` 处理整批图像
- 🔥 **多 Context 并发**：N 个 Worker 各持独立 context + stream，满血榨干 GPU
- 🧠 **工业级内存池**：显存 / 页锁定内存池化复用，`cudaMalloc` 调用降低 **98.9%**
- 🧩 **模块解耦**：数据源 / 推理 / 渲染 / 调度彻底分离，**核心层零 OpenCV 依赖**
- 🔌 **可替换**：数据源、渲染器可无缝切换到 FFmpeg / Qt / Skia / DeepStream
- 🎨 **易扩展**：新增模型 = 1 个 `.cpp` + 1 行注册
- 🛡️ **安全**：RAII、Debug 双重归还检测、完整错误上下文

---

## 🏗️ 架构图

```
╔══════════════════════════════════════════════════════════════════════════════╗
║                              APP  LAYER                                      ║
║                                                                              ║
║   ┌─────────────┐    ┌─────────────┐    ┌─────────────┐    ┌─────────────┐  ║
║   │   run       │    │   bench     │    │   list      │    │   build     │  ║
║   │  (infer)    │    │  (benchmark)│    │  (registry) │    │ (onnx→trt)  │  ║
║   └─────────────┘    └─────────────┘    └─────────────┘    └─────────────┘  ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                      │
                                      ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                            PIPELINE  LAYER                                   ║
║                                                                              ║
║   ┌────────────────────────────────────────────────────────────────────────┐ ║
║   │                          Pipeline                                      │ ║
║   │                                                                        │ ║
║   │   [Source Threads]  ──submit──▶  [InferencePool]  ──future──▶  [Result │ ║
║   │         (N)                          (M workers)                Queue] │ ║
║   │                                                                   │    │ ║
║   │                                                                   ▼    │ ║
║   │                                                        [Render Thread] │ ║
║   └────────────────────────────────────────────────────────────────────────┘ ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
        │                          │                          │
        ▼                          ▼                          ▼
╔════════════════╗    ╔════════════════════╗    ╔════════════════════╗
║  DATASOURCE    ║    ║   INFERENCE        ║    ║    RENDERER        ║
║  (OpenCV)      ║    ║                    ║    ║    (OpenCV)        ║
║ ─────────────  ║    ║ ─────────────────  ║    ║ ─────────────────  ║
║ IDataSource    ║    ║ InferencePool      ║    ║ IRenderer          ║
║ ├ Image        ║    ║ └ N × IModel       ║    ║ ├ Detections       ║
║ ├ Image Dir    ║    ║    (context+stream)║    ║ ├ Segmentations    ║
║ ├ Video File   ║    ║                    ║    ║ ├ Classifications  ║
║ └ Camera       ║    ║ IModel (Factory)   ║    ║ └ Save / Show      ║
║                ║    ║ ├ init(cfg)        ║    ║                    ║
║ 可换: FFmpeg   ║    ║ ├ setBatch(Batch)  ║    ║ 可换: Qt / Skia    ║
║      DeepStream║    ║ ├ preprocess()     ║    ║      JSON          ║
║      自定义     ║    ║ ├ infer()          ║    ║                    ║
║                ║    ║ ├ postprocess()    ║    ║                    ║
║                ║    ║ └ commitResult()   ║    ║                    ║
╚════════════════╝    ╚════════════════════╝    ╚════════════════════╝
        │                          │                          │
        └──────────────────────────┼──────────────────────────┘
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                          DATA  CONTRACT                                      ║
║                                                                              ║
║   Batch { buffer, views, validCount, sourceId, firstFrameIndex }             ║
║   BatchResult { views, buffer, validCount,                                  ║
║                 detections, segmentations, classifications,                 ║
║                 inferenceMs, submitTime }                                    ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             KERNELS  (CUDA)                                  ║
║                                                                              ║
║   ┌─────────────────────────────┐    ┌─────────────────────────────┐        ║
║   │  PREPROCESS (.cu)           │    │  POSTPROCESS (.cu)          │        ║
║   │                             │    │                             │        ║
║   │  resizeLetterbox            │    │  transposeAnchors           │        ║
║   │  bgrToNchwNormalized        │    │  decodeYoloV8Head           │        ║
║   │                             │    │  decodeYoloV8SegHead        │        ║
║   │                             │    │  nmsFast                    │        ║
║   └─────────────────────────────┘    └─────────────────────────────┘        ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                             CORE  (no OpenCV)                                ║
║                                                                              ║
║   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       ║
║   │  TrtEngine   │ │ MemoryPool   │ │   Logger     │ │    Paths     │       ║
║   │  TRT 10 RAII │ │ Device +     │ │  thread-safe │ │  root detect │       ║
║   │              │ │ PinnedHost   │ │  boxed logs  │ │              │       ║
║   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       ║
║   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       ║
║   │  DeviceBuf   │ │ PinnedBuf    │ │ CudaStream   │ │ ModelConfig  │       ║
║   │  RAII        │ │ RAII         │ │ RAII         │ │ INI + extras │       ║
║   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       ║
║   ┌──────────────┐ ┌──────────────┐ ┌──────────────┐ ┌──────────────┐       ║
║   │  BoundedQueue│ │ IModel +     │ │ BufferView   │ │  INI Parser  │       ║
║   │  Drop/Block  │ │ Registry     │ │ + Batch      │ │              │       ║
║   └──────────────┘ └──────────────┘ └──────────────┘ └──────────────┘       ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
                                   │
                                   ▼
╔══════════════════════════════════════════════════════════════════════════════╗
║                              RUNTIME                                         ║
║                                                                              ║
║        CUDA 12.x   ·   TensorRT 10.16   ·   cuDNN   ·   NVIDIA GPU          ║
║                                                                              ║
╚══════════════════════════════════════════════════════════════════════════════╝
```

---

## 🧩 模块职责

| 层 | 模块 | 职责 |
|---|---|---|
| **app** | CLI | 参数解析 / 命令分发 |
| **pipeline** | `Pipeline` | 三级流水线调度（源 / 推理 / 渲染） |
| **datasource** | `IDataSource` | 读帧 → 产出 `Batch` |
| **inference** | `InferencePool` + `IModel` | 多 context 并发推理 |
| **renderer** | `IRenderer` | 结果可视化 / 存盘 |
| **kernels** | CUDA 算子 | 预处理 / 后处理 |
| **core** | 底座 | TRT 封装 / 内存池 / 日志 / 配置 |

---

## 🗓️ Release Plan

| 阶段 | 状态 |
|---|---|
| Phase 1 — core 层 | ✅ |
| Phase 2 — kernels | ✅ |
| Phase 3 — YoloV8 端到端 | ✅ |
| Phase 4 — IRenderer | ✅ |
| Phase 5 — IDataSource | ✅ |
| Phase 6 — Pipeline | ✅ |
| Phase 7 — 更多模型 | 🚧 |
| Phase 8 — 完整 CLI | 🚧 |
| **v1.0 Release** | **2026-10-07+** |

---

## 📜 License

MIT