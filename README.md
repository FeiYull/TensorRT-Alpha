````markdown
# 🚀 TensorRT-Alpha

### A Modern, Modular C++ TensorRT Inference Framework

[![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)
[![CUDA](https://img.shields.io/badge/CUDA-12.x-76B900.svg)](https://developer.nvidia.com/cuda-toolkit)
[![TensorRT](https://img.shields.io/badge/TensorRT-10.16-76B900.svg)](https://developer.nvidia.com/tensorrt)
[![CMake](https://img.shields.io/badge/CMake-3.25%2B-064F8C.svg)](https://cmake.org/)
[![Platform](https://img.shields.io/badge/Platform-Windows%20%7C%20Linux-lightgrey.svg)]()
[![License](https://img.shields.io/badge/License-MIT-green.svg)](./LICENSE)

---

## 🚧 v1.0 Refactor in Progress

> **⚠️ Important**
>
> The legacy codebase (v0.9) is archived on the `main` branch and remains
> stable. The `refactor` branch is the new framework under development; its
> API will change significantly. **Do not depend on it in production.**

```
┌──────────────────────────────────────────────────────────────────┐
│                                                                  │
│   main       →  Legacy v0.9   · stable · archived                │
│   refactor   →  New v1.0      · WIP    · API not frozen          │
│                                                                  │
└──────────────────────────────────────────────────────────────────┘
```

---

## ✨ Why v1.0

The previous version worked, but had accumulated structural problems:

- 🚫 OpenCV leaked into core interfaces (`cv::Mat` everywhere)
- 🔗 The inference API was tied to a specific model type
- 🧱 Engine loading, model config, and scheduling were all coupled

v1.0 rebuilds the framework from scratch around a small set of interfaces,
so that adding a model, swapping a data source, or replacing the renderer
does not require touching the rest of the codebase.

---

## 🎯 Design Goals

| 🎯 Goal | 📝 Description |
|---|---|
| 🧩 **Decoupled Modules** | Data source / Inference / Rendering / Scheduling are fully separated |
| 🚫 **OpenCV-free Core** | Core layer has zero OpenCV dependency |
| 🔌 **Replaceable** | Data source and renderer can be swapped (OpenCV → FFmpeg / Qt / Skia) |
| ⚡ **High Performance** | Memory pool, multi-context concurrency, dedicated CUDA streams, zero-copy |
| 🎨 **Extensible** | Add a new model = one `.cpp` + one registration line |
| 🌐 **Cross-platform** | Windows / Linux / (future) Jetson |
| 🛡️ **Safe** | RAII, Debug double-free detection, rich error context |

---

## 🏗️ Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                          app / CLI                               │
└──────────────────────────────────────────────────────────────────┘
                              │
        ┌─────────────────────┼─────────────────────┐
        ▼                     ▼                     ▼
┌───────────────┐     ┌───────────────┐     ┌───────────────┐
│  IDataSource  │     │ InferencePool │     │   IRenderer   │
│  (OpenCV)     │     │  (N workers)  │     │   (OpenCV)    │
└───────────────┘     └───────────────┘     └───────────────┘
        │                     │                     │
        └─────────────────────┼─────────────────────┘
                              ▼
                      ┌───────────────┐
                      │  BatchResult  │  ← data contract
                      └───────────────┘
                              │
                              ▼
                      ┌───────────────┐
                      │     core      │  ← no OpenCV
                      │  TRT wrapper  │
                      │  memory pool  │
                      │  logger / cfg │
                      └───────────────┘
```

### Rules

1. 🚫 Core headers must not include `<opencv2/...>` or use `cv::Mat`
2. 🔌 OpenCV is allowed only in data source and renderer implementations
3. 📦 `core` does not depend on `nvonnxparser`
4. 🧩 Every task type (`det`, `seg`, `cls`) has its own `types.hpp`
5. ✨ Adding a new model = one `.cpp` + one registration line

---

## 📦 Current Progress

### ✅ Completed

| Area | Content |
|---|---|
| **core** | Type system, memory views, batch, result contract |
| **core** | Thread-safe logger with boxed allocation output |
| **core** | Paths, INI parser, model config |
| **core** | TensorRT engine wrapper |
| **core** | IModel interface, model registry |
| **core** | Memory pool, RAII buffers, CUDA stream |
| **kernels** | Preprocessing and postprocessing CUDA operators |
| **det** | `IDetector` and a working `YoloV8` |
| **test** | 15 test executables, all passing |

### 🚧 In Progress

- 🖼️ `IRenderer` (abstract renderer)
- 📷 `IDataSource` (abstract data source)
- 🎬 `Pipeline` (3-stage scheduler)

### 🔮 Planned

- 🎯 More models: YOLOv8-seg, YOLOv9, YOLOv5, YOLOv6, YOLOv7, EfficientDet
- 📹 Video / webcam / RTSP input
- 🎨 Alternative renderers (Qt, Skia, JSON output)

---

## 🎁 Highlights

### 🧠 Memory Pool

`cudaMalloc` / `cudaFree` are expensive and called frequently during inference.
The pool caches freed blocks and reuses them. Numbers from the test suite:

```
requests=804   hits=795   cudaAllocs=9   peak=7044 KiB
                       └─ hit rate 98.9%
```

**`cudaMalloc` calls dropped from 804 to 9.**

Implementation notes:

- Size-ordered map (`std::map<capacity, blocks>`), no power-of-two rounding
- 256-byte alignment (matches `cudaMalloc` guarantee)
- Per-kind mutex; heavy CUDA calls happen outside the lock
- Debug-build double-free detection, zero overhead in Release

### 📊 Boxed Allocation Log

The most common source of bugs in TensorRT deserialization is getting buffer
sizes wrong. Every buffer allocation prints a box:

```
+------------------------------------------------------------------+
| [ALLOC] yolov8.input_nchw                                        |
|   batch    = 1                                                   |
|   channels = 3                                                   |
|   height   = 640                                                 |
|   width    = 640                                                 |
|   dtype    = fp32 (4 bytes)                                      |
|   calc     = 1 * 3 * 640 * 640 * 4                               |
|   bytes    = 4915200                                             |
|   MB       = 4.69 MB                                             |
|   space    = Device                                              |
+------------------------------------------------------------------+
```

Full computation, byte count, and MB value in one place.

### ⚡ CUDA Architecture Detection

CMake detects the local GPU's compute capability at configure time via
`nvidia-smi`, and falls back to a list of common architectures when no GPU is
present. This avoids PTX JIT and the corresponding first-inference stall.

| GPU | Compute Cap |
|---|---|
| RTX 50 series | sm_120 |
| RTX 40 series | sm_89 |
| RTX 30 series | sm_86 |
| CI (no GPU) | fallback list |

On the test machine (RTX 5060 Ti, sm_120), the happy path went from **505 ms
to 8.55 ms**.

---

## 🛠️ Build

**Requirements**

- C++17
- CMake 3.25+
- CUDA 12.x
- TensorRT 10.16
- OpenCV 4.5+ (optional)

**Configure & Build**

```bash
cmake -S . -B build -G "Visual Studio 17 2022" -A x64 \
    -DOpenCV_DIR=<opencv_dir> \
    -DTensorRT_ROOT=<tensorrt_dir>

cmake --build build --config Release -j 16
```

**Run Tests**

```bash
ctest --test-dir build -C Release
```

**Run a Single Model Test**

```bash
./build/bin/Release/test_yolov8.exe <engine.trt> <image.jpg>
```

---

## 📂 Layout

```
trt_alpha/
├── include/trt_alpha/     public headers
│   ├── core/              base layer, no OpenCV
│   ├── kernels/           CUDA operator declarations
│   ├── det/ seg/ cls/     task types
│   ├── datasource/        data source implementations
│   ├── renderer/          renderer implementations
│   └── pipeline/          scheduler
│
├── src/                   implementations
│   ├── core/
│   ├── kernels/           .cu files
│   ├── det/               model implementations
│   └── app/               CLI
│
├── test/                  per-component tests
├── samples/               usage examples
├── configs/               one INI per model
├── data/                  images, class lists, engines
└── docs/                  documentation
```

---

## 🗺️ Roadmap

- [x] **Phase 1** — core (types / memory / logger / config / engine / registry)
- [x] **Phase 2** — kernels (preprocess / postprocess)
- [x] **Phase 3** — YoloV8 end-to-end
- [ ] **Phase 4** — `IRenderer`
- [ ] **Phase 5** — `IDataSource`
- [ ] **Phase 6** — `Pipeline`
- [ ] **Phase 7** — more models
- [ ] **Phase 8** — full CLI
- [ ] **Phase 9** — v1.0 release

---

## 📜 License

MIT

---

<div align="center">

**⭐ Star this repo if it helps you!**

</div>
````