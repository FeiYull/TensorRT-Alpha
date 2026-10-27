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
  English | <a href="README.md">简体中文</a>
</p>

## Introduction

<details open>
<summary><b>Highlights</b></summary>

A C++ inference repository built on **TensorRT + CUDA**, with **every CUDA C operator written by hand**:

- **End-to-end GPU pipeline**: data source → batching → GPU preprocess → inference → GPU postprocess → render / save, with zero CPU round-trips.
- **Multi-batch inference**: `1 engine + N contexts` running concurrently, each worker owning its own `cudaStream`.
- **Bit-exact replication of the original PyTorch inference**: aligned with the official ultralytics implementation — see [Bit-exact Accuracy Alignment](#bit-exact-accuracy-alignment).
- **Many input sources**: images, image directories, video streams (RTSP / RTMP / HTTP(S) / UDP), webcams and URLs — a single `--source` argument figures out the type by itself.
- **Hardware video decoding**: [FFmpeg hardware decoding](docs/build_ffmpeg.md) (NVDEC) keeps decoding off the CPU.

</details>

<details>
<summary><b>Roadmap</b></summary>



- **TODO**: edge devices such as Nvidia Nano (testing planned)
- **TODO**: GStreamer support — make OpenCV detachable and replace it with a GStreamer data source
- **TODO**: `trt_alpha build` — one-command ONNX → engine serialization
- **TODO**: finer-grained performance profiling — **Nsight Systems** to trace the whole pipeline timeline (how H2D / inference / D2H / postprocess / render are laid out, and where the bubbles are), and **Nsight Compute** to profile the CUDA kernels one by one (occupancy, memory access, bottleneck hunting)
- **TODO**: more model families

</details>

<details>
<summary><b>Changelog</b></summary>

- 2026.10.27 📚 Docs rebuilt: the README keeps a single default path (TensorRT + OpenCV); FFmpeg decoding and FFmpeg + OpenGL rendering each get their own document; the six build scripts follow `build_<os>_<libs>`
- 2026.10.24 🎮 Multi-stream rendering completed: zero-copy CUDA-GL interop display, click-to-zoom, per-source result queues; 9x 1080p stable (pinned memory falls back with a single warning when `RLIMIT_MEMLOCK` is tight)
- 2026.10.24 🧩 Three build trees fixed (`build` / `build-ffm` / `build-gl`) + two switches (tests off, samples on by default); yolov8_old / yolov8_pose_old / yolov8_seg_old removed
- 2026.10.24 ⚡ seg / depth / sem result composition moved entirely to CUDA; the OpenCV and GL renderers now share one implementation
- 2026.10.23 🖼️ Multi-stream video + OpenGL render backend (GLFW + CUDA-GL interop); OpenCV demoted to an optional backend (`-DTRT_ALPHA_WITH_OPENCV=OFF`)
- 2026.10.21 🚀 release v1.3.0: FFmpeg decoding is now compile-time optional (`--decode ffmpeg` / `--hwaccel nvdec`), NVDEC device pass-through + pinned copy-back, `[source]` section and tri-state `HostCopy`
- 2026.10.20 🐛 Fixed the `float4` alignment overrun in fused preprocessing for non-640 inputs (e.g. 672)
- 2026.10.19 📦 samples switched to one sample per task; unused parameters and `input_output_names` removed; training / deployment docs added (EN / ZH)
- 2026.10.18 🚀 YOLOV11 / YOLOV8 / YOLOV9 / YOLOV10 support, all on the YOLO26 fused pre/post-processing: yolov11 (5 tasks); yolov8 rewritten for all 5 tasks; yolov9 det + seg; yolov10 det (one2many only). Measured against Python per box: boxes within 3.4e-4 px, conf within 2e-6, zero mismatched mask pixels
- 2026.10.16 💖 release v1.0
- 2026.10.15 🐧 Linux support: GCC 13 port, unified line-ending / editor policy, engine serialization script; automatic Windows runtime DLL deployment; hardened argument parsing and source detection + a full-parameter-matrix `run` test script
- 2026.10.14 🧠 Complete YOLO26 task family: cls / depth / obb / pose / seg / sem; the pipeline diagram on the landing page
- 2026.10.12 🎯 YOLO26-det integration: postprocess operator family (decode / NMS / fused / vectorized) and a uniform `yolo26` → `yolov26` naming
- 2026.10.11 ⚡ CUDA preprocess operator family bit-exact with OpenCV (resize / copyMakeBorder / cvtColor / hwc2chw / normalize + fused kernels)
- 2026.10.10 📦 Legacy CUDA kernels archived under `kernels/legacy/`, public headers fully translated to English
- 2026.10.07 🔍 Code-review hardening: batch upper-bound guards, frame-drop policy, overwrite-input guards; CTest introduced (32 cases)
- 2026.10.06 🛡️ Runtime configuration introspection (`logConfigBox` + `[unused]` dead-key detection); URL streams (rtsp / rtmp / http(s) / udp) + directory sources + output-directory rules
- 2026.10.05 🔢 Input layout abstraction (axis-letter strings, `Layout`); static / dynamic batch semantics unified, with the engine profile as the single source of truth
- 2026.10.04 📄 Two-level configuration: `configs/base.ini` added, per-model ini files keep only their own fields
- 2026.10.03 ⚙️ High-level `Infer` API (`InferParams` / `Infer` / `Stream` / `Result`) + pluggable renderer
- 2026.10.01 🚀 `1 engine + N context`: shared `ICudaEngine`, 4 workers 356 → 309 MiB of device memory
- 2026.09.30 🧩 Directories split by task (`det` / `seg` / `kpt`); CLI `--net` supports every model and derives `configs/<net>.ini` automatically
- 2026.09.29 🎯 More task types: YOLOv8-seg / U2Net (seg), YOLOv8-pose (kpt)
- 2026.09.28 🔥 YOLOv3 / YOLOv4 / YOLO-NAS / EfficientDet / YuNet integrated (11 detection models in total)
- 2026.09.27 ⚡ `trt_alpha bench` added (per-stage timings + p50/p90/p99 + FPS); YOLOv5 / v6 / v7 / YOLOX / YOLOR integrated
- 2026.09.26 🖥️ CLI `list` / `run` end-to-end inference; logging module and memory pool finalized
- 2026.09.25 🔧 Three-stage pipeline skeleton (source → pool → render) completed
- 2026.09.24 🎉 Full rewrite: new `core` + `kernels` framework, first model (YOLOv8) running
- 2026.09.20 🔥 Framework rewritten for higher performance, long-standing legacy bugs fixed
- ...
- 2023.01.01 🔥 Added yolov3, yolov4, yolov5, yolov6
- 2023.01.04 🍅 Added yolov7, yolox, yolor
- 2023.01.05 🎉 Added u2net, libfacedetection
- 2023.01.08 🚀 The fastest TensorRT deployment for yolov8 on the web
- 2023.01.20 🍏 Added efficientdet, pphumanseg
- 2023.12.09 🍁 Added yolov8-pose
- 2023.12.19 🍉 Added yolov8-seg
- 2023.12.27 💖 Added yolonas

</details>

## Bit-exact Accuracy Alignment

<details open>
<summary><b>example:yolov26</b></summary>

<p align="center">
  <img src=".github/yolov26_compare_w070.webp" alt="YOLO26 task family: this repository (left) vs PyTorch (right)">
</p>

<p align="center"><sub>Partial alignment results — bit-exact for <b>every model</b></sub></p>

More: [Disabling rectangular inference when aligning](docs/rect_false.md)

</details>

## Performance Preview

<p align="center">
  <img src=".github/stage_compare_linear.png" alt="YOLO26n per-frame stage cost: TensorRT GPU vs ultralytics CPU" width="100%">
</p>

<p align="center"><sub>YOLO26n per-frame stage cost　·　1080p / imgsz 640 / batch=2 folded per frame　·　excluding the H2D upload and the postprocess D2H</sub></p>

## Installation

| Dependency | Version | Notes |
|---|---|---|
| CUDA | 12.9 | CUDA C pre/post-processing, dual batch supported |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 | FP32 / FP16 / INT8 quantization |
| [OpenCV](https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.1) | >= 4.5 | decoding, window, saving images |
| C++ | C++17 | |
| gcc (Linux) | 13.3 | |
| Visual Studio (Windows) | 2017 / 2019 / 2022 | |

Detailed steps: [Windows](docs/install_for_windows.md)　or　[Ubuntu 24.04](docs/install_for_ubuntu24.04.md)　or　[Docker](docs/install_with_docker.md)

Hardware video decoding and multi-stream GL rendering are two optional paths, each with its
own document covering dependencies, build and run:

- [FFmpeg hardware decoding](docs/build_ffmpeg.md)
- [FFmpeg + OpenGL multi-stream rendering](docs/build_ffmpeg_opengl.md)

## Quick Start

<details>
<summary><b>1. Clone and build</b></summary>

```bash
git clone -b v2-dev https://github.com/FeiYull/TensorRT-Alpha.git
cd TensorRT-Alpha
```

<details>
<summary><small>Build on Windows (PowerShell)</small></summary>

```powershell
cmake -S D:\VS2022_Project\TensorRT-Alpha `
      -B D:\VS2022_Project\TensorRT-Alpha\build `
      -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11

cmake --build build --config Release -j 24
.\build\bin\Release\trt_alpha.exe list
```

</details>

<details>
<summary><small>Build on Ubuntu 24.04</small></summary>

```bash
cmake -S ~/workspace/TensorRT-Alpha -B ~/workspace/TensorRT-Alpha/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/home/feiyull/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/home/feiyull/thirdparty/TensorRT-10.16.1.11 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc

cmake --build build -j24
./build/bin/Release/trt_alpha list
```

Debug: use `--config Debug`, or `-DCMAKE_BUILD_TYPE=Debug` on a single-config generator; the output goes to `bin/Debug/`.

</details>

</details>

<details>
<summary><b>2. Serialize the model</b></summary>

```bash
# ONNX: exported with the official ultralytics CLI
pip install ultralytics
yolo export model=yolo26n.pt format=onnx imgsz=640 batch=2 opset=17 simplify=True

# serialization: ONNX -> engine
trtexec --onnx=yolo26n.onnx --saveEngine=engines/yolov26n.trt --noTF32
```

More: [Exporting and serializing](docs/export_and_serialize.md)

> The `yolo export` line is **only an ultralytics example**, not a requirement. Any other framework (MMDetection, Detectron2, PaddleDetection, your own PyTorch training script) follows the same flow: **export to ONNX manually** inside that framework, then let `trtexec` serialize it. This repository only consumes ONNX — it does not export models.

</details>

<details open>
<summary><b>3. Run</b></summary>

```bash
./build/bin/Release/trt_alpha run --net yolov26 --source data/bus.jpg --show    # image
./build/bin/Release/trt_alpha run --net yolov26 --source data/people.mp4 --show  # video
./build/bin/Release/trt_alpha run --net yolov26 --source data/images/ --save output/det/  # directory
./build/bin/Release/trt_alpha run --net yolov26 --source 0 --show                # camera
./build/bin/Release/trt_alpha run --net yolov26 --source data/bus.jpg --save output/det/
# quantized engine: swap in --engine
./build/bin/Release/trt_alpha run --net yolov26 --source data/bus.jpg --engine engines/yolov26n_fp16.trt --show
```

On Windows, use `.\build\bin\Release\trt_alpha.exe` instead.

More: [Full CLI reference](docs/cli.md)

Called directly from C++:

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

More: [`samples/sample_infer/sample_infer.cpp`](samples/sample_infer/sample_infer.cpp)

</details>

## Model Support

<details open>
<summary><b>Model matrix</b></summary>

`✅` implemented and benchmarked · `🚧` planned · `—` not provided · `📥` download onnx

<div align="center">

| model | fp32 | fp16 | int8 | download onnx |
|---|:---:|:---:|:---:|:---:|
| **YOLO26** | | | | |
| yolo26-det | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n.onnx) |
| yolo26-seg | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-seg.onnx) |
| yolo26-obb | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-obb.onnx) |
| yolo26-cls | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-cls.onnx) |
| yolo26-depth | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-depth.onnx) |
| yolo26-sem | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-sem.onnx) |
| yolo26-pose | ✅ | ✅ | ✅ | [📥](https://github.com/FeiYull/TensorRT-Alpha/releases/download/v1.0.0/yolo26n-pose.onnx) |
| yolo26-tracking | 🚧 | 🚧 | 🚧 | — |
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

**Others (reusing the legacy assets)**

<div align="center">

| model | fp32 | fp16 | int8 | download onnx |
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

## Directory Structure

<details open>
<summary><b>Directory tree</b></summary>

```
app/                 CLI (run / bench / list / build)
cmake/               OpenCV / TensorRT / CUDA detection + runtime deployment
configs/             model ini files (two levels: base.ini + <model>.ini)
data/                images / videos / classes (bring your own)
include/trt_alpha/   public headers
  ├── core/          engine / memory pool / queue / pool / model registry (no OpenCV at all)
  ├── datasource/    data source interface (OpenCV implementation, swappable for GStreamer)
  ├── pipeline/      three-stage pipeline scheduling
  ├── renderer/      renderer interface (OpenCV implementation)
  ├── kernels/       CUDA operators (ops/ = preprocess & postprocess operator family, legacy/ = archived old code)
  └── det|seg|kpt|cls|depth|sem|obb/  task types and models
src/                 implementations
samples/             usage examples (including the high-level sample_infer API sample)
test/                unit tests (off by default, -DTRT_ALPHA_BUILD_TESTS=ON)
scripts/             build scripts (build_<os>_<libs>) + engine serialization / parameter matrix
```

</details>

## Other

### [1. Configuration](docs/config.md)
### [2. Quantization](docs/quantization.md)
### [3.Train and deploy custom model](docs/train_and_deploy_custom.md)
### [4.FFmpeg hardware decoding (`--decode` / `--hwaccel`)](docs/build_ffmpeg.md)

### [5.FFmpeg + OpenGL multi-stream rendering (`--render gl`)](docs/build_ffmpeg_opengl.md)

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

