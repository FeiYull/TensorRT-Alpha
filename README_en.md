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

- 2026.10.18 🚀 YOLO11 support: det / cls / obb / pose / seg, sharing the YOLO26 pre/post-processing (measured: zero mismatched mask pixels, boxes within 3.4e-4 px); five new registered names and configs
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

<details open>
<summary><small>Linux</small></summary>

| Dependency | Version |
|---|---|
| gcc | 13.3.0 |
| C++ | C++17 |
| [OpenCV](https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.1) | >= 4.5 |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 |
| CUDA | 12.9 |

Detailed steps: [Ubuntu 24.04](docs/install_for_ubuntu24.04.md)　or　[Docker](docs/install_with_docker.md)

</details>

<details open>
<summary><small>Windows</small></summary>

| Dependency | Version |
|---|---|
| Visual Studio | 2017 / 2019 / 2022 |
| C++ | C++17 |
| [OpenCV](https://release-assets.githubusercontent.com/github-production-release-asset/5108051/2d11b48c-60e6-4f11-acc4-158da7fcb799?sp=r&sv=2018-11-09&sr=b&spr=https&se=2026-10-16T02%3A28%3A08Z&rscd=attachment%3B+filename%3Dopencv-4.8.1-windows.exe&rsct=application%2Foctet-stream&skoid=96c2d410-5711-43a1-aedd-ab1947aa7ab0&sktid=398a6654-997b-47e9-b12b-9515b896b4de&skt=2026-10-16T01%3A27%3A29Z&ske=2026-10-16T02%3A28%3A08Z&sks=b&skv=2018-11-09&sig=OcWdIQ0t5iJThr4cjQEp%2BbsEw865e73NiQWS4Fsz0V8%3D&jwt=eyJ0eXAiOiJKV1QiLCJhbGciOiJIUzI1NiJ9.eyJpc3MiOiJnaXRodWIuY29tIiwiYXVkIjoicmVsZWFzZS1hc3NldHMuZ2l0aHVidXNlcmNvbnRlbnQuY29tIiwia2V5Ijoia2V5MSIsImV4cCI6MTc5MjExODU1OCwibmJmIjoxNzkyMTE0OTU4LCJwYXRoIjoicmVsZWFzZWFzc2V0cHJvZHVjdGlvbi5ibG9iLmNvcmUud2luZG93cy5uZXQifQ.THYDLnoAwzXO-Ha4OP4HwG83NRh5N-X5_1xaqFRjxFA&response-content-disposition=attachment%3B%20filename%3Dopencv-4.8.1-windows.exe&response-content-type=application%2Foctet-stream) | >= 4.5 |
| [TensorRT](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/zip/TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip?t=eyJscyI6ImJzZW8iLCJsc2QiOiJodHRwczovL2NuLmJpbmcuY29tLyJ9) | 10.16 |
| CUDA | 12.9 |

Detailed steps: [Windows](docs/install_for_windows.md)

</details>

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

cmake --build build --config Debug   -j 24
cmake --build build --config Release -j 24

# verify
./build/bin/Debug/trt_alpha.exe list

```


</details>

<details>
<summary><small>Build on Ubuntu 24.04</small></summary>

```bash
cmake -S ~/workspace/TensorRT-Alpha -B ~/workspace/TensorRT-Alpha/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/home/feiyull/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/home/feiyull/thirdparty/TensorRT-10.16.1.11

cmake --build build -j24
# verify
./build/bin/Release/trt_alpha list
```

Output: `build/bin/Release/trt_alpha`. For a Debug build, reconfigure with `-DCMAKE_BUILD_TYPE=Debug`; the output goes to `build/bin/Debug/`.

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

More: [Exporting and serializing](docs/export_and_serialize.md)　or　[Serialization parameters per model](docs/serialize_engines.md)

</details>

<details open>
<summary><b>3. Run</b></summary>

```bash
./trt_alpha run --net yolov26 --source data/bus.jpg --show
./trt_alpha run --net yolov26 --source data/xxx.mp4 --save
./trt_alpha run --net yolov26 --source rtsp:xxxxx --engine yolov26.trt --save output/det/
# load a quantized engine
./trt_alpha run --net yolov26 --source data/xxx.mp4 --engine yolov26_fp16.trt
```

More: [Full CLI reference](docs/cli.md)

Called directly from C++:

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

More: [`samples/sample_infer/sample_infer.cpp`](samples/sample_infer/sample_infer.cpp)

</details>

## Model Support

<details open>
<summary><b>Model matrix</b></summary>

`✅` implemented and benchmarked · `🚧` planned · `—` not provided

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
| yolo11-det | ✅ | ✅ | ✅ | ✅ |
| yolo11-seg | ✅ | ✅ | ✅ | ✅ |
| yolo11-obb | ✅ | ✅ | ✅ | ✅ |
| yolo11-cls | ✅ | ✅ | ✅ | ✅ |
| yolo11-pose | ✅ | ✅ | ✅ | ✅ |

</div>

**Others (reusing the legacy assets)**

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
test/                unit tests (CTest)
scripts/             platform scripts (linux/ = engine serialization, parameter-matrix testing)
```

</details>

## Other

### [1. Configuration](docs/config.md)
### [2. Quantization](docs/quantization.md)

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

