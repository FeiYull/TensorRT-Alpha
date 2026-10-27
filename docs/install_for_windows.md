# Windows

Only download and extract - nothing is installed into system paths and no environment variable is set.

## 0. Dependencies

| Dependency | Version | Notes |
|---|---|---|
| Visual Studio | 2022 (Desktop development with C++) | 2017 / 2019 also work |
| CMake | >= 3.25 | |
| CUDA Toolkit | 12.9 | CUDA C pre/post-processing, dual batch |
| [TensorRT](#2-tensorrt-1016111) | 10.16.1.11 | FP32 / FP16 / INT8 quantization |
| [OpenCV](#1-opencv-481) | >= 4.5 | decode + window + saving images |
| C++ | C++17 | |
| driver | 610.x | |

This is the default build (`build`). Two optional extras have their own documents:

- FFmpeg, hardware video decoding: [build_ffmpeg.md](build_ffmpeg.md)
- FFmpeg + OpenGL, zero-copy multi-stream rendering: [build_ffmpeg_opengl.md](build_ffmpeg_opengl.md)

## 1. OpenCV 4.8.1

[Download opencv-4.8.1-windows.exe](https://github.com/opencv/opencv/releases/download/4.8.1/opencv-4.8.1-windows.exe)
(mirror: [SourceForge](https://sourceforge.net/projects/opencvlibrary/files/4.8.1/opencv-4.8.1-windows.exe/download))

Run it and set the extraction target to `D:\ThirdParty`. Do not tick "Add to the system PATH".

Result: `D:\ThirdParty\opencv\build\x64\vc16\{bin,lib}` - ready to use.
This machine currently has 4.8.0, so the file names read `480` instead of `481`; nothing else changes.


## 2. TensorRT 10.16.1.11

[Download TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/zip/TensorRT-10.16.1.11.Windows.amd64.cuda-12.9.zip)

Unzip into `D:\ThirdParty`. Do not run any installer and do not add `bin` to `PATH`.

Result: `D:\ThirdParty\TensorRT-10.16.1.11\{bin,include,lib}` - ready to use.

## 3. Build

```powershell
cmake -S D:\VS2022_Project\TensorRT-Alpha `
      -B D:\VS2022_Project\TensorRT-Alpha\build `
      -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11

cmake --build build --config Release -j 24
.\build\bin\Release\trt_alpha.exe list
```

Output: `build\bin\Release\trt_alpha.exe`, plus `samples\*` and every model family
(det / seg / obb / pose / cls / depth / sem). One-liner equivalent:
`.\scripts\build_windows_tensorrt-opencv.ps1`.

Dropping OpenCV is optional: add `-DTRT_ALPHA_WITH_OPENCV=OFF` and that build takes video
files and streams only, and can only `--show` (no `--save`, no images / directories / cameras).

`DeployRuntime.cmake` copies the OpenCV / TensorRT / CUDA DLLs next to the `.exe`,
so no `PATH` change is needed. Two deployment switches, both OFF by default:

```powershell
-DTRT_ALPHA_DEPLOY_BUILDER=ON     # only to build engines from ONNX, +2.8 GB
-DTRT_ALPHA_DEPLOY_CUDA_MATH=ON   # add if cublas64_*.dll / cudnn64_*.dll is reported missing, +730 MB
```

## 4. Uninstall

```powershell
Remove-Item -Recurse -Force D:\ThirdParty\opencv, D:\ThirdParty\TensorRT-10.16.1.11
```
