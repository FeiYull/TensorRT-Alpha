# Ubuntu 24.04

Only download and extract - nothing is installed into system paths and no environment variable is set.
The one exception is OpenCV, which has to be compiled here.

## 0. Dependencies

```bash
sudo apt update
sudo apt install -y build-essential cmake unzip wget
```

| Dependency | Version | Notes |
|---|---|---|
| gcc | 13.3 | |
| CMake | >= 3.25 | |
| CUDA Toolkit | 12.9 | CUDA C pre/post-processing, dual batch; `nvidia-smi` must work |
| [TensorRT](#1-tensorrt-1016111) | 10.16.1.11 | FP32 / FP16 / INT8 quantization |
| [OpenCV](#2-opencv-481) | >= 4.5 | decode + window + saving images |
| C++ | C++17 | |

This is the default build (`build`). Two optional extras have their own documents:

- FFmpeg, hardware video decoding: [build_ffmpeg.md](build_ffmpeg.md)
- FFmpeg + OpenGL, zero-copy multi-stream rendering: [build_ffmpeg_opengl.md](build_ffmpeg_opengl.md)

## 1. TensorRT 10.16.1.11

[Download TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz (8.0 GB)](https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz)

```bash
mkdir -p /home/feiyull/thirdparty && cd /home/feiyull/thirdparty
wget -O TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz \
  "https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/10.16.1/tars/TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz"
tar -xzf TensorRT-10.16.1.11.Linux.x86_64-gnu.cuda-12.9.tar.gz
```

Result: `/home/feiyull/thirdparty/TensorRT-10.16.1.11/{bin,include,lib}` - ready to use.
The URL directory is `10.16.1`, the file name is `10.16.1.11`. Do not install the `.deb`.

Optional, saves ~10 GB:
`rm -f /home/feiyull/thirdparty/TensorRT-10.16.1.11/lib/*.a /home/feiyull/thirdparty/TensorRT-10.16.1.11/lib/libnvinfer_builder_resource_*.so*`

## 2. OpenCV 4.8.1

Only the optional `-DTRT_ALPHA_WITH_OPENCV=OFF` build skips this section - and its
15 minute compile.

Dependencies:

```bash
sudo apt install -y libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
  libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
  libgtk-3-dev libcairo2-dev libgdk-pixbuf-2.0-dev libv4l-dev
```

[Download opencv-4.8.1.zip (source)](https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.1)

```bash
cd /home/feiyull/thirdparty
wget -O opencv-4.8.1.zip "https://codeload.github.com/opencv/opencv/zip/refs/tags/4.8.1"
unzip -q opencv-4.8.1.zip

cmake -S opencv-4.8.1 -B opencv-4.8.1-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/home/feiyull/thirdparty/opencv4.8.1 \
      -DBUILD_SHARED_LIBS=ON -DBUILD_opencv_world=ON -DWITH_CUDA=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF
cmake --build   opencv-4.8.1-build -j"$(nproc)"
cmake --install opencv-4.8.1-build          # no sudo
```

Result: `/home/feiyull/thirdparty/opencv4.8.1/{bin,include,lib,share}`

Three similar names, do not mix them up: `opencv-4.8.1/` is the extracted source, `opencv-4.8.1-build/` is the build tree, and `opencv4.8.1/` (no hyphen) is the install prefix. Step 3 passes `-DOpenCV_DIR=<prefix>/lib/cmake/opencv4`, because `OpenCVConfig.cmake` is installed there, not at the prefix root.

## 3. Build

```bash
cd ~/workspace/TensorRT-Alpha
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/home/feiyull/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/home/feiyull/thirdparty/TensorRT-10.16.1.11 \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc
cmake --build build -j"$(nproc)"
./build/bin/Release/trt_alpha list
```

Output: `build/bin/Release/trt_alpha`, plus `samples/` and every model family
(det / seg / obb / pose / cls / depth / sem). One-liner equivalent:
`./scripts/build_ubuntu_tensorrt-opencv.sh`.

`-DCMAKE_CUDA_COMPILER` is not optional here: `cmake/TensorRT.cmake` only looks in
`/usr/local/TensorRT`, and without a visible `nvcc` the CUDA probe fails with
"Failed to detect a default CUDA architecture".

Dropping OpenCV is optional: add `-DTRT_ALPHA_WITH_OPENCV=OFF` and that build takes video
files and streams only, and can only `--show` (no `--save`, no images / directories / cameras).

## 4. Uninstall

```bash
rm -rf /home/feiyull/thirdparty/opencv4.8.1 \
       /home/feiyull/thirdparty/opencv-4.8.1 \
       /home/feiyull/thirdparty/opencv-4.8.1-build \
       /home/feiyull/thirdparty/TensorRT-10.16.1.11
```
