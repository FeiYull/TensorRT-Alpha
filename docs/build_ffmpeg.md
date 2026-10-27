# FFmpeg hardware decoding — TensorRT + OpenCV + FFmpeg

One extra dependency over the default install, built into `build-ffm`. Video is decoded by NVDEC;
OpenCV still owns the window and the output files.

Prerequisite: the dependencies from [Windows](install_for_windows.md) or
[Ubuntu 24.04](install_for_ubuntu24.04.md) are already in place.

## 1. Install FFmpeg 9.0

It must be a `gpl-shared` build with cuvid, otherwise NVDEC is not there at all.

### Windows

[Download ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip)

Unzip into `D:\ThirdParty` and rename the extracted folder to `ffmpeg-shared`.

Result: `D:\ThirdParty\ffmpeg-shared\{bin,include,lib}`

```powershell
D:\ThirdParty\ffmpeg-shared\bin\ffmpeg.exe -hwaccels                                      # must list cuda
D:\ThirdParty\ffmpeg-shared\bin\ffmpeg.exe -hide_banner -decoders | Select-String cuvid    # must contain h264_cuvid
```

### Ubuntu 24.04

The distribution `libavcodec-dev` is not built with cuvid - do not use it.

[Download ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz (66 MB)](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz)

```bash
mkdir -p ~/thirdparty && cd ~/thirdparty
wget -O ffmpeg.tar.xz \
  "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz"
mkdir -p ffmpeg-shared
tar -xJf ffmpeg.tar.xz --strip-components=1 -C ffmpeg-shared
rm ffmpeg.tar.xz
ffmpeg-shared/bin/ffmpeg -hwaccels | grep cuda                                 # must list cuda
ffmpeg-shared/bin/ffmpeg -hide_banner -decoders | grep cuvid                   # must contain h264_cuvid
```

Result: `~/thirdparty/ffmpeg-shared/{bin,include,lib}`

## 2. Build

Output directory `build-ffm`; the configure log must print `FFmpeg: ON`.

### Windows

```powershell
cmake -S D:\VS2022_Project\TensorRT-Alpha `
      -B D:\VS2022_Project\TensorRT-Alpha\build-ffm `
      -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11 `
      -DTRT_ALPHA_WITH_FFMPEG=ON `
      -DFFMPEG_ROOT=D:/ThirdParty/ffmpeg-shared

cmake --build build-ffm --config Release -j 24                        # whole tree (CLI + samples)
cmake --build build-ffm --config Release --target trt_alpha -j 24     # executable only
.\build-ffm\bin\Release\trt_alpha.exe list
```

Executable: `build-ffm\bin\Release\trt_alpha.exe` (the runtime DLLs are copied next to it).

One-liner script: `.\scripts\build_windows_tensorrt-opencv-ffmpeg.ps1`

### Ubuntu 24.04

```bash
cmake -S ~/workspace/TensorRT-Alpha -B ~/workspace/TensorRT-Alpha/build-ffm \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_COMPILER=/usr/local/cuda-12.9/bin/nvcc \
      -DOpenCV_DIR=$HOME/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=$HOME/thirdparty/TensorRT-10.16.1.11 \
      -DTRT_ALPHA_WITH_FFMPEG=ON \
      -DFFMPEG_ROOT=$HOME/thirdparty/ffmpeg-shared

cmake --build build-ffm -j"$(nproc)"                       # whole tree
cmake --build build-ffm --target trt_alpha -j"$(nproc)"    # executable only
./build-ffm/bin/Release/trt_alpha list
```

Executable: `build-ffm/bin/Release/trt_alpha`

Script: `./scripts/build_ubuntu_tensorrt-opencv-ffmpeg.sh`

## 3. Run

```bash
./build-ffm/bin/Release/trt_alpha run --net yolov26 --source data/people.mp4 --show --decode ffmpeg --hwaccel nvdec
./build-ffm/bin/Release/trt_alpha run --net yolov26 --source data/people.mp4 --save output/det/ --decode ffmpeg --hwaccel nvdec
./build-ffm/bin/Release/trt_alpha run --net yolov26 --source rtsp://host:554/s1 --show --decode ffmpeg --hwaccel nvdec
```

On Windows use `.\build-ffm\bin\Release\trt_alpha.exe` instead.

- `--hwaccel nvdec` forces hardware decoding; `auto` uses NVDEC when available; `none` is plain CPU software decoding.
- Images, directories and cameras always go through OpenCV and ignore `--decode`.
- Without `--decode ffmpeg` the behaviour is exactly the default build (OpenCV decoding).

FFmpeg is a shared build on Linux and the binary finds the `.so` files through `DT_RPATH`.
If you move or rename `ffmpeg-shared`, re-run configure, otherwise the runtime reports
`libavcodec.so.*: cannot open shared object file`.

## 4. Uninstall

```powershell
Remove-Item -Recurse -Force D:\ThirdParty\ffmpeg-shared
```

```bash
rm -rf ~/thirdparty/ffmpeg-shared
```
