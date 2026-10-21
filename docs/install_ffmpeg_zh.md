# FFmpeg 安装（Windows + Ubuntu）

只给 `--decode ffmpeg` / `--hwaccel nvdec` 用。不装 FFmpeg 也能正常跑，默认走 OpenCV。

两个平台都用 **BtbN 的 shared 构建**（`gpl-shared`）：自带 `cuvid`（NVDEC）和 `nvenc`，
且仓库链的就是 shared 版。发行版 / 系统自带的 FFmpeg **没有编 cuvid**，替换上去 NVDEC 直接不存在。

版本跟仓库对齐：`n9.0`（`libavcodec` 63 / `libavutil` 61），Windows 与 Linux 同一份构建串。

---

## 1. Windows

[下载 ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip)

解压到 `D:\ThirdParty`，把里面的目录改名成 `ffmpeg-shared`（不要安装、不要加 PATH）：

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip" -DestinationPath D:\ThirdParty\tmp
$src = (Get-ChildItem D:\ThirdParty\tmp -Directory | Select-Object -First 1).FullName
Move-Item $src D:\ThirdParty\ffmpeg-shared
Remove-Item -Recurse -Force D:\ThirdParty\tmp
```

产物：`D:\ThirdParty\ffmpeg-shared\{bin,include,lib}`（`bin` 里是 `avcodec-63.dll` 等）

校验：

```powershell
D:\ThirdParty\ffmpeg-shared\bin\ffmpeg.exe -hwaccels
D:\ThirdParty\ffmpeg-shared\bin\ffmpeg.exe -hide_banner -decoders | Select-String cuvid
```

第一条要出现 `cuda`，第二条要出现 `h264_cuvid` / `hevc_cuvid`。

配置（`FFMPEG_ROOT` 的默认值就是 `D:/ThirdParty/ffmpeg-shared`，路径一致时可省）：

```powershell
cmake -S D:\VS2022_Project\TensorRT-Alpha -B D:\VS2022_Project\TensorRT-Alpha\build `
      -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11 `
      -DTRT_ALPHA_WITH_FFMPEG=ON `
      -DFFMPEG_ROOT=D:/ThirdParty/ffmpeg-shared

cmake --build build --config Release -j 24
```

`DeployRuntime.cmake` 会把 `bin\*.dll` 拷到 `exe` 旁边，不用改 PATH。

---

## 2. Ubuntu 24.04

[下载 ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz (66 MB)](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz)

```bash
cd /home/feiyull/thirdparty
wget -O ffmpeg.tar.xz \
  "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz"
mkdir -p ffmpeg-shared
tar -xJf ffmpeg.tar.xz --strip-components=1 -C ffmpeg-shared
rm ffmpeg.tar.xz
```

产物：`/home/feiyull/thirdparty/ffmpeg-shared/{bin,include,lib}`（235 MB）

校验：

```bash
ffmpeg-shared/bin/ffmpeg -hwaccels                              # 要出现 cuda
ffmpeg-shared/bin/ffmpeg -hide_banner -decoders | grep cuvid     # 要出现 h264_cuvid
```

配置：

```bash
cd ~/workspace/TensorRT-Alpha
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/home/feiyull/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/home/feiyull/thirdparty/TensorRT-10.16.1.11 \
      -DTRT_ALPHA_WITH_FFMPEG=ON \
      -DFFMPEG_ROOT=/home/feiyull/thirdparty/ffmpeg-shared
cmake --build build -j"$(nproc)"
```

**不需要 `export LD_LIBRARY_PATH`**：CMake 用 `--disable-new-dtags` 写成继承式 `DT_RPATH`，
二进制自己找得到 `ffmpeg-shared/lib`（BtbN 的 `.so` 自身不带 RPATH，靠 `DT_RUNPATH` 会漏掉
`libswresample`）。

---

## 3. 运行

```bash
# Windows 把 trt_alpha 换成 .\build\bin\Release\trt_alpha.exe
./build/bin/Release/trt_alpha run --source data/people.mp4 --net yolov26 \
    --decode ffmpeg --hwaccel nvdec --save
```

- `--decode ffmpeg --hwaccel nvdec` = 硬解 + 显存直通；`--hwaccel none` = FFmpeg 软解。
- 图片、图片目录、摄像头**永远走 OpenCV**，不受这两个开关影响。
- 配置文件的 `[source]` 段可写死同样的值（`decode` / `hwaccel` / `open_timeout_ms` / `read_timeout_ms` / `need_host_copy`）。

## 4. 不开 FFmpeg 时

`-DTRT_ALPHA_WITH_FFMPEG` 默认 `OFF`。此时 `--decode ffmpeg` **直接报错**，不会静默退回 OpenCV。
取值：`OFF` / `ON`（找不到就 FATAL）/ `AUTO`（找到就用）。

## 5. 卸载

```powershell
# Windows
Remove-Item -Recurse -Force D:\ThirdParty\ffmpeg-shared
```

```bash
# Ubuntu
rm -rf /home/feiyull/thirdparty/ffmpeg-shared
```
