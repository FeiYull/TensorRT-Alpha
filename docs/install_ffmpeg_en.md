# FFmpeg installation (Windows + Ubuntu)

Only needed for `--decode ffmpeg` and `--hwaccel nvdec`. Without it everything still runs, on OpenCV.

Both platforms use a **BtbN shared build** (`gpl-shared`): it ships `cuvid` (NVDEC) and `nvenc`,
and the repository links against the shared variant. The FFmpeg bundled with the distro / shipped
with OpenCV is **built without cuvid**, so NVDEC would simply not exist.

Version is pinned to `n9.0` (`libavcodec` 63 / `libavutil` 61) - the same build on Windows and Linux.

---

## 1. Windows

[Download ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip)

Extract into `D:\ThirdParty` and rename the inner folder to `ffmpeg-shared`
(no installer, do not add anything to `PATH`):

```powershell
Expand-Archive -Path "$env:USERPROFILE\Downloads\ffmpeg-n9.0-latest-win64-gpl-shared-9.0.zip" -DestinationPath D:\ThirdParty\tmp
$src = (Get-ChildItem D:\ThirdParty\tmp -Directory | Select-Object -First 1).FullName
Move-Item $src D:\ThirdParty\ffmpeg-shared
Remove-Item -Recurse -Force D:\ThirdParty\tmp
```

Result: `D:\ThirdParty\ffmpeg-shared\{bin,include,lib}` (`bin` holds `avcodec-63.dll` etc.)

Verify:

```powershell
D:\ThirdParty\ffmpeg-shared\bin\ffmpeg.exe -hwaccels
D:\ThirdParty\ffmpeg-shared\bin\ffmpeg.exe -hide_banner -decoders | Select-String cuvid
```

The first must list `cuda`, the second must list `h264_cuvid` / `hevc_cuvid`.

Configure (`FFMPEG_ROOT` already defaults to `D:/ThirdParty/ffmpeg-shared`; omit it if the path matches):

```powershell
cmake -S D:\VS2022_Project\TensorRT-Alpha -B D:\VS2022_Project\TensorRT-Alpha\build `
      -G "Visual Studio 17 2022" -A x64 `
      -DOpenCV_DIR=D:/ThirdParty/opencv/build/x64/vc16/lib `
      -DTensorRT_ROOT=D:/ThirdParty/TensorRT-10.16.1.11 `
      -DTRT_ALPHA_WITH_FFMPEG=ON `
      -DFFMPEG_ROOT=D:/ThirdParty/ffmpeg-shared

cmake --build build --config Release -j 24
```

`DeployRuntime.cmake` copies `bin\*.dll` next to the `.exe`, so no `PATH` change is needed.

---

## 2. Ubuntu 24.04

[Download ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz (66 MB)](https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz)

```bash
cd /home/feiyull/thirdparty
wget -O ffmpeg.tar.xz \
  "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/ffmpeg-n9.0-latest-linux64-gpl-shared-9.0.tar.xz"
mkdir -p ffmpeg-shared
tar -xJf ffmpeg.tar.xz --strip-components=1 -C ffmpeg-shared
rm ffmpeg.tar.xz
```

Result: `/home/feiyull/thirdparty/ffmpeg-shared/{bin,include,lib}` (235 MB)

Verify:

```bash
ffmpeg-shared/bin/ffmpeg -hwaccels                              # must list cuda
ffmpeg-shared/bin/ffmpeg -hide_banner -decoders | grep cuvid     # must list h264_cuvid
```

Configure:

```bash
cd ~/workspace/TensorRT-Alpha
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/home/feiyull/thirdparty/opencv4.8.1/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/home/feiyull/thirdparty/TensorRT-10.16.1.11 \
      -DTRT_ALPHA_WITH_FFMPEG=ON \
      -DFFMPEG_ROOT=/home/feiyull/thirdparty/ffmpeg-shared
cmake --build build -j"$(nproc)"
```

**No `export LD_LIBRARY_PATH` needed**: CMake emits an inheritable `DT_RPATH`
(`--disable-new-dtags`), so the binary finds `ffmpeg-shared/lib` by itself. BtbN's `.so` files carry
no RPATH of their own, and a plain `DT_RUNPATH` would miss `libswresample` because glibc does not
propagate it to transitive dependencies.

---

## 3. Run

```bash
# on Windows: .\build\bin\Release\trt_alpha.exe
./build/bin/Release/trt_alpha run --source data/people.mp4 --net yolov26 \
    --decode ffmpeg --hwaccel nvdec --save
```

- `--decode ffmpeg --hwaccel nvdec` = hardware decode + zero-copy device path; `--hwaccel none` = FFmpeg software decode.
- Images, image directories and cameras **always stay on OpenCV**, regardless of these two flags.
- The same values can be pinned in the `[source]` INI section (`decode` / `hwaccel` / `open_timeout_ms` /
  `read_timeout_ms` / `need_host_copy`).

## 4. Without FFmpeg

`-DTRT_ALPHA_WITH_FFMPEG` defaults to `OFF`. `--decode ffmpeg` then **fails loudly** instead of
silently falling back to OpenCV. Values: `OFF` / `ON` (FATAL if not found) / `AUTO` (use if found).

## 5. Uninstall

```powershell
# Windows
Remove-Item -Recurse -Force D:\ThirdParty\ffmpeg-shared
```

```bash
# Ubuntu
rm -rf /home/feiyull/thirdparty/ffmpeg-shared
```
