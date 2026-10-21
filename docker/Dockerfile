# =============================================================================
#  TensorRT-Alpha —— 构建 / 运行镜像
# -----------------------------------------------------------------------------
#  在仓库根目录执行（构建上下文必须是仓库根目录，.dockerignore 也在根目录）：
#
#    docker build -f docker/Dockerfile -t trt-alpha:latest .
#
#    # 看模型清单
#    docker run --rm --gpus all trt-alpha:latest list
#
#    # 跑一张图（engines/ 与 save/ 从宿主机挂进来）
#    docker run --rm --gpus all \
#      -v "$PWD/engines:/app/engines" -v "$PWD/save:/app/save" \
#      trt-alpha:latest run --net yolov26 --source data/12801.jpg --save
#
#    # 跑一段视频（走 FFmpeg + NVDEC 这条零拷贝路径）
#    docker run --rm --gpus all \
#      -v "$PWD/engines:/app/engines" -v "$PWD/save:/app/save" \
#      trt-alpha:latest run --net yolov26 --source data/people.mp4 \
#          --decode ffmpeg --hwaccel nvdec --save
#
#    # 进 shell（镜像里带完整源码和 build/ 目录）
#    docker run --rm -it --gpus all --entrypoint bash trt-alpha:latest
#
#    # 只跑到编译段（要 build/ 产物、不打运行镜像）
#    docker build -f docker/Dockerfile --target build -t trt-alpha:build .
#
#  设计要点
#  ----------------------------------------------------------------------------
#  · 分 5 段构建：opencv / ffmpeg / tensorrt 三个"依赖产出"段跑在精简的 ubuntu 上，
#    编译段用 CUDA devel（需要 nvcc），最终运行段用 CUDA runtime（体积小一半）。
#  · 依赖装法与本仓库 docs/install_for_ubuntu24.04.md 完全一致：
#      TensorRT 官方 tar.gz 解压使用（不装 .deb、不落系统路径）
#      OpenCV   4.8.1 源码编译，装到独立前缀 /opt/opencv
#      FFmpeg   BtbN gpl-shared 包解压到 /opt/ffmpeg（带 cuvid，distro 的
#               libavcodec-dev 没有 NVDEC，所以不能拿 apt 版本顶替）
#  · FFmpeg 是编译期 ON：CMake 里 -DTRT_ALPHA_WITH_FFMPEG=ON 会让它成为硬运行时
#    依赖，所以运行段必须把 /opt/ffmpeg/lib 一起带上并注册到 ld.so.conf。
#  · 镜像内不设 LD_LIBRARY_PATH：
#      仓库自己的二进制靠 CMake 写进 RUNPATH 的绝对路径（/opt/TensorRT/lib、
#      /opt/opencv/lib、/opt/ffmpeg/lib），开箱即用；独立二进制（trtexec 等）靠
#      /etc/ld.so.conf.d 注册 —— 容器内这是常规做法。
#  · 基础镜像走 NGC（nvcr.io）而不是 Docker Hub：nvidia/cuda 的 tag 与
#    Docker Hub 完全同名，且 nvcr.io 在国内可直连。
#  · NVDEC 要宿主机把驱动库挂进容器：NVIDIA_DRIVER_CAPABILITIES 因此含 video
#    （只给 compute,utility 时 av_hwdevice_ctx_create 会直接失败）。
# =============================================================================

ARG NGC_BASE=nvcr.io/nvidia/cuda
ARG CUDA_VERSION=12.9.1
ARG UBUNTU_TAG=ubuntu24.04
ARG UBUNTU_BASE=ubuntu:24.04

ARG OPENCV_VERSION=4.8.1
ARG TRT_VERSION=10.16.1.11
# TRT 官方下载目录用的是 10.16.1（三位），跟完整的 10.16.1.11 不是一个串
ARG TRT_DIR=10.16.1
ARG TRT_CUDA=12.9

# FFmpeg：直接用 BtbN 的 linux64 gpl-shared 包（自带 cuvid / nvenc）。
# 不走 apt 的 libavcodec-dev —— 发行版那个没编 cuvid，NVDEC 会直接不存在。
# 版本跟着 Windows 侧走（n9.0.x），保证两边头文件/ABI 一致。
# 换版本只改这一个 ARG，下载地址由它拼出来。
ARG FFMPEG_BUILD=ffmpeg-n9.0-latest-linux64-gpl-shared-9.0

# 并行编译任务数。OpenCV 单个 g++ 峰值 0.5~1 GB 内存，16 在 24 GB 内存上够用；
# 内存充裕的机器可以 --build-arg JOBS=32。
ARG JOBS=16

# 0 = 不把 TensorRT 的 nvinfer_builder_resource_*.so（约 2.9 GB）打进镜像。
#     这几个库只在"从 ONNX 现场构建 engine"（trt_alpha build / trtexec）时用，
#     跑现成的 .trt 完全不需要 —— 与仓库 cmake/DeployRuntime.cmake 的默认策略一致。
# 1 = 打进去，此时才可以用 trtexec 构建引擎。
ARG WITH_BUILDER=0

# 目标 GPU 架构。默认覆盖主流卡（含 RTX 50 系的 sm_120）。
ARG CUDA_ARCH=75;86;89;120

# OpenCV 的 C/C++ 依赖。运行段也要装（OpenCV 的 .so 是硬 DT_NEEDED 这些库），
# 所以抽出来共用，避免两处清单漂移。
ARG LIB_DEPS="libjpeg-dev libpng-dev libtiff-dev libwebp-dev \
libavcodec-dev libavformat-dev libavutil-dev libswscale-dev \
libgtk-3-dev libcairo2-dev libgdk-pixbuf-2.0-dev libv4l-dev"


# =============================================================================
#  stage 1 / 4 —— OpenCV 4.8.1（源码编译，world 模块）
#  WITH_CUDA=OFF：本仓库 OpenCV.cmake 里 OpenCV_CUDA 也是 OFF，
#  且只用到 core / imgproc / imgcodecs / videoio / highgui。
# =============================================================================
FROM ${UBUNTU_BASE} AS opencv
ARG OPENCV_VERSION
ARG JOBS
ARG LIB_DEPS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake unzip ca-certificates curl ${LIB_DEPS} \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 3 -o /tmp/opencv.zip \
      "https://codeload.github.com/opencv/opencv/zip/refs/tags/${OPENCV_VERSION}" \
 && unzip -q /tmp/opencv.zip -d /tmp \
 && rm -f /tmp/opencv.zip

RUN cmake -S /tmp/opencv-${OPENCV_VERSION} -B /tmp/ocv-build \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_INSTALL_PREFIX=/opt/opencv \
      -DBUILD_SHARED_LIBS=ON \
      -DBUILD_opencv_world=ON \
      -DWITH_CUDA=OFF \
      -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF \
      -DBUILD_opencv_python2=OFF -DBUILD_opencv_python3=OFF -DBUILD_JAVA=OFF \
 && cmake --build /tmp/ocv-build -j"${JOBS}" \
 && cmake --install /tmp/ocv-build \
 && rm -rf /tmp/ocv-build /tmp/opencv-${OPENCV_VERSION} \
 && ls -l /opt/opencv/lib/cmake/opencv4/OpenCVConfig.cmake


# =============================================================================
#  stage 2 / 5 —— FFmpeg（BtbN gpl-shared，解压即用）
#  只留 include/ 与 lib/：bin/ 里的 ffmpeg / ffplay CLI 我们不用（省约 66 MB）。
#  运行段靠 /etc/ld.so.conf.d 找到 /opt/ffmpeg/lib —— 注意 BtbN 的 .so 自身不带
#  RPATH，所以容器里必须注册路径，不能只靠 RUNPATH（RUNPATH 不传给传递依赖）。
# =============================================================================
FROM ${UBUNTU_BASE} AS ffmpeg
ARG FFMPEG_BUILD
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl xz-utils \
 && rm -rf /var/lib/apt/lists/*

RUN curl -fsSL --retry 3 --retry-delay 5 -o /tmp/ffmpeg.tar.xz \
      "https://github.com/BtbN/FFmpeg-Builds/releases/download/latest/${FFMPEG_BUILD}.tar.xz" \
 && mkdir -p /tmp/x /opt/ffmpeg \
 && tar -xJf /tmp/ffmpeg.tar.xz -C /tmp/x --strip-components=1 \
 && rm -f /tmp/ffmpeg.tar.xz \
 && mv /tmp/x/include /tmp/x/lib /opt/ffmpeg/ \
 && rm -rf /tmp/x \
 && grep FFMPEG_VERSION /opt/ffmpeg/include/libavutil/ffversion.h \
 && test -f /opt/ffmpeg/lib/libavcodec.so \
 && du -sh /opt/ffmpeg


# =============================================================================
#  stage 3 / 5 —— TensorRT（官方 tar.gz，解压即用）
# =============================================================================
FROM ${UBUNTU_BASE} AS tensorrt
ARG TRT_VERSION
ARG TRT_DIR
ARG TRT_CUDA
ARG WITH_BUILDER
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates curl \
 && rm -rf /var/lib/apt/lists/*

# 下载单独占一层：tar.gz 有 8.0 GB，下载成功后就进构建缓存，
# 之后再改解包逻辑/排错都不用重新下载。
RUN curl -fsSL --retry 3 --retry-delay 5 -o /tmp/trt.tar.gz \
      "https://developer.download.nvidia.cn/compute/machine-learning/tensorrt/${TRT_DIR}/tars/TensorRT-${TRT_VERSION}.Linux.x86_64-gnu.cuda-${TRT_CUDA}.tar.gz" \
 && ls -l /tmp/trt.tar.gz

# 官方包里 lib/ 混着 7 GB+ 的静态库（*.a）和约 2.9 GB 的
# libnvinfer_builder_resource_*.so（每个 SM 架构一份，只在"从 ONNX 现场构建
# engine"时才加载），链接和跑现成 .trt 都用不到，所以剔掉。
#
# 这里用 --exclude 而不是 --wildcards 挑文件：后者只要有一个模式匹配不到成员，
# tar 就会以 "TensorRT-...: Not found in archive" 整体报错退出（实测本包就触发过），
# 而 --exclude 少匹配一个模式没有任何副作用。
RUN ex=""; \
    if [ "${WITH_BUILDER}" = "0" ]; then \
        ex="--exclude=TensorRT-*/lib/libnvinfer_builder_resource_*.so*"; \
    fi; \
    mkdir -p /tmp/x \
 && tar -xzf /tmp/trt.tar.gz -C /tmp/x --wildcards \
      --exclude="TensorRT-*/lib/*.a" \
      --exclude="TensorRT-*/doc/*" \
      ${ex} \
 && rm -f /tmp/trt.tar.gz \
 && mv /tmp/x/TensorRT-${TRT_VERSION} /opt/TensorRT \
 && rm -rf /tmp/x \
 && if [ "${WITH_BUILDER}" = "0" ]; then \
        if ls /opt/TensorRT/lib/libnvinfer_builder_resource_* >/dev/null 2>&1; then \
            echo "[TensorRT] ERROR: builder resources were not excluded"; exit 1; \
        fi; \
        echo "[TensorRT] builder resources excluded (WITH_BUILDER=0)"; \
    fi \
 && ls /opt/TensorRT \
 && du -sh /opt/TensorRT


# =============================================================================
#  stage 4 / 5 —— 编译本仓库（需要 nvcc，所以用 CUDA devel 基础镜像）
#  产物留在 /app/build/bin/Release/trt_alpha。
# =============================================================================
FROM ${NGC_BASE}:${CUDA_VERSION}-devel-${UBUNTU_TAG} AS build
ARG JOBS
ARG CUDA_ARCH
ARG LIB_DEPS
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      build-essential cmake git ${LIB_DEPS} \
 && rm -rf /var/lib/apt/lists/*

COPY --from=opencv   /opt/opencv   /opt/opencv
COPY --from=ffmpeg   /opt/ffmpeg   /opt/ffmpeg
COPY --from=tensorrt /opt/TensorRT /opt/TensorRT

WORKDIR /app
COPY . /app

# 构建镜像里没有 GPU，CMakeLists 会走"无本地 GPU"分支；
# 这里显式给 -DTRT_ALPHA_CUDA_ARCH，保证结果与宿主机无关、可复现。
# FFmpeg 显式 ON：镜像里既然装了，就不允许它悄悄变成 OFF（"以为编进去了其实没有"）。
RUN cmake -S /app -B /app/build \
      -DCMAKE_BUILD_TYPE=Release \
      -DOpenCV_DIR=/opt/opencv/lib/cmake/opencv4 \
      -DTensorRT_ROOT=/opt/TensorRT \
      -DTRT_ALPHA_CUDA_ARCH="${CUDA_ARCH}" \
      -DTRT_ALPHA_WITH_FFMPEG=ON \
      -DFFMPEG_ROOT=/opt/ffmpeg \
 && cmake --build /app/build -j"${JOBS}" \
 && /app/build/bin/Release/trt_alpha list \
 && /app/build/bin/Release/trt_alpha --help | grep -q -- "--decode"


# =============================================================================
#  stage 5 / 5 —— 最终运行镜像（CUDA runtime，不含 nvcc）
# =============================================================================
FROM ${NGC_BASE}:${CUDA_VERSION}-runtime-${UBUNTU_TAG} AS runtime
ARG TRT_VERSION
ARG OPENCV_VERSION
ARG FFMPEG_BUILD
ARG LIB_DEPS
ENV DEBIAN_FRONTEND=noninteractive

# OpenCV 的 .so 硬依赖这些运行时库（GTK / FFmpeg / 图像编解码），必须装上
RUN apt-get update \
 && apt-get install -y --no-install-recommends ca-certificates ${LIB_DEPS} \
 && rm -rf /var/lib/apt/lists/*

COPY --from=opencv   /opt/opencv   /opt/opencv
COPY --from=ffmpeg   /opt/ffmpeg   /opt/ffmpeg
COPY --from=tensorrt /opt/TensorRT /opt/TensorRT
COPY --from=build    /app          /app

# 让 trtexec 之类的独立二进制也能找到库；仓库自己的二进制靠 RPATH。
# /opt/ffmpeg/lib 必须在这里：BtbN 的 .so 自身没有 RPATH，libavcodec 找它的
# libswresample / libavutil 时只会走 ldcache + 默认目录（RUNPATH 不传给传递依赖）。
RUN printf '/opt/TensorRT/lib\n/opt/opencv/lib\n/opt/ffmpeg/lib\n' \
      > /etc/ld.so.conf.d/trt-alpha.conf \
 && ldconfig

# 构建期兜底：任何一个动态库没解析出来就直接让镜像构建失败，
# 而不是等到用户 docker run 时才报错。
RUN ldd /app/build/bin/Release/trt_alpha > /tmp/ldd.txt 2>&1 || true \
 && if grep -q "not found" /tmp/ldd.txt; then \
        echo "=== missing shared libraries ==="; cat /tmp/ldd.txt; exit 1; \
    fi \
 && grep -E "libnvinfer\.so|libnvinfer_plugin|libcudart|libopencv_world|libavcodec|libavformat" /tmp/ldd.txt

ENV NVIDIA_VISIBLE_DEVICES=all \
    NVIDIA_DRIVER_CAPABILITIES=compute,utility,video

WORKDIR /app
# 仓库自己的二进制 / trtexec / opencv_version 都进 PATH，方便 docker run 直接调
ENV PATH=/app/build/bin/Release:/opt/TensorRT/bin:/opt/opencv/bin:$PATH
ENTRYPOINT ["/app/build/bin/Release/trt_alpha"]
CMD ["list"]

LABEL org.opencontainers.image.title="TensorRT-Alpha" \
      org.opencontainers.image.description="C++ TensorRT inference framework (CUDA + TensorRT + OpenCV)" \
      org.opencontainers.image.source="https://github.com/FeiYull/TensorRT-Alpha" \
      org.opencontainers.image.licenses="MIT" \
      tensorrt.version="${TRT_VERSION}" \
      opencv.version="${OPENCV_VERSION}" \
      ffmpeg.build="${FFMPEG_BUILD}"
