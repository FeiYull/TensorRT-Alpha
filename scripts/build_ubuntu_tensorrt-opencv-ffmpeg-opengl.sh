#!/usr/bin/env bash
# =============================================================================
#  build_ubuntu_tensorrt-opencv-ffmpeg-opengl.sh
# -----------------------------------------------------------------------------
#  Ubuntu build :  CUDA + TensorRT + OpenCV + FFmpeg + OpenGL(GLFW)
#  Build dir    :  build-gl
#  Output       :  build-gl/bin/<BUILD_TYPE>/trt_alpha
#
#  Unlocks: --decode ffmpeg --hwaccel nvdec   (NVDEC)
#           --render gl                      (CUDA-GL interop, multi-stream)
#
#  Usage:
#      ./scripts/build_ubuntu_tensorrt-opencv-ffmpeg-opengl.sh
#      BUILD_TYPE=Debug ./scripts/build_ubuntu_tensorrt-opencv-ffmpeg-opengl.sh
#
#  Env overrides (defaults match docs/install_for_ubuntu24.04.md):
#      TRT_ROOT  OPENCV_DIR  FFMPEG_ROOT  GLFW_ROOT  BUILD_DIR  BUILD_TYPE
#      JOBS  CUDA_COMPILER  BUILD_TESTS=OFF  BUILD_SAMPLES=ON
# =============================================================================
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

TRT_ROOT="${TRT_ROOT:-$HOME/thirdparty/TensorRT-10.16.1.11}"
OPENCV_DIR="${OPENCV_DIR:-$HOME/thirdparty/opencv4.8.1/lib/cmake/opencv4}"
# Must be a shared build that carries cuvid; the distro package has none.
FFMPEG_ROOT="${FFMPEG_ROOT:-$HOME/thirdparty/ffmpeg-shared}"
GLFW_ROOT="${GLFW_ROOT:-$HOME/thirdparty/glfw}"
BUILD_DIR="${BUILD_DIR:-$ROOT/build-gl}"
BUILD_TYPE="${BUILD_TYPE:-Release}"
JOBS="${JOBS:-$(nproc)}"
# Explicitly: cmake/TensorRT.cmake only knows /usr/local/TensorRT, and without a
# visible nvcc the CUDA probe dies with "Failed to detect a default CUDA architecture".
CUDA_COMPILER="${CUDA_COMPILER:-/usr/local/cuda-12.9/bin/nvcc}"
BUILD_TESTS="${BUILD_TESTS:-OFF}"
BUILD_SAMPLES="${BUILD_SAMPLES:-ON}"

echo "[build] CUDA + TensorRT + OpenCV + FFmpeg + OpenGL  ->  $BUILD_DIR"

cmake -S "$ROOT" -B "$BUILD_DIR" \
      -DCMAKE_BUILD_TYPE="$BUILD_TYPE" \
      -DCMAKE_CUDA_COMPILER="$CUDA_COMPILER" \
      -DTensorRT_ROOT="$TRT_ROOT" \
      -DOpenCV_DIR="$OPENCV_DIR" \
      -DFFMPEG_ROOT="$FFMPEG_ROOT" \
      -DGLFW_ROOT="$GLFW_ROOT" \
      -DTRT_ALPHA_WITH_OPENCV=ON \
      -DTRT_ALPHA_WITH_FFMPEG=ON \
      -DTRT_ALPHA_WITH_GL=ON \
      -DTRT_ALPHA_BUILD_TESTS="$BUILD_TESTS" \
      -DTRT_ALPHA_BUILD_SAMPLES="$BUILD_SAMPLES"

# Unix Makefiles is single-config: --config is ignored, BUILD_TYPE decides the dir.
cmake --build "$BUILD_DIR" -j "$JOBS"

EXE="$BUILD_DIR/bin/${BUILD_TYPE}/trt_alpha"
MODELS="$("$EXE" list | wc -l)"
echo "[ok] $EXE   ($MODELS lines from 'list')"
