# =============================================================================
#  trt_alpha —— 顶层 CMakeLists
# -----------------------------------------------------------------------------
#  职责：
#    * 声明 C++17 / CUDA 17 标准
#    * 自动探测 CUDA 架构（nvidia-smi）
#    * 收集 src/ 下所有 .cpp / .cu，编成 trt_alpha_core 静态库
#    * 加载依赖（OpenCV / TensorRT，见 cmake/）
#    * 挂上 test/ 子目录
# =============================================================================
cmake_minimum_required(VERSION 3.25)

project(TensorRT_Alpha LANGUAGES CXX CUDA)


# =============================================================================
# CUDA 架构自动选择
# -----------------------------------------------------------------------------
# 优先级：
#   1. -DTRT_ALPHA_CUDA_ARCH=<arch> 用户显式指定
#   2. 本机 nvidia-smi 探测
#   3. 回退主流消费卡列表
#
# 注意：不用 CMAKE_CUDA_ARCHITECTURES 判断"用户是否指定"，因为
# project(... CUDA) 会自动给它设默认值（CUDA 12.9 下是 52）。
# 用自己的 TRT_ALPHA_CUDA_ARCH 作为唯一判据。
# set(... CACHE ... FORCE) 确保覆盖 CMake 隐式设的默认值。
# =============================================================================
if(DEFINED TRT_ALPHA_CUDA_ARCH AND NOT TRT_ALPHA_CUDA_ARCH STREQUAL "")
    set(CMAKE_CUDA_ARCHITECTURES "${TRT_ALPHA_CUDA_ARCH}"
        CACHE STRING "CUDA architectures" FORCE)
    message(STATUS "[CUDA] arch from -DTRT_ALPHA_CUDA_ARCH: ${CMAKE_CUDA_ARCHITECTURES}")
else()
    find_program(TRT_ALPHA_NVIDIA_SMI nvidia-smi)
    if(TRT_ALPHA_NVIDIA_SMI)
        execute_process(
            COMMAND "${TRT_ALPHA_NVIDIA_SMI}" --query-gpu=compute_cap --format=csv,noheader
            OUTPUT_VARIABLE TRT_ALPHA_GPU_CC
            OUTPUT_STRIP_TRAILING_WHITESPACE
            ERROR_QUIET
        )
    endif()

    if(TRT_ALPHA_GPU_CC)
        # 多 GPU 时 nvidia-smi 返回多行（compute_cap 一行一张卡）；取第一张即可。
        string(REPLACE "\n" ";" TRT_ALPHA_GPU_CC_LIST "${TRT_ALPHA_GPU_CC}")
        list(GET TRT_ALPHA_GPU_CC_LIST 0 TRT_ALPHA_GPU_CC)
        string(STRIP "${TRT_ALPHA_GPU_CC}" TRT_ALPHA_GPU_CC)
        string(REPLACE "." "" TRT_ALPHA_CUDA_ARCH "${TRT_ALPHA_GPU_CC}")
        set(CMAKE_CUDA_ARCHITECTURES "${TRT_ALPHA_CUDA_ARCH}"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "[CUDA] arch detected from local GPU: ${TRT_ALPHA_CUDA_ARCH}")
    else()
        set(CMAKE_CUDA_ARCHITECTURES "75;86;89;120"
            CACHE STRING "CUDA architectures" FORCE)
        message(STATUS "[CUDA] no local GPU detected, fallback: 75;86;89;120")
    endif()
endif()


# =============================================================================
# 语言标准
# =============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)


# =============================================================================
# 构建类型默认（单配置生成器）
# =============================================================================
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()


# =============================================================================
# MSVC 中文乱码：源文件按 UTF-8 编译
# =============================================================================
if(MSVC)
  add_compile_options("$<$<COMPILE_LANGUAGE:C>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CXX>:/utf-8>")
  add_compile_options("$<$<COMPILE_LANGUAGE:CUDA>:-Xcompiler=/utf-8>")
endif()


# =============================================================================
# 依赖模块目录 + 加载依赖
# =============================================================================
list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")

include(OpenCV)
include(GL)              # 可选 OpenGL 后端（默认 OFF；-DTRT_ALPHA_WITH_GL=ON 开启）
include(TensorRT)
include(DeployRuntime)   # 提供 trt_alpha_deploy_runtime(<target>)：把第三方 DLL 拷到 exe 旁


# =============================================================================
# 统一输出目录
# =============================================================================
# 按构建类型分子目录：多配置生成器用 $<CONFIG>，单配置生成器用 CMAKE_BUILD_TYPE。
# 产物：bin/Debug/trt_alpha、bin/Release/trt_alpha（Windows 仍是 bin\Release\trt_alpha.exe）。
# 注意：单配置生成器（Unix Makefiles / Ninja）下 `cmake --build build --config Release`
# 的 --config 会被忽略，切配置必须重新 `cmake -S . -B build -DCMAKE_BUILD_TYPE=Debug`。
if(CMAKE_CONFIGURATION_TYPES)
  set(_trt_alpha_cfg "$<CONFIG>")
else()
  set(_trt_alpha_cfg "${CMAKE_BUILD_TYPE}")
endif()
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${_trt_alpha_cfg}")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin/${_trt_alpha_cfg}")
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib/${_trt_alpha_cfg}")


# =============================================================================
# trt_alpha_core —— 底座静态库
# -----------------------------------------------------------------------------
# 收集 src/ 下所有 .cpp 和 .cu，编成一个静态库。
# CUDA_SEPARABLE_COMPILATION + CUDA_RESOLVE_DEVICE_SYMBOLS 是 .cu 与 .cpp
# 跨编译单元互相引用时必须的（否则 __cudaRegisterLinkedBinary_* 链接失败）。
# =============================================================================
file(GLOB_RECURSE TRT_ALPHA_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_SOURCE_DIR}/src/*.cpp"
  "${CMAKE_SOURCE_DIR}/src/*.cu")

# 🔴 可选后端：关掉某个后端时，必须把它独占的源文件从编译列表里摘掉。
#    GLOB 会自动收新文件（CONFIGURE_DEPENDS），所以【新增后端独占文件时
#    要在这里同步登记】，否则 OFF 构建会因为找不到第三方头而失败。
if(NOT TRT_ALPHA_WITH_OPENCV)
  list(FILTER TRT_ALPHA_SOURCES EXCLUDE REGEX "/opencv_(source|renderer)\\.cpp$")
  message(STATUS "Sources: excluded opencv_source.cpp / opencv_renderer.cpp")
endif()
if(NOT TRT_ALPHA_WITH_GL)
  list(FILTER TRT_ALPHA_SOURCES EXCLUDE REGEX "/gl_(renderer|context)\\.cpp$")
endif()

add_library(trt_alpha_core STATIC ${TRT_ALPHA_SOURCES})

# 编译期写入源码根路径：out-of-tree 构建时给 Paths 一个兜底
target_compile_definitions(trt_alpha_core PRIVATE
    TRT_ALPHA_ROOT_DIR="${CMAKE_SOURCE_DIR}"
)

# =============================================================================
# 日志级别（按构建类型自动设）
# -----------------------------------------------------------------------------
#   Debug / RelWithDebInfo  →  DEBUG（全开，冗余日志，崩溃可诊断）
#   Release / MinSizeRel    →  INFO （只保留关键信息，DEBUG 零开销）
#   PUBLIC 让所有依赖 trt_alpha_core 的 target 继承。
# =============================================================================
target_compile_definitions(trt_alpha_core PUBLIC
    $<$<CONFIG:Debug>:TRT_ALPHA_LOG_MIN_LEVEL=0>
    $<$<CONFIG:RelWithDebInfo>:TRT_ALPHA_LOG_MIN_LEVEL=0>
    $<$<CONFIG:Release>:TRT_ALPHA_LOG_MIN_LEVEL=1>
    $<$<CONFIG:MinSizeRel>:TRT_ALPHA_LOG_MIN_LEVEL=2>
)

# 上面的 PUBLIC 定义已覆盖本 target 自身及其依赖者；
# Release 下 TRT_LOG_DEBUG 变成 do {} while(0)，零开销；Debug 全开。


set_target_properties(trt_alpha_core PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_RESOLVE_DEVICE_SYMBOLS ON
)

target_include_directories(trt_alpha_core
  PUBLIC "${CMAKE_SOURCE_DIR}/include")

# =============================================================================
# FFmpeg（可选）—— 视频解码后端
# -----------------------------------------------------------------------------
#   三种取值：
#     OFF   默认。完全不编，不产生任何 FFmpeg 依赖；--decode ffmpeg 会报一条
#           明确的错误（不是静默回退）。想用就显式打开：
#               cmake -S . -B build -DTRT_ALPHA_WITH_FFMPEG=ON
#     ON    必须找到，找不到就 FATAL —— 用于 CI / 发布，避免"以为编进去了其实没有"。
#     AUTO  找到就用（ON），找不到就静默降级为 OFF。适合"本机有就用"的脚本。
#
#   为什么默认 OFF 而不是 AUTO：链接的是 shared 构建，FFmpeg 一旦编进去就成了
#   **硬运行时依赖**（avcodec-*.dll 缺失时 exe 直接起不来，哪怕你只用 opencv）。
#   "可选"就该是显式 opt-in，而不是配好了路径就悄悄产生依赖。
#
#   需要的是 **shared 构建**（include/ + lib/*.lib + bin/*.dll），不是 CLI 的静态
#   exe。下载：BtbN FFmpeg-Builds 的 *-win64-gpl-shared.zip
#   只链 4 个库：avformat / avcodec / avutil（+ swscale 给软件解码路径做
#   YUV->BGR）。avfilter / avdevice 不用。
#
#   Linux 特有的坑：BtbN 的 shared 包**自身不带 RPATH**，而 libavcodec.so 是
#   DT_NEEDED libswresample.so / libavutil.so。glibc 的 DT_RUNPATH **不向传递
#   依赖传播**（只管直接依赖），所以只在 RUNPATH 里写 ${FFMPEG_ROOT}/lib 的话，
#   程序能加载 libavcodec、却在找 libswresample 时直接死：
#       "libswresample.so.7: cannot open shared object file"
#   修法见下面对 --disable-new-dtags 的注释。Windows 无此问题（DLL 在 exe 旁）。
# =============================================================================
set(TRT_ALPHA_WITH_FFMPEG "OFF" CACHE STRING
    "FFmpeg video decode backend: OFF (default) | ON (require) | AUTO (use if found)")
set_property(CACHE TRT_ALPHA_WITH_FFMPEG PROPERTY STRINGS OFF ON AUTO)

set(FFMPEG_ROOT "D:/ThirdParty/ffmpeg-shared" CACHE PATH
    "Root of a shared FFmpeg build (include/ lib/ bin/)")

string(TOUPPER "${TRT_ALPHA_WITH_FFMPEG}" TRT_ALPHA_FF_MODE)

set(TRT_ALPHA_FFMPEG_ENABLED FALSE)

# The import library / shared object is named differently per platform, so probe
# for the one this toolchain actually links against. Hard-coding avcodec.lib
# makes the whole block dead code on Linux (AUTO then silently reports OFF).
if(WIN32)
    set(TRT_ALPHA_FF_LIBGLOB "${FFMPEG_ROOT}/lib/avcodec.lib")
else()
    file(GLOB TRT_ALPHA_FF_LIBGLOB "${FFMPEG_ROOT}/lib/libavcodec.so")
    if(NOT TRT_ALPHA_FF_LIBGLOB)
        file(GLOB TRT_ALPHA_FF_LIBGLOB "${FFMPEG_ROOT}/lib/libavcodec.so.*")
    endif()
endif()

if(NOT TRT_ALPHA_FF_MODE STREQUAL "OFF")
    if(EXISTS "${FFMPEG_ROOT}/include/libavcodec/avcodec.h"
       AND EXISTS "${FFMPEG_ROOT}/include/libavformat/avformat.h"
       AND TRT_ALPHA_FF_LIBGLOB)
        set(TRT_ALPHA_FFMPEG_ENABLED TRUE)
    endif()
endif()

if(TRT_ALPHA_FFMPEG_ENABLED)
    foreach(_ff_lib avformat avcodec avutil swscale)
        if(WIN32)
            # The DLL carries the version in its name (avcodec-63.dll) and is what
            # has to sit next to the exe at run time.
            file(GLOB _ff_dll "${FFMPEG_ROOT}/bin/${_ff_lib}-*.dll")
            if(NOT _ff_dll)
                message(FATAL_ERROR
                    "FFmpeg: ${_ff_lib} DLL not found under ${FFMPEG_ROOT}/bin")
            endif()
            list(GET _ff_dll 0 _ff_binary)
            set(_ff_implib "${FFMPEG_ROOT}/lib/${_ff_lib}.lib")
        else()
            file(GLOB _ff_so "${FFMPEG_ROOT}/lib/lib${_ff_lib}.so")
            if(NOT _ff_so)
                file(GLOB _ff_so "${FFMPEG_ROOT}/lib/lib${_ff_lib}.so.*")
            endif()
            if(NOT _ff_so)
                message(FATAL_ERROR
                    "FFmpeg: lib${_ff_lib}.so not found under ${FFMPEG_ROOT}/lib")
            endif()
            list(GET _ff_so 0 _ff_binary)
            set(_ff_implib "")
        endif()

        add_library(ffmpeg::${_ff_lib} SHARED IMPORTED)
        set_target_properties(ffmpeg::${_ff_lib} PROPERTIES
            IMPORTED_LOCATION              "${_ff_binary}"
            INTERFACE_INCLUDE_DIRECTORIES  "${FFMPEG_ROOT}/include")
        if(_ff_implib)
            set_target_properties(ffmpeg::${_ff_lib} PROPERTIES
                IMPORTED_IMPLIB "${_ff_implib}")
        endif()
    endforeach()

    target_link_libraries(trt_alpha_core PUBLIC
        ffmpeg::avformat ffmpeg::avcodec ffmpeg::avutil ffmpeg::swscale)

    # Linux: 让构建树里的二进制自己找到 FFmpeg 的 .so，不要求用户 export
    # LD_LIBRARY_PATH —— 与本仓库"靠绝对 RUNPATH 开箱即用"的策略一致。
    #
    # 关键在 --disable-new-dtags：它让链接器写老式 DT_RPATH 而不是 DT_RUNPATH。
    # 两者只有一个区别，但正好卡在这件事上：
    #   DT_RPATH   会被所有后代对象继承（libavcodec 找 libswresample 时用得上）
    #   DT_RUNPATH 只对"直接依赖"生效（glibc 明确不做传递查找）
    # 不加它的话，RUNPATH 里明明写着 ${FFMPEG_ROOT}/lib，libavcodec 也加载成功，
    # 但它在找自己的 libswresample 时完全看不见这个目录。
    # （不额外 set CMAKE_BUILD_RPATH：CMake 本来就把 imported target 的目录写进
    #  构建树 RPATH，我们只是把 DT_RUNPATH 改成 DT_RPATH，不覆盖已有内容。）
    if(NOT WIN32)
        add_link_options("-Wl,--disable-new-dtags")
    endif()

    target_compile_definitions(trt_alpha_core PUBLIC TRT_ALPHA_WITH_FFMPEG=1)
    message(STATUS "FFmpeg: ON  (${FFMPEG_ROOT})")
elseif(TRT_ALPHA_FF_MODE STREQUAL "ON")
    message(FATAL_ERROR
        "TRT_ALPHA_WITH_FFMPEG=ON but no usable FFmpeg build was found at "
        "'${FFMPEG_ROOT}'. Point -DFFMPEG_ROOT=<dir containing include/ lib/ bin/> "
        "at a shared build (Windows: a BtbN *-win64-gpl-shared package; "
        "Linux: e.g. a jellyfin-ffmpeg / self-built prefix -- the distro's "
        "libavcodec-dev ships no cuvid, so NVDEC will not exist there).")
else()
    target_compile_definitions(trt_alpha_core PUBLIC TRT_ALPHA_WITH_FFMPEG=0)
    if(TRT_ALPHA_FF_MODE STREQUAL "OFF")
        message(STATUS "FFmpeg: OFF (disabled by default; "
                       "-DTRT_ALPHA_WITH_FFMPEG=ON to enable)")
    else()
        message(STATUS "FFmpeg: OFF (no shared build at '${FFMPEG_ROOT}'); "
                       "--decode ffmpeg will report an error at runtime")
    endif()
endif()

# ocv::ocv / gl::glfw 在这种写法下始终存在（OFF 时是空的 INTERFACE target），
# 因此这里不需要再包 if()。两者的 TRT_ALPHA_WITH_* 宏通过 INTERFACE 传播到
# 所有下游 target，保证 #if 判断处处一致。
target_link_libraries(trt_alpha_core
  PUBLIC  ocv::ocv
          gl::glfw
  PRIVATE trt::trt)


# =============================================================================
# 测试
# =============================================================================
enable_testing()          # 顶层开启 CTest，注册的各 test_* 可 `ctest -C Release` 一键跑
#add_subdirectory(test)

# 单个测试的独立入口：整个 test/ 树默认不编（上面那行是注释状态），但解码后端
# 的 A/B 需要随时能跑，所以给一个只挂 test_ffmpeg_source 的开关：
#     cmake -S . -B build -DTRT_ALPHA_BUILD_FFMPEG_TEST=ON
#     cmake --build build --config Release --target test_ffmpeg_source
#     ctest -C Release -R test_ffmpeg_source --output-on-failure
option(TRT_ALPHA_BUILD_FFMPEG_TEST
       "Build test/test_ffmpeg_source (opencv / ffmpeg-sw / ffmpeg-nvdec A/B)" OFF)
if(TRT_ALPHA_BUILD_FFMPEG_TEST AND NOT TARGET test_ffmpeg_source)
    add_subdirectory(test/test_ffmpeg_source)
    add_test(NAME test_ffmpeg_source COMMAND test_ffmpeg_source)
    set_tests_properties(test_ffmpeg_source PROPERTIES SKIP_RETURN_CODE 77)
endif()

add_subdirectory(app)
add_subdirectory(samples)

# demos/ is a local-only playground (ignored by .gitignore). Guard the mount so a
# fresh clone that does not carry the directory still configures cleanly.
if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/demos/CMakeLists.txt")
    add_subdirectory(demos)
endif()
