# =============================================================================
#  cmake/DeployRuntime.cmake
# -----------------------------------------------------------------------------
#  让程序在【没有为 OpenCV / TensorRT 配置 PATH / LD_LIBRARY_PATH】的机器上
#  也能直接运行。两个平台的手段不同，因为加载器规则不同：
#
#      Windows：exe 目录 -> 系统目录 -> 当前目录 -> PATH      （exe 目录优先级最高）
#      Linux  ：RPATH/RUNPATH -> LD_LIBRARY_PATH -> ldconfig 缓存 -> /lib, /usr/lib
#
#    => Windows 上"把 DLL 拷到 exe 旁"就够（命中最高优先级，能压住 PATH 里的旧版本）；
#    => Linux  上拷到 exe 旁【默认无效】，必须用 RPATH 告诉加载器（见下方 RPATH 段）。
#
#  为什么需要：
#    链接期只需要 .lib / .so —— -DOpenCV_DIR / -DTensorRT_ROOT 只作用于链接器。
#    Windows 加载器在运行期【不认识任何 CMake 变量】，它只按固定顺序找 DLL。
#
#  为什么不能只靠 $<TARGET_RUNTIME_DLLS:...>（2026-10-14 实测结论）：
#    该表达式只收集【SHARED / MODULE 型 CMake target】的 IMPORTED_LOCATION：
#      - OpenCV   : 官方 Config 建的是 SHARED IMPORTED target -> 能自动拿到
#      - TensorRT : 本项目用 find_library 拿到的是【裸路径字符串】，不是 target
#                   -> 被完全忽略
#      - CUDA     : CUDA::cudart 是 UNKNOWN IMPORTED（FindCUDAToolkit.cmake 里
#                   写死 CUDA_IMPORT_TYPE UNKNOWN），且 IMPORTED_LOCATION 指向
#                   .lib 而非 .dll -> 同样被忽略
#    实测 dump 结果 = “只有 opencv_world*.dll（重复 16 次）”。
#    因此 CUDA / TensorRT 的 DLL 只能在本文件里手工列。
#
#  另注：TensorRT 10 的 CUDA 依赖是【运行期动态加载】的（不写进 PE 导入表），
#        dumpbin 看不到；本文件里的 CUDA 清单是按“实跑 + 抓进程模块”确定的。
#
#  用法：trt_alpha_deploy_runtime(<exe_target>)
#        Windows 上给 <target> 挂 POST_BUILD 拷贝；Linux/macOS 上是空操作
#        （RPATH 已在下方全局设好，无需拷贝）。
# =============================================================================

# -----------------------------------------------------------------------------
#  Linux / macOS：RPATH（不拷贝）
# -----------------------------------------------------------------------------
#  ① 构建树：CMake 默认已把链接库所在目录（即 -DOpenCV_DIR / -DTensorRT_ROOT
#     指向的路径）写成 BUILD_RPATH，所以 `cmake --build build` 之后直接跑即可，
#     本来就不必设 LD_LIBRARY_PATH。
#     BUILD_RPATH_USE_ORIGIN 让这些条目尽量改写成相对 $ORIGIN 的形式，
#     于是"把 .so 手工拷到 exe 旁"这种部署方式也能生效。
#  ② 安装/分发：INSTALL_RPATH = $ORIGIN（可执行文件自身所在目录），
#     库和 exe 放同一目录即可零配置运行；INSTALL_RPATH_USE_LINK_PATH 保证
#     依赖库不在 exe 旁时还能退回它原本的安装目录。
#     （本项目当前没有 install() 规则，这一条是为将来打包预留。）
#
#  注：拷 .so 之所以不做，是因为 Linux 的库名是
#      libfoo.so -> libfoo.so.4.8 -> libfoo.so.4.8.0 的符号链接链，
#      copy 会解引用成三份实体文件，反而更占空间；RPATH 才是平台惯例。
# -----------------------------------------------------------------------------
if(UNIX AND NOT APPLE)
    set(CMAKE_BUILD_RPATH_USE_ORIGIN    TRUE)    # CMake 3.20+：构建树 RPATH 用 $ORIGIN
    set(CMAKE_INSTALL_RPATH             "$ORIGIN")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH  FALSE)
elseif(APPLE)
    # macOS 的等价写法（本项目未在 macOS 上验证过）。
    set(CMAKE_INSTALL_RPATH             "@loader_path")
    set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
    set(CMAKE_BUILD_WITH_INSTALL_RPATH  FALSE)
endif()


option(TRT_ALPHA_DEPLOY_BUILDER
       "[Windows only] Deploy TensorRT nvinfer_builder_resource_*.dll (only needed to build engines from ONNX, ~2.8 GB)"
       OFF)

option(TRT_ALPHA_DEPLOY_CUDA_MATH
       "[Windows only] Deploy cuBLAS / cuBLASLt / cuDNN DLLs as well (not loaded by the current engines, ~730 MB)"
       OFF)


function(trt_alpha_deploy_runtime target)
    if(NOT WIN32)
        # Linux / macOS：加载器按 RPATH 找库，上面已经把 RPATH 设好了；
        # 若把库放到 exe 旁，RPATH 里的 $ORIGIN 也会让它生效。
        # 所以这里不做拷贝（理由见文件头的符号链接链说明）。
        return()
    endif()

    set(_dlls "")

    # -------------------------------------------------------------------------
    # 1) OpenCV
    #    $<TARGET_RUNTIME_DLLS> 自带配置感知：Debug -> opencv_world480d.dll。
    #    16 个 OpenCV 模块组件都指向同一个 world DLL，必须去重，
    #    否则同一条 copy 命令会重复执行 16 次。
    # -------------------------------------------------------------------------
    list(APPEND _dlls "$<REMOVE_DUPLICATES:$<TARGET_RUNTIME_DLLS:${target}>>")

    # FFmpeg 视频后端 DLL 不是 CMake target（散落在 OpenCV bin/），手工补。
    if(DEFINED OpenCV_DIR)
        string(REPLACE "." "" _ocv_tag "${OpenCV_VERSION}")   # 4.8.0 -> 480
        file(GLOB _ocv_ffmpeg
             "${OpenCV_DIR}/../bin/opencv_videoio_ffmpeg${_ocv_tag}_64.dll")
        list(APPEND _dlls ${_ocv_ffmpeg})
    endif()

    # -------------------------------------------------------------------------
    # 1b) FFmpeg（可选解码后端）
    #     imported target 不参与 $<TARGET_RUNTIME_DLLS>，直接 glob shared 构建
    #     的 bin/：avcodec-63 / avformat-63 / avutil-61 / swscale-10。
    #     没有它 exe 会报 "avcodec-63.dll not found"，且只影响 --decode ffmpeg。
    # -------------------------------------------------------------------------
    if(TRT_ALPHA_FFMPEG_ENABLED AND DEFINED FFMPEG_ROOT AND EXISTS "${FFMPEG_ROOT}/bin")
        file(GLOB _ffmpeg_dlls "${FFMPEG_ROOT}/bin/*.dll")
        list(APPEND _dlls ${_ffmpeg_dlls})
    endif()

    # -------------------------------------------------------------------------
    # 2) TensorRT
    #    bin/ 是官方发行包的固定布局，直接 glob。
    #    默认排除 nvinfer_builder_resource_*：只有从 ONNX 现场构建 engine 时才
    #    加载，且按 SM 架构各一份（sm75/80/86/89/90/100/120 + ptx），合计约 2.8 GB；
    #    跑现成 .trt 完全用不到。
    # -------------------------------------------------------------------------
    if(DEFINED TensorRT_ROOT AND EXISTS "${TensorRT_ROOT}/bin")
        file(GLOB _trt_dlls "${TensorRT_ROOT}/bin/*.dll")
        if(NOT TRT_ALPHA_DEPLOY_BUILDER)
            list(FILTER _trt_dlls EXCLUDE REGEX "nvinfer_builder_resource_")
        endif()
        list(APPEND _dlls ${_trt_dlls})
    else()
        message(WARNING "[Deploy] TensorRT_ROOT/bin not found; TensorRT DLLs will not be deployed")
    endif()

    # -------------------------------------------------------------------------
    # 3) CUDA
    #    实测（2026-10-14，yolov26 全任务族实跑并抓取进程模块）：真正被加载的
    #    只有 cudart64_12.dll；cuBLAS / cuDNN 都没有出现在模块列表里。
    #    所以默认只拷 cudart，省约 730 MB。
    #    若某个模型/引擎报 “cublas64_*.dll not found”，改用
    #    -DTRT_ALPHA_DEPLOY_CUDA_MATH=ON 重新 configure。
    # -------------------------------------------------------------------------
    if(DEFINED CUDAToolkit_BIN_DIR AND EXISTS "${CUDAToolkit_BIN_DIR}")
        file(GLOB _cuda_dlls "${CUDAToolkit_BIN_DIR}/cudart64_*.dll")
        if(TRT_ALPHA_DEPLOY_CUDA_MATH)
            file(GLOB _cuda_math_dlls
                 "${CUDAToolkit_BIN_DIR}/cublas64_*.dll"
                 "${CUDAToolkit_BIN_DIR}/cublasLt64_*.dll"
                 "${CUDAToolkit_BIN_DIR}/cudnn64_*.dll"
                 "${CUDAToolkit_BIN_DIR}/cudnn*_infer64_*.dll")
            list(APPEND _cuda_dlls ${_cuda_math_dlls})
        endif()
        list(APPEND _dlls ${_cuda_dlls})
    else()
        message(WARNING "[Deploy] CUDA bin dir unknown; CUDA DLLs will not be deployed")
    endif()

    if(NOT _dlls)
        message(WARNING "[Deploy] nothing to deploy for ${target}")
        return()
    endif()

    # -------------------------------------------------------------------------
    # copy_if_different：内容没变就零操作 -> 增量构建无额外开销。
    # 挂在 POST_BUILD（链接之后），不参与编译/链接，不影响编译速度。
    # 代价只有磁盘：每个 build/bin/<Config>/ 各存一份副本。
    # -------------------------------------------------------------------------
    add_custom_command(TARGET ${target} POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
                ${_dlls}
                "$<TARGET_FILE_DIR:${target}>"
        COMMAND_EXPAND_LISTS
        COMMENT "[Deploy] runtime DLLs -> $<TARGET_FILE_DIR:${target}>"
        VERBATIM
    )
endfunction()
