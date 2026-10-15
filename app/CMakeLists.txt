add_executable(trt_alpha
    trt_alpha.cpp
    options.cpp
    commands.cpp
)

target_link_libraries(trt_alpha
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)

# 把 OpenCV / TensorRT / CUDA 的运行时 DLL 拷到 build/bin/<Config>/（exe 同级），
# 这样不需要把它们加进 PATH 也能直接跑。其他 exe（samples/ test/）与 trt_alpha
# 共用同一个输出目录，DLL 拷一次即可。详见 cmake/DeployRuntime.cmake。
trt_alpha_deploy_runtime(trt_alpha)
