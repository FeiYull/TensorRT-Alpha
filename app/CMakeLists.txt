add_executable(trt_alpha
    trt_alpha.cpp
    options.cpp
    commands.cpp
)

target_link_libraries(trt_alpha
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)