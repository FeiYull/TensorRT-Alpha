add_executable(sample_yolonas sample_yolonas.cpp)

target_link_libraries(sample_yolonas
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)