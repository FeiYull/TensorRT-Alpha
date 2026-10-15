add_executable(sample_efficientdet sample_efficientdet.cpp)

target_link_libraries(sample_efficientdet
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
