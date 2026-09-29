add_executable(sample_yolor sample_yolor.cpp)

target_link_libraries(sample_yolor
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)