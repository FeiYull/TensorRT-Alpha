add_executable(sample_yolor sample_yolor.cpp)

target_link_libraries(sample_yolor
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
