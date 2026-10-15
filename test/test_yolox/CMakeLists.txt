add_executable(test_yolox test_yolox.cpp)

target_link_libraries(test_yolox
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
