add_executable(test_yolov6 test_yolov6.cpp)

target_link_libraries(test_yolov6
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)