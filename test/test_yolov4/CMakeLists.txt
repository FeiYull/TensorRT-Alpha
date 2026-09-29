add_executable(test_yolov4 test_yolov4.cpp)
target_link_libraries(test_yolov4
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)