add_executable(test_yolov3 test_yolov3.cpp)

# test_yolov3 用 OpenCV 读图 -> 需要 ocv::ocv
# 需要 WHOLE_ARCHIVE 让 yolov6 的静态注册生效
target_link_libraries(test_yolov3
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)