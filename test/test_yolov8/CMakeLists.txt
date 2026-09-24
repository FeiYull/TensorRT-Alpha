add_executable(test_yolov8 test_yolov8.cpp)

# test_yolov8 用 OpenCV 读图 -> 需要 ocv::ocv
# 需要 WHOLE_ARCHIVE 让 yolov8 的静态注册生效
target_link_libraries(test_yolov8
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)