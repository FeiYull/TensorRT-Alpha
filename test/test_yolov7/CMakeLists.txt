add_executable(test_yolov7 test_yolov7.cpp)

# test_yolov7 用 OpenCV 读图 -> 需要 ocv::ocv
# 需要 WHOLE_ARCHIVE 让 yolov7 的静态注册生效
target_link_libraries(test_yolov7
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)