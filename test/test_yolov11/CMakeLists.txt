add_executable(test_yolov11 test_yolov11.cpp)

# 需要 WHOLE_ARCHIVE 让 yolov11 的静态注册生效（否则 names() 看不到它们）。
# 只用 core（配置 / 注册中心），不需要 OpenCV 和真实引擎。
target_link_libraries(test_yolov11
    PRIVATE
        trt::trt
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
