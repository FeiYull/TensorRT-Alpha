add_executable(test_yolor test_yolor.cpp)

# test_yolor 用 OpenCV 读图 -> 需要 ocv::ocv
# 需要 WHOLE_ARCHIVE 让 yolor 的静态注册生效
target_link_libraries(test_yolor
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)