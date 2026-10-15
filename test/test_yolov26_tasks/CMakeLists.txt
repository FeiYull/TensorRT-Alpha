add_executable(test_yolov26_tasks test_yolov26_tasks.cpp)

# 需要 WHOLE_ARCHIVE 让新模型的静态注册生效（否则 names() 看不到它们）。
# ocv::ocv 只用于可选段 [8]/[9]/[10]/[11]：读图造 Batch。
target_link_libraries(test_yolov26_tasks
    PRIVATE
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)
