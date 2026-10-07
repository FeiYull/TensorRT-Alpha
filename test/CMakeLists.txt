add_subdirectory(test_buffer_view)
add_subdirectory(test_buffer)
add_subdirectory(test_logger)
add_subdirectory(test_batch)
add_subdirectory(test_paths)
add_subdirectory(test_ini_parser)
add_subdirectory(test_config)
add_subdirectory(test_engine)
add_subdirectory(test_task_types)
add_subdirectory(test_batch_result)
add_subdirectory(test_model_registry)
add_subdirectory(test_memory_pool)
add_subdirectory(test_kernels)
add_subdirectory(test_renderer)
add_subdirectory(test_datasource)
add_subdirectory(test_bounded_queue)
add_subdirectory(test_pipeline)
add_subdirectory(test_yolov3)
add_subdirectory(test_yolov4)
add_subdirectory(test_yolov5)
add_subdirectory(test_yolov6)
add_subdirectory(test_yolov7)
add_subdirectory(test_yolov8)
add_subdirectory(test_yolox)
add_subdirectory(test_yolor)
add_subdirectory(test_yolonas)
add_subdirectory(test_efficientdet)
add_subdirectory(test_yunet)
add_subdirectory(test_yolov8_seg)
add_subdirectory(test_u2net)
add_subdirectory(test_yolov8_pose)
add_subdirectory(test_infer)

# -----------------------------------------------------------------------------
# 注册到 CTest：在 build 目录执行
#     ctest -C Release --output-on-failure
# 即可一键跑全部测试。需要真实引擎/图片的测试在资源缺失时会返回 77，
# CTest 记为 SKIP（而非 FAIL），见各测试的 SKIP_RETURN_CODE。
# -----------------------------------------------------------------------------
set(TRT_ALPHA_TEST_TARGETS
    test_buffer_view test_buffer test_logger test_batch test_paths
    test_ini_parser test_config test_engine test_task_types
    test_batch_result test_model_registry test_memory_pool test_kernels
    test_renderer test_datasource test_bounded_queue test_pipeline
    test_yolov3 test_yolov4 test_yolov5 test_yolov6 test_yolov7
    test_yolov8 test_yolox test_yolor test_yolonas test_efficientdet
    test_yunet test_yolov8_seg test_u2net test_yolov8_pose test_infer
)
foreach(_t IN LISTS TRT_ALPHA_TEST_TARGETS)
    if(TARGET ${_t})
        add_test(NAME ${_t} COMMAND ${_t})
        set_tests_properties(${_t} PROPERTIES SKIP_RETURN_CODE 77)
    endif()
endforeach()