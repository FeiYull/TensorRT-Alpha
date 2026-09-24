# 每个 demo 单独一个子目录，这里只负责汇总

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
add_subdirectory(test_yolov8)
# 以后加新 demo，在这里加一行即可：
# add_subdirectory(test_xxx)