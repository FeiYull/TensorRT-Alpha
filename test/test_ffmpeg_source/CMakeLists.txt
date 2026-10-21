add_executable(test_ffmpeg_source test_ffmpeg_source.cpp)

target_link_libraries(test_ffmpeg_source
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
)

# ctest registration lives next to add_subdirectory() -- the central
# TRT_ALPHA_TEST_TARGETS loop in test/CMakeLists.txt, or the opt-in block in the
# root CMakeLists.txt (TRT_ALPHA_BUILD_FFMPEG_TEST).
