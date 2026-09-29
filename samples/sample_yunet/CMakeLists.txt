add_executable(sample_yunet sample_yunet.cpp)

target_link_libraries(sample_yunet
    PRIVATE
        trt_alpha_core
        trt::trt
        ocv::ocv
        $<LINK_LIBRARY:WHOLE_ARCHIVE,trt_alpha_core>
)