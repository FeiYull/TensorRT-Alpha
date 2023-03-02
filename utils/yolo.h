#pragma once
#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"../utils/kernel_function.h"

namespace yolo
{
    class YOLO
    {
    public:
        YOLO(const utils::InitParameter& param);
        ~YOLO();

    public:
        virtual bool init(const std::vector<unsigned char>& trtFile);
        virtual void check();
        virtual void copy(const std::vector<cv::Mat>& imgsBatch);
        virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
        virtual bool infer();
        virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
        virtual void reset();

    public:
        std::vector<std::vector<utils::Box>> getObjectss() const;

    protected:
        std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context;

    protected:
        utils::InitParameter m_param;
        nvinfer1::Dims m_output_dims;   // (1, 25200, 85, 0, 0...)
        int m_output_area;              // 1 * 25200 * 85
        int m_total_objects;            // 25200
        std::vector<std::vector<utils::Box>> m_objectss;

        // (m_param.dst_h, m_param.dst_w) to (m_param.src_h, m_param.src_w) 
        utils::AffineMat m_dst2src;     // 2*3

        // input
        //float* m_input_src_device;
        unsigned char* m_input_src_device;
        float* m_input_resize_device;
        float* m_input_rgb_device;
        float* m_input_norm_device;
        float* m_input_hwc_device;
        //float* m_input_dst_device;

        // output
        float* m_output_src_device;     // malloc in init()
        float* m_output_objects_device;
        float* m_output_objects_host;
        int m_output_objects_width;     // 7:left, top, right, bottom, confidence, class, keepflag; 
        int* m_output_idx_device;       // nmsv2
        float* m_output_conf_device;
    };
}
