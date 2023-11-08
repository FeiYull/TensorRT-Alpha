#pragma once
#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"../utils/kernel_function.h"

class EfficientDet
{
public:
    EfficientDet(const utils::InitParameter& param);
    ~EfficientDet();

public:
    bool init(const std::vector<unsigned char>& trtFile);
    void check();
    void copy(const std::vector<cv::Mat>& imgsBatch);
    void preprocess(const std::vector<cv::Mat>& imgsBatch);
    bool infer();
    void postprocess(const std::vector<cv::Mat>& imgsBatch);
    void reset();

public:
    std::vector<std::vector<utils::Box>> getObjectss() const;

protected:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    utils::InitParameter m_param;
    std::vector<std::vector<utils::Box>> m_objectss;
    utils::AffineMat m_dst2src;
    // input
    float* m_input_src_device;
    float* m_input_resize_device;
    float* m_input_rgb_device;
    // output
    int* m_output_num_device;     
    int* m_output_boxes_device;   
    int* m_output_scores_device;   
    int* m_output_classes_device; 
    int* m_output_num_host;       
    int* m_output_boxes_host;      
    int* m_output_scores_host;    
    int* m_output_classes_host;  
};