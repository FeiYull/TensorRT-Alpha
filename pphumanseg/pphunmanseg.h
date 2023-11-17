#pragma once
#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"../utils/kernel_function.h"


class PPHunmanSeg
{
public:
    PPHunmanSeg(const utils::InitParameter& param);
    ~PPHunmanSeg();

public:
    bool init(const std::vector<unsigned char>& trtFile);
    void check();
    void copy(const std::vector<cv::Mat>& imgsBatch);
    void preprocess(const std::vector<cv::Mat>& imgsBatch);
    bool infer();
    void postprocess(const std::vector<cv::Mat>& imgsBatch);
    void reset();
    void showMask(const std::vector<cv::Mat>& imgsBatch, const int& cvDelayTime);
    void saveMask(const std::vector<cv::Mat>& imgsBatch, const std::string& savePath, const int& batchSize, const int& batchi);

protected:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    utils::InitParameter m_param;
    nvinfer1::Dims m_output_src_dims; 
    int m_output_src_area;          

    utils::AffineMat m_dst2src;
    utils::AffineMat m_src2dst; 

    // input
    float* m_input_src_device;
    float* m_input_resize_device;
    float* m_input_rgb_device;
    float* m_input_norm_device;
    float* m_input_hwc_device; 

    // output
    float* m_output_src_device; 
    float* m_output_mask_device;   
    float* m_output_resize_device; 
    float* m_output_resize_host; 
};
