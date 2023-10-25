#pragma once
#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"../utils/kernel_function.h"

class FaceMesh
{
public:
    FaceMesh(const utils::InitParameter& param);
    ~FaceMesh();

public:
    bool init(const std::vector<unsigned char>& trtFile);
    void check();
    void resize(std::vector<cv::Mat>& imgsBatch);
    void copy(const std::vector<cv::Mat>& imgsBatch);
    void preprocess(const std::vector<cv::Mat>& imgsBatch);
    bool infer();
    void postprocess(const std::vector<cv::Mat>& imgsBatch);
    void reset();
    // todo
    void saveToPointsCloud();

public:
    std::vector<std::vector<cv::Point2f>> getLandMarkss() const;

protected:
    std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
    std::unique_ptr<nvinfer1::IExecutionContext> m_context;

protected:
    utils::InitParameter m_param;
    nvinfer1::Dims m_output_conf_dims;  // (1, 1, 0...)
    nvinfer1::Dims m_output_preds_dims; // (1, 1404, 0...)
    
    int m_output_conf_area;
    int m_output_preds_area;
    std::vector<std::vector<utils::Box>> m_objectss;
    std::vector<utils::AffineMat> m_dst2src; // 2*3,  (m_param.dst_h, m_param.dst_w) to (m_param.src_h, m_param.src_w) 
    std::vector<utils::AffineMat> m_src2dst; // 2*3

    // input
    float* m_input_src_device;
    float* m_input_resize_device;
    float* m_input_rgb_device;
    float* m_input_norm_device;
    float* m_input_hwc_device;

    // output
    float* m_output_conf_device;  // malloc in init(), conf :batch_size * 1
    float* m_output_preds_device; // malloc in init(), preds:batch_size * 1404, 1404 = 468 * 3, 468 3d-points
    float* m_ppreds_host;
    float* m_pconf_host;
    std::vector<std::vector<cv::Point2f>> m_land_markss;
};
