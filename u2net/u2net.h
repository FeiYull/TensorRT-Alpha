#pragma once
#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"../utils/kernel_function.h"
namespace u2net
{
    class U2NET
    {
    public:
        U2NET(const utils::InitParameter& param);
        ~U2NET();

    public:
        bool init(const std::vector<unsigned char>& trtFile);
        void check();
        void copy(const std::vector<cv::Mat>& imgsBatch);
        void preprocess(const std::vector<cv::Mat>& imgsBatch);
        bool infer();
        void postprocess(const std::vector<cv::Mat>& imgsBatch);
        void showMask(const std::vector<cv::Mat>& imgsBatch, const int& cvDelayTime);
        void saveMask(const std::vector<cv::Mat>& imgsBatch, const std::string& savePath, const int& batchSize, const int& batchi);
        void reset();
    private:
        std::shared_ptr<nvinfer1::ICudaEngine> m_engine;
        std::unique_ptr<nvinfer1::IExecutionContext> m_context;

        //private:
    protected:
        utils::InitParameter m_param;
        nvinfer1::Dims m_output_dims;  
        int m_output_area;
        std::vector<std::vector<utils::Box>> m_objectss;

        
        utils::AffineMat m_dst2src;
        utils::AffineMat m_src2dst;

        // input
        float* m_input_src_device;
        float* m_input_resize_device;
        float* m_input_rgb_device;
        float* m_input_norm_device;
        float* m_input_hwc_device;
     
        float* m_max_val_device;
        float* m_min_val_device;

        // output
        float* m_output_src_device;
        float* m_output_resize_device; 
        float* m_output_resize_host; 
        float* m_output_mask_host; 
       
    };
}

void u2netDivMaxDevice(const int& batchSize, float* src, int srcWidth, int srcHeight, int channel, float* maxVals);

void u2netNormPredDevice(const int& batchSize, float* src, int srcWidth, int srcHeight, float scale, float* minVals, float* maxVals);