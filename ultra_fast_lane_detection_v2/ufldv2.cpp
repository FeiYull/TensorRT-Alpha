#include"ufldv2.h"
#include"decode_ufldv2.h"

UFLDV2::UFLDV2(const UFLD_Params& param) : m_param(param)
{
    // input
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_crop_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    
    checkRuntime(cudaMalloc(&m_input_src_device, param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.resize_height * param.resize_width * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_crop_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_norm_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_hwc_device,  param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    
    // output
    m_output_out0_device = nullptr;
    m_output_out1_device = nullptr;
    m_output_out2_device = nullptr;
    m_output_out3_device = nullptr;
    checkRuntime(cudaMalloc(&m_output_out0_device, param.batch_size * 200 * 72 * 4 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_out1_device, param.batch_size * 100 * 81 * 4 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_out2_device, param.batch_size * 2 * 72 * 4 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_out3_device, param.batch_size * 2 * 81 * 4 * sizeof(float)));

}

UFLDV2::~UFLDV2()
{
    // input
    checkRuntime(cudaFree(m_input_src_device));
    checkRuntime(cudaFree(m_input_resize_device));
    checkRuntime(cudaFree(m_input_crop_device));
    checkRuntime(cudaFree(m_input_norm_device));
    checkRuntime(cudaFree(m_input_hwc_device));
    // output
    checkRuntime(cudaFree(m_output_out0_device));
    checkRuntime(cudaFree(m_output_out1_device));
    checkRuntime(cudaFree(m_output_out2_device));
    checkRuntime(cudaFree(m_output_out3_device));
}

bool UFLDV2::init(const std::vector<unsigned char>& trtFile)
{
    // 1. init engine & context
    if (trtFile.empty())
    {
        return false;
    }
    // runtime
    std::unique_ptr<nvinfer1::IRuntime> runtime =
        std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (runtime == nullptr)
    {
        return false;
    }
    // deserializeCudaEngine
    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

    if (this->m_engine == nullptr)
    {
        return false;
    }
    // context
    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return false;
    }
    // binding dim
    if (m_param.dynamic_batch) // for some models only support static dynamic batch. eg: yolox
    {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }

    // 2. get output's dim
   /* m_output_dims = this->m_context->getBindingDimensions(1);
    m_total_objects = m_output_dims.d[1];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1;
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }*/
    // 3. malloc
    //checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
    
    // 4. cal affine matrix 下面还有个crop!!!!!!!!
    float scale_y = float(m_param.resize_height) / m_param.src_h;
    float scale_x = float(m_param.resize_width) / m_param.src_w;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale_x, 0.f, (-scale_x * m_param.src_w + m_param.resize_width + scale_x - 1) * 0.5,
        0.f, scale_y, (-scale_y * m_param.src_h + m_param.resize_height + scale_y - 1) * 0.5);
    cv::Mat cv_dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, cv_dst2src);

    m_dst2src.v0 = cv_dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = cv_dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = cv_dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = cv_dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = cv_dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = cv_dst2src.ptr<float>(1)[2];

    return true;
}

void UFLDV2::check()
{
    // print inputs and outputs' dims
    int idx;
    nvinfer1::Dims dims;

    sample::gLogInfo << "the engine's info:" << std::endl;
    for (auto layer_name : m_param.input_output_names)
    {
        idx = this->m_engine->getBindingIndex(layer_name.c_str());
        dims = this->m_engine->getBindingDimensions(idx);
        sample::gLogInfo << "idx = " << idx << ", " << layer_name << ": ";
        for (int i = 0; i < dims.nbDims; i++)
        {
            sample::gLogInfo << dims.d[i] << ", ";
        }
        sample::gLogInfo << std::endl;
    }

    sample::gLogInfo << "the context's info:" << std::endl;
    for (auto layer_name : m_param.input_output_names)
    {
        idx = this->m_engine->getBindingIndex(layer_name.c_str());
        dims = this->m_context->getBindingDimensions(idx);
        sample::gLogInfo << "idx = " << idx << ", " << layer_name << ": ";
        for (int i = 0; i < dims.nbDims; i++)
        {
            sample::gLogInfo << dims.d[i] << ", ";
        }
        sample::gLogInfo << std::endl;
    }
}

void UFLDV2::copy(const std::vector<cv::Mat>& imgsBatch)
{
    cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
    cudaHostRegister(img_fp32.data, img_fp32.elemSize() * img_fp32.total(), cudaHostRegisterPortable);
    // copy to device
    float* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        imgsBatch[i].convertTo(img_fp32, CV_32FC3);
        checkRuntime(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
    cudaHostUnregister(img_fp32.data);
}

void UFLDV2::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    // 1.resize
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.resize_width, m_param.resize_height, utils::ColorMode::RGB, m_dst2src);

#if 1 // valid
    {
        float* phost = new float[3 * m_param.resize_height * m_param.resize_width];
        float* pdevice = m_input_resize_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.resize_height * m_param.resize_width,
                sizeof(float) * 3 * m_param.resize_height * m_param.resize_width, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.resize_height, m_param.resize_width, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::Mat img_src = imgsBatch[j].clone();
            cv::resize(img_src, img_src, cv::Size(m_param.src_w, m_param.src_h), 0., 0., cv::INTER_LINEAR);
            //cv::namedWindow("ret", cv::WINDOW_NORMAL);
           /* cv::imshow("ret", ret);
            cv::waitKey(1);*/
        }
        delete[] phost;
    }
#endif // 0

    // 2. crop
    int y_low = m_param.resize_height - m_param.dst_h;
    ufld::cropDevice(m_param.batch_size, 0, m_param.dst_w, y_low, m_param.resize_height,
        m_input_resize_device, m_param.resize_width, m_param.resize_height,
        m_input_crop_device, m_param.dst_w, m_param.dst_h);
#if 1 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_crop_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 3. norm:scale mean std
    normDevice(m_param.batch_size, m_input_crop_device, m_param.dst_w, m_param.dst_h,
        m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);

#if 1 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_norm_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            for (size_t y = 0; y < ret.rows; y++)
            {
                for (size_t x = 0; x < ret.cols; x++)
                {
                    for (size_t c = 0; c < 3; c++)
                    {
                        //
                        ret.at<cv::Vec3f>(y, x)[c]
                            = m_param.scale * (ret.at<cv::Vec3f>(y, x)[c] * m_param.stds[c] + m_param.means[c]);
                    }

                }
            }
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
          /*  cv::imshow("ret", ret);
            cv::waitKey(1);*/
        }
        delete[] phost;
    }
#endif // 0

    // 4. hwc2chw
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h,
        m_input_hwc_device, m_param.dst_w, m_param.dst_h);
#if 1
    {

        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_hwc_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));

            cv::Mat b(m_param.dst_h, m_param.dst_w, CV_32FC1, phost);
            cv::Mat g(m_param.dst_h, m_param.dst_w, CV_32FC1, phost + 1 * m_param.dst_h * m_param.dst_w);
            cv::Mat r(m_param.dst_h, m_param.dst_w, CV_32FC1, phost + 2 * m_param.dst_h * m_param.dst_w);
            std::vector<cv::Mat> bgr{ b, g, r };
            cv::Mat ret;
            cv::merge(bgr, ret);
            ret.convertTo(ret, CV_8UC3, 255, 0.0);
           /* cv::imshow("ret", ret);
            cv::waitKey(1);*/
        }
        delete[] phost;

    }
#endif
}

bool UFLDV2::infer()
{
    float* bindings[] = { m_input_hwc_device, 
        m_output_out0_device,
        m_output_out1_device,
        m_output_out2_device,
        m_output_out3_device
    };
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void UFLDV2::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
#if 1 valid
    {

        /*
        
    checkRuntime(cudaMalloc(&m_output_out0_device, param.batch_size * 200 * 72 * 4 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_out1_device, param.batch_size * 100 * 81 * 4 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_out2_device, param.batch_size * 2 * 72 * 4 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_out3_device, param.batch_size * 2 * 81 * 4 * sizeof(float)));
        
        
        */

        float* phost0 = new float[m_param.batch_size * 200 * 72 * 4];
        float* phost1 = new float[m_param.batch_size * 100 * 81 * 4];
        float* phost2 = new float[m_param.batch_size * 2 * 72 * 4];
        float* phost3 = new float[m_param.batch_size * 2 * 81 * 4];
        float* pdevice0 = m_output_out0_device;
        float* pdevice1 = m_output_out1_device;
        float* pdevice2 = m_output_out2_device;
        float* pdevice3 = m_output_out3_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost0, pdevice0 + j * 200 * 72 * 4, sizeof(float) * m_param.batch_size * 200 * 72 * 4, cudaMemcpyDeviceToHost));
            checkRuntime(cudaMemcpy(phost1, pdevice1 + j * 100 * 81 * 4, sizeof(float) * m_param.batch_size * 100 * 81 * 4, cudaMemcpyDeviceToHost));
            checkRuntime(cudaMemcpy(phost2, pdevice2 + j * 2 * 72 * 4,   sizeof(float) * m_param.batch_size * 2 * 72 * 4, cudaMemcpyDeviceToHost));
            checkRuntime(cudaMemcpy(phost3, pdevice3 + j * 2 * 81 * 4,   sizeof(float) * m_param.batch_size * 2 * 81 * 4, cudaMemcpyDeviceToHost));
        }
        delete[] phost0;
        delete[] phost1;
        delete[] phost2;
        delete[] phost3;
    }
#endif
}


void UFLDV2::reset()
{
   /* checkRuntime(cudaMemset(m_output_objects_device, 0, sizeof(float) * m_param.batch_size * (1 + 7 * m_param.topK)));
    for (size_t bi = 0; bi < m_param.batch_size; bi++)
    {
        m_objectss[bi].clear();
    }*/
}



