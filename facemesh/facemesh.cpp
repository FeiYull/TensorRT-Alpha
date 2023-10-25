#include"facemesh.h"

FaceMesh::FaceMesh(const utils::InitParameter& param) : m_param(param)
{
    // input
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    checkRuntime(cudaMalloc(&m_input_src_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_rgb_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_norm_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_hwc_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    
    // output
    m_output_conf_device = nullptr;
    m_output_preds_device = nullptr;
    m_ppreds_host = nullptr;
    m_pconf_host = nullptr;
}

 FaceMesh::~FaceMesh()
{
    // input
    checkRuntime(cudaFree(m_input_src_device));
    checkRuntime(cudaFree(m_input_resize_device));
    checkRuntime(cudaFree(m_input_rgb_device));
    checkRuntime(cudaFree(m_input_norm_device));
    checkRuntime(cudaFree(m_input_hwc_device));
    // output
    checkRuntime(cudaFree(m_output_conf_device));
    checkRuntime(cudaFree(m_output_preds_device));
    delete[] m_ppreds_host;
    delete[] m_pconf_host;
}

bool  FaceMesh::init(const std::vector<unsigned char>& trtFile)
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
    m_output_conf_dims = this->m_context->getBindingDimensions(1); // preds:batch_size * 1404, 1404 = 468 * 3
    m_output_preds_dims = this->m_context->getBindingDimensions(2);
    assert(m_param.batch_size <= m_output_preds_dims.d[0]);
    m_output_conf_area = m_output_conf_dims.d[1]; // 1
    m_output_preds_area = m_output_preds_dims.d[1]; // 1404

    // 3. malloc
    checkRuntime(cudaMalloc(&m_output_conf_device,  m_param.batch_size * m_output_conf_area  * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_preds_device, m_param.batch_size * m_output_preds_area * sizeof(float)));
    m_ppreds_host = new float[m_param.batch_size * m_output_preds_area];
    m_pconf_host = new float[m_param.batch_size * m_output_conf_area];


    std::vector<cv::Point2f> land_marks;
    land_marks.reserve(m_output_preds_area / 3);
    for (size_t i = 0; i < m_param.batch_size; i++)
    {
        m_land_markss.emplace_back(land_marks);
    }
    return true;
}

void  FaceMesh::check()
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

void FaceMesh::resize(std::vector<cv::Mat>& imgsBatch)
{
    // cal affine matrix
    float scale_x, scale_y;
    cv::Mat cv_src2dst, cv_dst2src;
    utils::AffineMat src2dst;
    utils::AffineMat dst2src;
    //cv::Mat img_temp = cv::Mat::zeros(cv::Size(m_param.dst_w, m_param.dst_h), CV_8UC3);
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        m_param.src_h = imgsBatch[i].rows;
        m_param.src_w = imgsBatch[i].cols;

        float scale_y = float(m_param.dst_h) / m_param.src_h;
        float scale_x = float(m_param.dst_w) / m_param.src_w;
        cv::Mat cv_src2dst = (cv::Mat_<float>(2, 3) << scale_x, 0.f, (-scale_x * m_param.src_w + m_param.dst_w + scale_x - 1) * 0.5,
            0.f, scale_y, (-scale_y * m_param.src_h + m_param.dst_h + scale_y - 1) * 0.5);
        cv::Mat cv_dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
        cv::invertAffineTransform(cv_src2dst, cv_dst2src);

        dst2src.v0 = cv_dst2src.ptr<float>(0)[0];
        dst2src.v1 = cv_dst2src.ptr<float>(0)[1];
        dst2src.v2 = cv_dst2src.ptr<float>(0)[2];
        dst2src.v3 = cv_dst2src.ptr<float>(1)[0];
        dst2src.v4 = cv_dst2src.ptr<float>(1)[1];
        dst2src.v5 = cv_dst2src.ptr<float>(1)[2];
        m_dst2src.emplace_back(dst2src);

        cv::resize(imgsBatch[i], imgsBatch[i], cv::Size(m_param.dst_w, m_param.dst_h));
        //imgsBatch[i] = img_temp.clone();
    }
}

void FaceMesh::copy(const std::vector<cv::Mat>& imgsBatch)
{
    cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
    cudaHostRegister(img_fp32.data, img_fp32.elemSize() * img_fp32.total(), cudaHostRegisterPortable);

    // copy to device
    float* pi = m_input_src_device;
    //for (size_t i = 0; i < m_param.batch_size; i++)
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        //std::vector<float> img_vec = std::vector<float>(imgsBatch[i].reshape(1, 1));
        imgsBatch[i].convertTo(img_fp32, CV_32FC3);
        checkRuntime(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyHostToDevice));
        /*imgsBatch[i].convertTo(imgsBatch[i], CV_32FC3);
        checkRuntime(cudaMemcpy(pi, imgsBatch[i].data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));*/
        pi += 3 * m_param.dst_h * m_param.dst_w;
    }

    cudaHostUnregister(img_fp32.data);
}

void  FaceMesh::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    // 2.resize
    /* resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, 114, m_dst2src);

    resizeDevice(param.batch_size, input_src_device, param.src_w, param.src_h,
        input_resize_rgb_without_padding_device, param.dst_w, param.dst_h, utils::ColorMode::RGB, dst2src2);*/

#if 0 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_resize_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::namedWindow("ret", cv::WINDOW_NORMAL);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 3. bgr2rgb
    /* bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h); */
    bgr2rgbDevice(m_param.batch_size, m_input_src_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h);

#if 0 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_rgb_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.dst_h, m_param.dst_w, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::namedWindow("ret", cv::WINDOW_NORMAL);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 4. norm:scale mean std
    normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w, m_param.dst_h,
        m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);

#if 0 // valid
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
                        ret.at<cv::Vec3f>(y, x)[c]
                            = m_param.scale * (ret.at<cv::Vec3f>(y, x)[c] * m_param.stds[c] + m_param.means[c]);
                    }

                }
            }
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::imshow("ret", ret);
            cv::waitKey(1);
        }
        delete[] phost;
    }
#endif // 0

    // 5. hwc2chw
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h,
        m_input_hwc_device, m_param.dst_w, m_param.dst_h);
#if 0
    {

        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_hwc_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.dst_h * m_param.dst_w,
                sizeof(float) * 3 * m_param.dst_h * m_param.dst_w, cudaMemcpyDeviceToHost));

            cv::Mat tmp = imgsBatch[j].clone();

            cv::Mat b(m_param.dst_h, m_param.dst_w, CV_32FC1, phost);
            cv::Mat g(m_param.dst_h, m_param.dst_w, CV_32FC1, phost + 1 * m_param.dst_h * m_param.dst_w);
            cv::Mat r(m_param.dst_h, m_param.dst_w, CV_32FC1, phost + 2 * m_param.dst_h * m_param.dst_w);
            std::vector<cv::Mat> bgr{ b, g, r };
            cv::Mat ret;
            cv::merge(bgr, ret);
            ret.convertTo(ret, CV_8UC3, m_param.scale, m_param.scale * m_param.means[0]); // note: m_param.means[0] = m_param.means[1] = m_param.means[2]
            cv::imshow("ret", ret);

            /* SYSTEMTIME st = { 0 };
             GetLocalTime(&st);
             std::string t = std::to_string(st.wHour) + std::to_string(st.wMinute) + std::to_string(st.wMilliseconds);
             std::string save_path = "F:/Data/temp/";;
             cv::imwrite(save_path + t + ".jpg", ret);*/
            cv::waitKey(1);

            cv::Mat img_ = imgsBatch[j].clone();
        }
        delete[] phost;

    }
#endif

}

bool  FaceMesh::infer()
{
    float* bindings[] = { m_input_hwc_device, m_output_conf_device, m_output_preds_device };
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void  FaceMesh::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
    checkRuntime(cudaMemcpy(m_ppreds_host, m_output_preds_device, sizeof(float)* m_param.batch_size * m_output_preds_area, cudaMemcpyDeviceToHost));
    checkRuntime(cudaMemcpy(m_pconf_host, m_output_conf_device,   sizeof(float) * m_param.batch_size * m_output_conf_area, cudaMemcpyDeviceToHost));
    float* ppreds_host = m_ppreds_host;
    float* pconf_host = m_pconf_host;
    for (size_t j = 0; j < imgsBatch.size(); j++)
    {
        /*cv::Mat prediction(m_output_preds_area / 3, 3, CV_32FC1, ppreds_host);
        std::cout << "conf = " << pconf_host[0] << std::endl;
        cv::Mat img_vis = imgsBatch[j].clone();*/

        std::vector<cv::Point2f> points468;
        for (int i = 0; i < 468; i++) // 468 = 1404/3
        {
            float x_ = m_dst2src[j].v0 * ppreds_host[i * 3] + m_dst2src[j].v1 * ppreds_host[i * 3 + 1] + m_dst2src[j].v2;
            float y_ = m_dst2src[j].v3 * ppreds_host[i * 3] + m_dst2src[j].v4 * ppreds_host[i * 3 + 1] + m_dst2src[j].v5;
            points468.emplace_back(cv::Point2f(x_, y_));
        }
        m_land_markss[j] = points468;
        // vis
       /* cv::namedWindow("img_vis", cv::WINDOW_NORMAL);
        cv::imshow("img_vis", img_vis);
        cv::waitKey(0);*/

        ppreds_host += m_output_preds_area;
        pconf_host += m_output_conf_area;
    }
}

void  FaceMesh::reset()
{
    for (size_t bi = 0; bi < m_param.batch_size; bi++)
    {
        m_land_markss[bi].clear();
    }
}

void FaceMesh::saveToPointsCloud()
{
}

std::vector<std::vector<cv::Point2f>> FaceMesh::getLandMarkss() const
{
    return m_land_markss;
}



