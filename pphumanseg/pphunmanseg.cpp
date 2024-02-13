#include"pphunmanseg.h"
#include"decode_pphunmanseg.h"

PPHunmanSeg::PPHunmanSeg(const utils::InitParameter& param) : m_param(param)
{
    // input
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    CHECK(cudaMalloc(&m_input_src_device,    param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_rgb_device,    param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_norm_device,   param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_hwc_device,    param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));

    // output
    m_output_src_device = nullptr;
    m_output_mask_device = nullptr;
    m_output_resize_device = nullptr;
    m_output_resize_host = nullptr;
    CHECK(cudaMalloc(&m_output_mask_device,   m_param.batch_size * 1 * m_param.dst_h * m_param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_output_resize_device, m_param.batch_size * 1 * m_param.src_h * m_param.src_w * sizeof(float)));
    m_output_resize_host = new float[m_param.batch_size * 1 * m_param.src_h * m_param.src_w];
}

PPHunmanSeg::~PPHunmanSeg()
{
    // input
    CHECK(cudaFree(m_input_src_device));
    CHECK(cudaFree(m_input_resize_device));
    CHECK(cudaFree(m_input_rgb_device));
    CHECK(cudaFree(m_input_norm_device));
    CHECK(cudaFree(m_input_hwc_device));
    // output
    CHECK(cudaFree(m_output_mask_device));
    CHECK(cudaFree(m_output_resize_device));
    delete[] m_output_resize_host;
}

bool PPHunmanSeg::init(const std::vector<unsigned char>& trtFile)
{
    if (trtFile.empty())
    {
        return false;
    }
    std::unique_ptr<nvinfer1::IRuntime> runtime =
        std::unique_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));
    if (runtime == nullptr)
    {
        return false;
    }
    this->m_engine = std::unique_ptr<nvinfer1::ICudaEngine>(runtime->deserializeCudaEngine(trtFile.data(), trtFile.size()));

    if (this->m_engine == nullptr)
    {
        return false;
    }
    this->m_context = std::unique_ptr<nvinfer1::IExecutionContext>(this->m_engine->createExecutionContext());
    if (this->m_context == nullptr)
    {
        return false;
    }
    if (m_param.dynamic_batch)
    {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }
    m_output_src_dims = this->m_context->getBindingDimensions(1);
    assert(m_param.batch_size <= m_output_src_dims.d[0]);
    
    auto get_area = [](const nvinfer1::Dims& dims) {
        int area = 1;
        for (int i = 1; i < dims.nbDims; i++)
        {
            if (dims.d[i] != 0)
                area *= dims.d[i];
        }
        return area;
    };
    m_output_src_area  = get_area(m_output_src_dims);
    CHECK(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_src_area * sizeof(float)));
    float scale_y = float(m_param.dst_h) / m_param.src_h;
    float scale_x = float(m_param.dst_w) / m_param.src_w;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale_x, 0.f, (-scale_x * m_param.src_w + m_param.dst_w + scale_x - 1) * 0.5,
        0.f, scale_y, (-scale_y * m_param.src_h + m_param.dst_h + scale_y - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);

    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];

    m_src2dst.v0 = src2dst.ptr<float>(0)[0];
    m_src2dst.v1 = src2dst.ptr<float>(0)[1];
    m_src2dst.v2 = src2dst.ptr<float>(0)[2];
    m_src2dst.v3 = src2dst.ptr<float>(1)[0];
    m_src2dst.v4 = src2dst.ptr<float>(1)[1];
    m_src2dst.v5 = src2dst.ptr<float>(1)[2];

    return true;
}

void PPHunmanSeg::check()
{
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

void PPHunmanSeg::copy(const std::vector<cv::Mat>& imgsBatch)
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
        CHECK(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        /*imgsBatch[i].convertTo(imgsBatch[i], CV_32FC3);
        CHECK(cudaMemcpy(pi, imgsBatch[i].data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));*/
        pi += 3 * m_param.src_h * m_param.src_w;
    }

    cudaHostUnregister(img_fp32.data);
}

void PPHunmanSeg::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, utils::ColorMode::RGB, m_dst2src);
    bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h);
    normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w, m_param.dst_h,
        m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h,
        m_input_hwc_device, m_param.dst_w, m_param.dst_h);
}

bool PPHunmanSeg::infer()
{
    float* bindings[] = { m_input_hwc_device, m_output_src_device };
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void PPHunmanSeg::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
    pphunmanseg::decodeDevice(m_param.batch_size, m_output_src_device, m_param.dst_w, m_param.dst_h, 
        m_output_mask_device, m_param.dst_w, m_param.dst_h);
    resizeDevice(m_param.batch_size, m_output_mask_device, m_param.dst_w, m_param.dst_h,
        m_output_resize_device, m_param.src_w, m_param.src_h, utils::ColorMode::GRAY, m_src2dst);
    CHECK(cudaMemcpy(m_output_resize_host, m_output_resize_device, m_param.batch_size * sizeof(float) * m_param.src_w * m_param.src_h, cudaMemcpyDeviceToHost));
}

void PPHunmanSeg::reset()
{
}

void PPHunmanSeg::showMask(const std::vector<cv::Mat>& imgsBatch, const int& cvDelayTime)
{
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        cv::Mat img_mask(m_param.src_h, m_param.src_w, CV_32FC1, m_output_resize_host + bi * m_param.src_w * m_param.src_h);
        img_mask.convertTo(img_mask, CV_8UC1, 255.0, 0.); 
        cv::imshow("img_mask", img_mask);
        cv::waitKey(cvDelayTime);
    }
}

void PPHunmanSeg::saveMask(const std::vector<cv::Mat>& imgsBatch, const std::string& savePath, const int& batchSize, const int& batchi)
{
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        cv::Mat img_mask(m_param.src_h, m_param.src_w, CV_32FC1, m_output_resize_host + bi * m_param.src_w * m_param.src_h);
        img_mask.convertTo(img_mask, CV_8UC1, 255.0, 0.);
        cv::imshow("img_mask", img_mask);
        int imgi = batchi * batchSize + bi;
        cv::imwrite(savePath + "_" + std::to_string(imgi) + ".bmp", img_mask);
        cv::waitKey(1); 
    }
}
