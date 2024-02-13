#include"efficientdet.h"
#include"NvInferPlugin.h"

EfficientDet::EfficientDet(const utils::InitParameter& param):m_param(param)
{
    // input
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    CHECK(cudaMalloc(&m_input_src_device,    param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_rgb_device,    param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    // output
    m_output_num_device     = nullptr;  
    m_output_boxes_device   = nullptr;  
    m_output_scores_device  = nullptr;  
    m_output_classes_device = nullptr;
    m_output_num_host     = nullptr;    
    m_output_boxes_host   = nullptr;    
    m_output_scores_host  = nullptr;    
    m_output_classes_host = nullptr;    
    CHECK(cudaMalloc(&m_output_num_device,     param.batch_size * sizeof(int)));
    CHECK(cudaMalloc(&m_output_boxes_device,   param.batch_size * 1 * param.topK * 4 * sizeof(int)));
    CHECK(cudaMalloc(&m_output_scores_device,  param.batch_size * 1 * param.topK * sizeof(int)));
    CHECK(cudaMalloc(&m_output_classes_device, param.batch_size * 1 * param.topK * sizeof(int)));
    m_output_num_host     = new int[param.batch_size];
    m_output_boxes_host   = new int[param.batch_size * 1 * param.topK * 4];
    m_output_scores_host  = new int[param.batch_size * 1 * param.topK];
    m_output_classes_host = new int[param.batch_size * 1 * param.topK];
    m_objectss.resize(param.batch_size);
}

EfficientDet::~EfficientDet()
{
    // input
    CHECK(cudaFree(m_input_src_device));
    CHECK(cudaFree(m_input_resize_device));
    CHECK(cudaFree(m_input_rgb_device));
    // output
    CHECK(cudaFree(m_output_num_device));
    CHECK(cudaFree(m_output_boxes_device));
    CHECK(cudaFree(m_output_scores_device));
    CHECK(cudaFree(m_output_classes_device));
    delete[] m_output_num_host;
    delete[] m_output_boxes_host;
    delete[] m_output_scores_host;
    delete[] m_output_classes_host;
}

bool EfficientDet::init(const std::vector<unsigned char>& trtFile)
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
    initLibNvInferPlugins(&sample::gLogger.getTRTLogger(), ""); // init plugin's lib(Efficient-NMS)
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
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, m_param.dst_h, m_param.dst_w, 3));
    }
    float a = float(m_param.dst_h) / m_param.src_h;
    float b = float(m_param.dst_w) / m_param.src_w;
    float scale = a < b ? a : b;
    cv::Mat src2dst = (cv::Mat_<float>(2, 3) << scale, 0.f, (-scale * m_param.src_w + m_param.dst_w + scale - 1) * 0.5,
        0.f, scale, (-scale * m_param.src_h + m_param.dst_h + scale - 1) * 0.5);
    cv::Mat dst2src = cv::Mat::zeros(2, 3, CV_32FC1);
    cv::invertAffineTransform(src2dst, dst2src);

    m_dst2src.v0 = dst2src.ptr<float>(0)[0];
    m_dst2src.v1 = dst2src.ptr<float>(0)[1];
    m_dst2src.v2 = dst2src.ptr<float>(0)[2];
    m_dst2src.v3 = dst2src.ptr<float>(1)[0];
    m_dst2src.v4 = dst2src.ptr<float>(1)[1];
    m_dst2src.v5 = dst2src.ptr<float>(1)[2];

    return true;
}

void EfficientDet::check()
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

void EfficientDet::copy(const std::vector<cv::Mat>& imgsBatch)
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

void EfficientDet::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, 114, m_dst2src);
    bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h);
}

bool EfficientDet::infer()
{
    void* bindings[] = {m_input_rgb_device,  
                        m_output_num_device, 
                        m_output_boxes_device, 
                        m_output_scores_device,
                        m_output_classes_device };
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void EfficientDet::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
    CHECK(cudaMemcpy(m_output_num_host,     m_output_num_device,     sizeof(int) * m_param.batch_size, cudaMemcpyDeviceToHost));
    CHECK(cudaMemcpy(m_output_boxes_host,   m_output_boxes_device,   sizeof(int) * m_param.batch_size * 1 * m_param.topK * 4, cudaMemcpyDeviceToHost));
    const auto boxes = reinterpret_cast<const float*>(m_output_boxes_host);
    CHECK(cudaMemcpy(m_output_scores_host,  m_output_scores_device,  sizeof(int) * m_param.batch_size * 1 * m_param.topK, cudaMemcpyDeviceToHost));
    const auto scores = reinterpret_cast<const float*>(m_output_scores_host);
    CHECK(cudaMemcpy(m_output_classes_host, m_output_classes_device, sizeof(int) * m_param.batch_size * 1 * m_param.topK, cudaMemcpyDeviceToHost));
    for (int bi = 0; bi < imgsBatch.size(); bi++)
    {
        for (int i = 0; i < m_output_num_host[bi]; i++)
        {
            float y1 = boxes[0 + i * 4 + bi * m_param.topK * 4];
            float x1 = boxes[1 + i * 4 + bi * m_param.topK * 4];
            float y2 = boxes[2 + i * 4 + bi * m_param.topK * 4];
            float x2 = boxes[3 + i * 4 + bi * m_param.topK * 4];

            float y_lt = m_dst2src.v3 * x1 + m_dst2src.v4 * y1 + m_dst2src.v5;
            float x_lt = m_dst2src.v0 * x1 + m_dst2src.v1 * y1 + m_dst2src.v2;
            float y_rb = m_dst2src.v3 * x2 + m_dst2src.v4 * y2 + m_dst2src.v5;
            float x_rb = m_dst2src.v0 * x2 + m_dst2src.v1 * y2 + m_dst2src.v2;

            float score = scores[i + bi * m_param.topK];
            if (score < m_param.conf_thresh)
            {
                continue;
            }
            int32_t class_id = m_output_classes_host[i + bi * m_param.topK];
            assert(class_id >= 0);  
            m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, score, class_id);
        }
    }
}

std::vector<std::vector<utils::Box>> EfficientDet::getObjectss() const
{
    return this->m_objectss;
}

void EfficientDet::reset()
{
    for (size_t bi = 0; bi < m_param.batch_size; bi++)
    {
        m_objectss[bi].clear();
    }
}


