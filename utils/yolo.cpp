#include"yolo.h"

yolo::YOLO::YOLO(const utils::InitParameter& param) : m_param(param)
{
    // input
    m_input_src_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    CHECK(cudaMalloc(&m_input_src_device,    param.batch_size * 3 * param.src_h * param.src_w * sizeof(unsigned char)));
    CHECK(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_rgb_device,    param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_norm_device,   param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));
    CHECK(cudaMalloc(&m_input_hwc_device,    param.batch_size * 3 * param.dst_h * param.dst_w * sizeof(float)));

    // output
    m_output_src_device = nullptr;
    m_output_objects_device = nullptr;
    m_output_objects_host = nullptr;
    m_output_objects_width = 7;
    m_output_idx_device = nullptr;
    m_output_conf_device = nullptr;
    int output_objects_size = param.batch_size * (1 + param.topK * m_output_objects_width); // 1: count
    CHECK(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    CHECK(cudaMalloc(&m_output_idx_device, m_param.batch_size * m_param.topK * sizeof(int)));
    CHECK(cudaMalloc(&m_output_conf_device, m_param.batch_size * m_param.topK * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];
    m_objectss.resize(param.batch_size);
}

yolo::YOLO::~YOLO()
{
    // input
    CHECK(cudaFree(m_input_src_device));
    CHECK(cudaFree(m_input_resize_device));
    CHECK(cudaFree(m_input_rgb_device));
    CHECK(cudaFree(m_input_norm_device));
    CHECK(cudaFree(m_input_hwc_device));
    // output
    CHECK(cudaFree(m_output_src_device));
    CHECK(cudaFree(m_output_objects_device));
    CHECK(cudaFree(m_output_idx_device));
    CHECK(cudaFree(m_output_conf_device));
    delete[] m_output_objects_host;
}

bool yolo::YOLO::init(const std::vector<unsigned char>& trtFile)
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
    if (m_param.dynamic_batch) // for some models only support static mutil-batch. eg: yolox
    {
        this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));
    }
    m_output_dims = this->m_context->getBindingDimensions(1);
    m_total_objects = m_output_dims.d[1];
    assert(m_param.batch_size <= m_output_dims.d[0]);
    m_output_area = 1;
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
    CHECK(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));
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

void yolo::YOLO::check()
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
void yolo::YOLO::copy(const std::vector<cv::Mat>& imgsBatch)
{
#if 0 
    cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
    cudaHostRegister(img_fp32.data, img_fp32.elemSize() * img_fp32.total(), cudaHostRegisterPortable);
    float* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        imgsBatch[i].convertTo(img_fp32, CV_32FC3);
        CHECK(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
    cudaHostUnregister(img_fp32.data);
#endif

#if 0 // for Nvidia TX2
    cv::Mat img_fp32 = cv::Mat::zeros(imgsBatch[0].size(), CV_32FC3); // todo 
    float* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        std::vector<float> img_vec = std::vector<float>(imgsBatch[i].reshape(1, 1));
        imgsBatch[i].convertTo(img_fp32, CV_32FC3);
        CHECK(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
#endif

    // update 20230302, faster. 
    // 1. move uint8_to_float in cuda kernel function. For 8*3*1920*1080, cost time 15ms -> 3.9ms
    // 2. Todo
    unsigned char* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        CHECK(cudaMemcpy(pi, imgsBatch[i].data, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        pi += 3 * m_param.src_h * m_param.src_w;
    }

#if 0 // cuda stream
    cudaStream_t streams[32];
    for (int i = 0; i < imgsBatch.size(); i++) 
    {
        CHECK(cudaStreamCreate(&streams[i]));
    }
    unsigned char* pi = m_input_src_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        CHECK(cudaMemcpyAsync(pi, imgsBatch[i].data, sizeof(unsigned char) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice, streams[i]));
        pi += 3 * m_param.src_h * m_param.src_w;
    }
    CHECK(cudaDeviceSynchronize());
#endif
}

void yolo::YOLO::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    resizeDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, 114, m_dst2src);
    bgr2rgbDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
        m_input_rgb_device, m_param.dst_w, m_param.dst_h);
    normDevice(m_param.batch_size, m_input_rgb_device, m_param.dst_w, m_param.dst_h,
        m_input_norm_device, m_param.dst_w, m_param.dst_h, m_param);
    hwc2chwDevice(m_param.batch_size, m_input_norm_device, m_param.dst_w, m_param.dst_h,
        m_input_hwc_device, m_param.dst_w, m_param.dst_h);
}

bool yolo::YOLO::infer()
{
    float* bindings[] = { m_input_hwc_device, m_output_src_device };
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void yolo::YOLO::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
    decodeDevice(m_param, m_output_src_device, 5 + m_param.num_class, m_total_objects, m_output_area,
        m_output_objects_device, m_output_objects_width, m_param.topK);

    // nmsv1(nms faster)
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1);

    // nmsv2(nms sort)
    //nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1, m_output_idx_device, m_output_conf_device);

    CHECK(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * (1 + 7 * m_param.topK), cudaMemcpyDeviceToHost));
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        for (size_t i = 0; i < num_boxes; i++)
        {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag)
            {
                float x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2; 
                float y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2; 
                float y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;
                m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4], (int)ptr[5]);
            }
        }
   
    }

}

std::vector<std::vector<utils::Box>> yolo::YOLO::getObjectss() const
{
    return this->m_objectss;
}

void yolo::YOLO::reset()
{
    CHECK(cudaMemset(m_output_objects_device, 0, sizeof(float) * m_param.batch_size * (1 + 7 * m_param.topK)));
    for (size_t bi = 0; bi < m_param.batch_size; bi++)
    {
        m_objectss[bi].clear();
    }
}



