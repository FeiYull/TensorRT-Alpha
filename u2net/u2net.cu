#include"u2net.h"
#include <thrust/extrema.h> // u2net
#include <thrust/device_ptr.h>

u2net::U2NET::U2NET(const utils::InitParameter& param) : m_param(param)
{
    // input
    m_input_src_device = nullptr;
    m_input_rgb_device = nullptr;
    m_input_resize_device = nullptr;
    m_input_norm_device = nullptr;
    m_input_hwc_device = nullptr;
    m_max_val_device = nullptr;
    m_min_val_device = nullptr;
    checkRuntime(cudaMalloc(&m_input_src_device,    param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_rgb_device,    param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_resize_device, param.batch_size * 3 * param.dst_h * param.dst_h * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_norm_device,   param.batch_size * 3 * param.dst_h * param.dst_h * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_hwc_device,    param.batch_size * 3 * param.dst_h * param.dst_h * sizeof(float)));
    checkRuntime(cudaMalloc(&m_max_val_device,      param.batch_size  * sizeof(float)));
    checkRuntime(cudaMalloc(&m_min_val_device,      param.batch_size  * sizeof(float)));

    // output
    m_output_src_device = nullptr;
    m_output_resize_device = nullptr;
    m_output_resize_host = nullptr;
    m_output_mask_host = nullptr;

    checkRuntime(cudaMalloc(&m_output_resize_device,    param.batch_size * 1 * param.src_h * param.src_w * sizeof(float)));
    m_output_resize_host = new float[param.batch_size * 1 * param.src_h * param.src_w];
    m_output_mask_host = new float[param.src_h * param.src_w];
}

u2net::U2NET::~U2NET()
{
    // input
    checkRuntime(cudaFree(m_input_src_device));
    checkRuntime(cudaFree(m_input_rgb_device));
    checkRuntime(cudaFree(m_input_resize_device));
    checkRuntime(cudaFree(m_input_norm_device));
    checkRuntime(cudaFree(m_input_hwc_device));
    checkRuntime(cudaFree(m_max_val_device));
    checkRuntime(cudaFree(m_min_val_device));

    // output
    checkRuntime(cudaFree(m_output_src_device));
    checkRuntime(cudaFree(m_output_resize_device));
    delete[] m_output_resize_host;
    delete[] m_output_mask_host;
}

bool u2net::U2NET::init(const std::vector<unsigned char>& trtFile)
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
    this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.dst_h, m_param.dst_w));

    // 2. get output's dim
    m_output_dims = this->m_context->getBindingDimensions(1);
    //m_total_objects = m_output_dims.d[1];
    m_output_area = 1; // 320 * 320
    for (int i = 1; i < m_output_dims.nbDims; i++)
    {
        if (m_output_dims.d[i] != 0)
        {
            m_output_area *= m_output_dims.d[i];
        }
    }
    // 3. malloc
    checkRuntime(cudaMalloc(&m_output_src_device, m_param.batch_size * m_output_area * sizeof(float)));

    // 4. cal affine matrix
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

void u2net::U2NET::check()
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

void u2net::U2NET::copy(const std::vector<cv::Mat>& imgsBatch)
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
        checkRuntime(cudaMemcpy(pi, img_fp32.data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));
        /*imgsBatch[i].convertTo(imgsBatch[i], CV_32FC3);
        checkRuntime(cudaMemcpy(pi, imgsBatch[i].data, sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyHostToDevice));*/
        pi += 3 * m_param.src_h * m_param.src_w;
    }

    cudaHostUnregister(img_fp32.data);
}

void u2net::U2NET::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    // 1. bgr2rgb
    bgr2rgbDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_rgb_device, m_param.src_w, m_param.src_h);
#if 0 // valid
    {
        float* phost = new float[3 * m_param.src_h * m_param.src_w];
        float* pdevice = m_input_rgb_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.src_h * m_param.src_w,
                sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyDeviceToHost));
            cv::Mat ret(m_param.src_h, m_param.src_w, CV_32FC3, phost);
            ret.convertTo(ret, CV_8UC3, 1.0, 0.0);
            cv::namedWindow("ret", cv::WINDOW_NORMAL);
            cv::imshow("ret", ret);
            cv::waitKey(0);
        }
        delete[] phost;
    }
#endif // 0

    // 2. resize 
    resizeDevice(m_param.batch_size, m_input_rgb_device, m_param.src_w, m_param.src_h,
        m_input_resize_device, m_param.dst_w, m_param.dst_h, utils::ColorMode::RGB, m_dst2src);
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
            cv::waitKey(0);
        }
        delete[] phost;
    }
#endif // 0
    // 3. norm:scale mean std
    
    // cal max value
    float* p_tmp = m_input_resize_device;
    float* p_max = m_max_val_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        float* max_dev = thrust::max_element(thrust::device, p_tmp, p_tmp + m_param.dst_h * m_param.dst_w);
        p_tmp += m_param.dst_h * m_param.dst_w;

#if 0     
        float max_host[1] = { -FLT_MAX };
        checkRuntime(cudaMemcpy(max_host, max_dev, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << " max_host = " << max_host[0] << std::endl;
#endif // 0

        // copy
        checkRuntime(cudaMemcpy(p_max++, max_dev, sizeof(float), cudaMemcpyDeviceToDevice));

    }
    
    // div by max
    u2netDivMaxDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h, 3, m_max_val_device);
#if 0 // valid
    {
        float* phost = new float[3 * m_param.dst_h * m_param.dst_w];
        float* pdevice = m_input_resize_device;
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
                            = 255. * (ret.at<cv::Vec3f>(y, x)[c]);
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
    normDevice(m_param.batch_size, m_input_resize_device, m_param.dst_w, m_param.dst_h,
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
                            = 255. * (ret.at<cv::Vec3f>(y, x)[c] * m_param.stds[c] + m_param.means[c]);
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
    // 4. hwc2chw
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
            ret.convertTo(ret, CV_8UC3, 255, 0.0);
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

bool u2net::U2NET::infer()
{
    float* bindings[] = { m_input_hwc_device, m_output_src_device };
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void u2net::U2NET::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
#if 0 // valid
    {
        float* phost = new float[m_output_area];
        float* pdevice = m_output_src_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * m_output_area, sizeof(float) * m_output_area, cudaMemcpyDeviceToHost));
            cv::Mat prediction(m_param.dst_h, m_param.dst_w, CV_32FC1, phost);
        }
        delete[] phost;
    }
#endif // 0

    // 1. norm : 

    // min and max value
    float* p_tmp = m_output_src_device;
    float* p_max = m_max_val_device;
    float* p_min = m_min_val_device;
    for (size_t i = 0; i < imgsBatch.size(); i++)
    {
        thrust::pair<float*, float*> min_max_dev = thrust::minmax_element(thrust::device, p_tmp, p_tmp + m_param.dst_h * m_param.dst_w);
        p_tmp += m_param.dst_h * m_param.dst_w;

#if 0     
        float min_host[1] = { FLT_MAX };
        float max_host[1] = { -FLT_MAX };
        checkRuntime(cudaMemcpy(min_host, min_max_dev.first, sizeof(float), cudaMemcpyDeviceToHost));
        checkRuntime(cudaMemcpy(max_host, min_max_dev.second, sizeof(float), cudaMemcpyDeviceToHost));
        std::cout << " min_host = " << min_host[0] << " max_host = " << max_host[0] << std::endl;
#endif // 0

        // copy
        checkRuntime(cudaMemcpy(p_min++, min_max_dev.first, sizeof(float), cudaMemcpyDeviceToDevice));
        checkRuntime(cudaMemcpy(p_max++, min_max_dev.second, sizeof(float), cudaMemcpyDeviceToDevice));

    }
    
    // 2. element = [255 * (element - min)] / (max - min)
    u2netNormPredDevice(m_param.batch_size, m_output_src_device, m_param.dst_w, m_param.dst_h, 255.f, m_min_val_device, m_max_val_device);
#if 0 // valid
    {
        float* phost = new float[m_output_area];
        float* pdevice = m_output_src_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * m_output_area, sizeof(float) * m_output_area, cudaMemcpyDeviceToHost));
            cv::Mat prediction(m_param.dst_h, m_param.dst_w, CV_32FC1, phost);
        }
        delete[] phost;
    }
#endif // 0

    // 3. resize gray without padding
    resizeDevice(m_param.batch_size, m_output_src_device, m_param.dst_w, m_param.dst_h,
        m_output_resize_device, m_param.src_w, m_param.src_h, utils::ColorMode::GRAY, m_src2dst); // note: 320*320 -> (src_w, src_h)
}

void u2net::U2NET::showMask(const std::vector<cv::Mat>& imgsBatch, const int& cvDelayTime)
{
    float* output_mask_device = m_output_resize_device;
    for (size_t j = 0; j < imgsBatch.size(); j++)
    {
        checkRuntime(cudaMemcpy(m_output_mask_host, output_mask_device + j * m_param.src_w * m_param.src_h,
            sizeof(float) * m_param.src_w * m_param.src_h, cudaMemcpyDeviceToHost));
        cv::Mat img_mask(m_param.src_h, m_param.src_w, CV_32FC1, m_output_mask_host);
        img_mask.convertTo(img_mask, CV_8UC1);
        cv::imshow("img_mask", img_mask);
        cv::waitKey(cvDelayTime);
    }
}

void u2net::U2NET::saveMask(const std::vector<cv::Mat>& imgsBatch, const std::string& savePath, const int& batchSize, const int& batchi)
{
    float* output_mask_device = m_output_resize_device;
    for (size_t j = 0; j < imgsBatch.size(); j++)
    {
        checkRuntime(cudaMemcpy(m_output_mask_host, output_mask_device + j * m_param.src_w * m_param.src_h,
            sizeof(float) * m_param.src_w * m_param.src_h, cudaMemcpyDeviceToHost));
        cv::Mat img_mask(m_param.src_h, m_param.src_w, CV_32FC1, m_output_mask_host);
        img_mask.convertTo(img_mask, CV_8UC1);
        int imgi = batchi * batchSize + j;
		cv::imwrite(savePath + "_" + std::to_string(imgi) + ".jpg", img_mask);
		cv::waitKey(1); // waitting for writting imgs
    }
}

void u2net::U2NET::reset()
{
}

__global__
void u2net_div_max_device_kernel(int batch_size, float* src, int img_height, int img_width, int img_volume, float* maxVals)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx < img_volume && dy < batch_size)
    {
        src[dy * img_volume + dx] /= maxVals[dy];
    }
}

static __device__
float u2net_norm_device(float val, float scale, float min_val, float max_val)
{
    return scale * (val  - min_val) / (max_val - min_val);
}

__global__
void u2net_norm_pred_device_kernel(int batch_size, float* src, int img_height, int img_width, int img_area, float scale, float* minVals, float* maxVals)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x;
    int dy = blockDim.y * blockIdx.y + threadIdx.y;
    if (dx < img_area && dy < batch_size)
    {
        src[dy * img_area + dx] = u2net_norm_device(src[dy * img_area + dx], scale, minVals[dy], maxVals[dy]);
    }
}

void u2netDivMaxDevice(const int& batchSize, float* src, int srcWidth, int srcHeight, int channel, float* maxVals)
{
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((srcWidth * srcHeight * channel + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int img_volume = channel * srcHeight * srcWidth;
    int img_height = srcHeight;
    int img_width = srcWidth;
    u2net_div_max_device_kernel << < grid_size, block_size, 0, nullptr >> > (batchSize, src, img_height, img_width, img_volume, maxVals);

}

void u2netNormPredDevice(const int& batchSize, float* src, int srcWidth, int srcHeight, float scale, float* minVals, float* maxVals)
{
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((srcWidth * srcHeight * 1 + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    int img_area = srcHeight * srcWidth;
    int img_height = srcHeight;
    int img_width = srcWidth;
    u2net_norm_pred_device_kernel << < grid_size, block_size, 0, nullptr >> > (batchSize, src, img_height, img_width, img_area, scale, minVals, maxVals);
}