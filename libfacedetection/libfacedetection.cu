#include"libfacedetection.h"

void calFeatureMapSize(const cv::Size& size, float* featureMapSize)
{
    // channel dim
    int p1_c = 51;
    int p2_c = 34;
    int p3_c = 34;
    int p4_c = 51;

    int h0 = int(int((size.height + 1) / 2) / 2);
    int w0 = int(int((size.width + 1) / 2) / 2);

    // P1(downsample by 8)
    int p1_h = int(h0 / 2);
    int p1_w = int(w0 / 2);

    // P2(downsample by 16)
    int p2_h = int(p1_h / 2);
    int p2_w = int(p1_w / 2);

    // P3(downsample by 32)
    int p3_h = int(p2_h / 2);
    int p3_w = int(p2_w / 2);

    // P4(downsample by 64)
    int p4_h = int(p3_h / 2);
    int p4_w = int(p3_w / 2);

    // todo
    featureMapSize[0] = (float)p1_h;
    featureMapSize[1] = (float)p1_w;
    featureMapSize[2] = (float)p1_c;

    featureMapSize[3] = (float)p2_h;
    featureMapSize[4] = (float)p2_w;
    featureMapSize[5] = (float)p2_c;

    featureMapSize[6] = (float)p3_h;
    featureMapSize[7] = (float)p3_w;
    featureMapSize[8] = (float)p3_c;

    featureMapSize[9] = p4_h;
    featureMapSize[10] = p4_w;
    featureMapSize[11] = p4_c;
}

void calPriorBox(float* featureMapSize, const float* minSizes, const int* dim2, const cv::Size& size, float* priorBox)
{
    float steps[4] = { 8, 16, 32, 64 };
    /* std::vector<float> min_size;
     min_size.reserve(3);*/
    cv::Vec4f anchor; // cx cy s_ky s_kx

    int idx = 0;
    for (size_t k = 0; k < 4; k++) // todo
    {
        //min_size = minSizes[k];
        for (size_t i = 0; i < featureMapSize[k * 3 + 0]; i++)
        {
            for (size_t j = 0; j < featureMapSize[k * 3 + 1]; j++)
            {
                for (size_t m = 0; m < dim2[k]; m++)
                {
                    priorBox[idx++] = ((float)j + 0.5) * steps[k] / size.width; // cx
                    priorBox[idx++] = ((float)i + 0.5) * steps[k] / size.height; // cy
                    priorBox[idx++] = minSizes[k * 3 + m] / size.width; // s_kx
                    priorBox[idx++] = minSizes[k * 3 + m] / size.height; // s_ky
                }

            }

        }

    }

}

LibFaceDet::LibFaceDet(const utils::InitParameter& param) : m_param(param)
{
    // const params
    m_min_sizes_device = nullptr;    // 4 * 3
    m_feat_hw_host_device = nullptr; // 4 * 3
    m_prior_boxes_device = nullptr;  // 18984 * 4
    m_variances_device = nullptr;    // 2 * 1
    checkRuntime(cudaMalloc(&m_min_sizes_device, 4 * 3 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_feat_hw_host_device, 4 * 3 * sizeof(float)));

    checkRuntime(cudaMalloc(&m_variances_device, 2 * 1 * sizeof(float)));
    m_feat_hw_host = new float[4 * 3];     // 4 * 3

    // input
    m_input_src_device = nullptr;
    m_input_hwc_device = nullptr;
    checkRuntime(cudaMalloc(&m_input_src_device, param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
    checkRuntime(cudaMalloc(&m_input_hwc_device, param.batch_size * 3 * param.src_h * param.src_w * sizeof(float)));
   
    // output
    m_output_loc_device = nullptr;
    m_output_conf_device = nullptr;
    m_output_iou_device = nullptr;
    m_output_objects_device = nullptr;
    m_output_objects_width = 17; // 17: xyxy(4) + score(1) + class(1) + keepflag(1) + landmarks(10)

    int output_objects_size = param.batch_size * (1 + param.topK * m_output_objects_width); // 1: count
    checkRuntime(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];
    m_objectss.resize(param.batch_size);
}

LibFaceDet::~LibFaceDet()
{
    // const params
    checkRuntime(cudaFree(m_min_sizes_device));
    checkRuntime(cudaFree(m_feat_hw_host_device));
    checkRuntime(cudaFree(m_prior_boxes_device));
    checkRuntime(cudaFree(m_variances_device));
    delete[] m_feat_hw_host;
    delete[] m_prior_boxes_host;

    // input
    checkRuntime(cudaFree(m_input_src_device));
    checkRuntime(cudaFree(m_input_hwc_device));
   
    // output
    checkRuntime(cudaFree(m_output_loc_device));
    checkRuntime(cudaFree(m_output_conf_device));
    checkRuntime(cudaFree(m_output_iou_device));
    checkRuntime(cudaFree(m_output_objects_device));
    delete[] m_output_objects_host;
}

bool LibFaceDet::init(const std::vector<unsigned char>& trtFile)
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
    this->m_context->setBindingDimensions(0, nvinfer1::Dims4(m_param.batch_size, 3, m_param.src_h, m_param.src_w));

    // 2. get output's dim
    auto get_area = [](const nvinfer1::Dims& dims) {
        int area = 1;
        for (int i = 1; i < dims.nbDims; i++)
        {
            if (dims.d[i] != 0)
            {
                area *= dims.d[i];
            }
        }
        return area;
    };

    m_output_loc_dims  = this->m_context->getBindingDimensions(1);
    m_output_conf_dims = this->m_context->getBindingDimensions(2);
    m_output_iou_dims  = this->m_context->getBindingDimensions(3);
    
    m_total_objects = m_output_loc_dims.d[1];  // 18984
   
    // 3. malloc
    checkRuntime(cudaMalloc(&m_prior_boxes_device, m_total_objects * 4 * sizeof(float))); // const params
    m_prior_boxes_host = new float[m_total_objects * 4];  // 18984 * 4
    checkRuntime(cudaMalloc(&m_output_loc_device, m_param.batch_size * m_total_objects * 14 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_conf_device,m_param.batch_size * m_total_objects * 2 * sizeof(float)));
    checkRuntime(cudaMalloc(&m_output_iou_device, m_param.batch_size * m_total_objects * 1 * sizeof(float)));

    // 4. init const params
    // min sizes
    checkRuntime(cudaMemcpy(m_min_sizes_device, m_min_sizes_host, sizeof(float) * 4 * 3, cudaMemcpyHostToDevice));
    
    // feature map size
    calFeatureMapSize(cv::Size(m_param.src_w, m_param.src_h), m_feat_hw_host);
    checkRuntime(cudaMemcpy(m_feat_hw_host_device, m_feat_hw_host, sizeof(float) * 4 * 3, cudaMemcpyHostToDevice));
    
    // prior boxes
    calPriorBox(m_feat_hw_host, m_min_sizes_host, m_min_sizes_host_dim, cv::Size(m_param.src_w, m_param.src_h), m_prior_boxes_host);
    //cv::Mat bbox_tmp(m_total_objects, 4, CV_32FC1, m_prior_boxes_host);
    checkRuntime(cudaMemcpy(m_prior_boxes_device, m_prior_boxes_host, sizeof(float) * m_total_objects * 4, cudaMemcpyHostToDevice));

    // variances
    checkRuntime(cudaMemcpy(m_variances_device, m_variances_host, sizeof(float) * 2, cudaMemcpyHostToDevice));
    return true;
}

void LibFaceDet::check()
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

void LibFaceDet::copy(const std::vector<cv::Mat>& imgsBatch)
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

void LibFaceDet::preprocess(const std::vector<cv::Mat>& imgsBatch)
{
    // 1. hwc2chw
    hwc2chwDevice(m_param.batch_size, m_input_src_device, m_param.src_w, m_param.src_h,
        m_input_hwc_device, m_param.src_w, m_param.src_h);
#if 0
    {

        float* phost = new float[3 * m_param.src_h * m_param.src_w];
        float* pdevice = m_input_hwc_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * 3 * m_param.src_h * m_param.src_w,
                sizeof(float) * 3 * m_param.src_h * m_param.src_w, cudaMemcpyDeviceToHost));

            cv::Mat tmp = imgsBatch[j].clone();

            cv::Mat b(m_param.src_h, m_param.src_w, CV_32FC1, phost);
            cv::Mat g(m_param.src_h, m_param.src_w, CV_32FC1, phost + 1 * m_param.src_h * m_param.src_w);
            cv::Mat r(m_param.src_h, m_param.src_w, CV_32FC1, phost + 2 * m_param.src_h * m_param.src_w);
            std::vector<cv::Mat> bgr{ b, g, r };
            cv::Mat ret;
            cv::merge(bgr, ret);
            ret.convertTo(ret, CV_8UC3);
            cv::imshow("ret", ret);
            cv::waitKey(1);

            cv::Mat img_ = imgsBatch[j].clone();
        }
        delete[] phost;

    }
#endif

}

bool LibFaceDet::infer()
{
    float* bindings[] = { m_input_hwc_device, m_output_loc_device, m_output_conf_device, m_output_iou_device};
    bool context = m_context->executeV2((void**)bindings);
    return context;
}

void LibFaceDet::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
#if 0 // valid
    {
        float* phost_loc  = new float[m_total_objects * 14];
        float* phost_conf = new float[m_total_objects * 2];
        float* phost_iou  = new float[m_total_objects * 1];
        
        float* pdevice_loc  = m_output_loc_device;
        float* pdevice_conf = m_output_conf_device;
        float* pdevice_iou  = m_output_iou_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost_loc,  pdevice_loc  + j * m_total_objects * 14, sizeof(float) * m_total_objects * 14, cudaMemcpyDeviceToHost));
            checkRuntime(cudaMemcpy(phost_conf, pdevice_conf + j * m_total_objects * 2,  sizeof(float) * m_total_objects * 2,  cudaMemcpyDeviceToHost));
            checkRuntime(cudaMemcpy(phost_iou,  pdevice_iou  + j * m_total_objects * 1,  sizeof(float) * m_total_objects * 1,  cudaMemcpyDeviceToHost));
            //cv::Mat img_loc(m_total_objects, 14, CV_32FC1, phost_loc);

            //save to binary
            utils::saveBinaryFile(phost_loc,  m_total_objects * 14, "loc.bin");
            utils::saveBinaryFile(phost_conf, m_total_objects * 2,  "conf.bin");
            utils::saveBinaryFile(phost_iou,  m_total_objects * 1,  "iou.bin");

        }
        delete[] phost_loc;
        delete[] phost_conf;
        delete[] phost_iou;
    }
#endif // 0
    // decode
    decodeLibFaceDetDevice(
        m_min_sizes_device,
        m_feat_hw_host_device,
        m_prior_boxes_device,
        m_variances_device,

        m_param.src_w, m_param.src_h,
        m_param.conf_thresh, m_param.batch_size, m_total_objects,

        m_output_loc_device, 14,
        m_output_conf_device, 2,
        m_output_iou_device, 1,
        m_output_objects_device, m_output_objects_width, m_param.topK  // 17: xyxy(4) + score(1) + class(1) + keepflag(1) + landmarks(10)
    );
#if 0 // valid 
    {
        // loc
        float* p_loc_host = new float[m_total_objects * 14];
        float* p_loc_device = nullptr;
        p_loc_device = m_output_loc_device;
        for (size_t i = 0; i < imgsBatch.size(); i++)
        {
            checkRuntime(cudaMemcpy(p_loc_host, p_loc_device + i * m_total_objects * 14,
                m_total_objects * 14 * sizeof(float), cudaMemcpyDeviceToHost));
            cv::Mat img_loc_(m_total_objects, 14, CV_32FC1, p_loc_host); 
        }
        delete[] p_loc_host;

        // m_output_objects_device
        float* p_obj_host = new float[1 + m_output_objects_width * m_param.topK];   // 17: xyxy(4) + score(1) + class(1) + keepflag(1) + landmarks(10)
        float* p_obj_device = m_output_objects_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(p_obj_host, p_obj_device + j * (1 + m_output_objects_width * m_param.topK), 
                sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
            int num_candidates = p_obj_host[0];
            cv::Mat img_bbox1(m_param.topK, m_output_objects_width, CV_32FC1, p_obj_host + 1);
        }
        delete[] p_obj_host;
    }
#endif // 0

    // nms
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_output_objects_width * m_param.topK + 1);
#if 0 // valid
    {
        float* phost = new float[1 + m_output_objects_width * m_param.topK];   // 17: xyxy(4) + score(1) + class(1) + keepflag(1) + landmarks(10)
        float* pdevice = m_output_objects_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * (1 + m_output_objects_width * m_param.topK),
                sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
            int num_candidates = phost[0];
            cv::Mat img_bbox2(m_param.topK, m_output_objects_width, CV_32FC1, phost + 1);
        }
        delete[] phost;
    }
#endif // 0

    // copy result
    checkRuntime(cudaMemcpy(m_output_objects_host, m_output_objects_device, 
        m_param.batch_size * sizeof(float)* (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));

    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        for (size_t i = 0; i < num_boxes; i++)
        {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag)
            {
                utils::Box bbox(ptr[0], ptr[1], ptr[2], ptr[3], ptr[4], (int)ptr[5], 5); // 5: 5 face landmark points
                bbox.land_marks.emplace_back(cv::Point2i(ptr[7], ptr[8]));
                bbox.land_marks.emplace_back(cv::Point2i(ptr[9], ptr[10]));
                bbox.land_marks.emplace_back(cv::Point2i(ptr[11], ptr[12]));
                bbox.land_marks.emplace_back(cv::Point2i(ptr[13], ptr[14]));
                bbox.land_marks.emplace_back(cv::Point2i(ptr[15], ptr[16]));

                m_objectss[bi].emplace_back(bbox); 
            }
        }

    }
}

std::vector<std::vector<utils::Box>> LibFaceDet::getObjectss() const
{
    return this->m_objectss;
}

void LibFaceDet::reset()
{
    checkRuntime(cudaMemset(m_output_objects_device, 0, sizeof(float) * m_param.batch_size * (1 + m_output_objects_width * m_param.topK)));
    for (size_t bi = 0; bi < m_param.batch_size; bi++)
    {
        m_objectss[bi].clear();
    }
}

__global__
void decode_face_det_device_kernel(float* minSizes, float* feat_hw, float* priorBoxes, float* variances,
    int srcImgWidth, int srcImgHeight,
    float confThreshold, int batchSize, int srcHeight,
    float* srcLoc, int srcLocWidth, int srcLocArea,
    float* srcConf, int srcConfWidth, int srcConfArea,
    float* srcIou, int srcIouWidth, int srcIouArea,
    float* dst, int dstWidth, int topK, int dstArea)
{
    int dx = blockDim.x * blockIdx.x + threadIdx.x; // "src's rpw" dim
    int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
    if (dx >= srcHeight || dy >= batchSize)
    {
        return;
    }
    // conf(softmax)
    float* pitem_conf = srcConf + dy * srcConfArea + dx * srcConfWidth;
    float* pitem_iou = srcIou + dy * srcIouArea + dx * srcIouWidth;
    if (pitem_iou[0] < 0)
    {
        pitem_iou[0] = 0;
    }
    if (pitem_iou[0] > 1)
    {
        pitem_iou[0] = 1;
    }
    float e0 = expf(pitem_conf[0]);
    float e1 = expf(pitem_conf[1]);
    float exp_sum = e0 + e1;
    //pitem_conf[0] = e0 / exp_sum;
    pitem_conf[1] = e1 / exp_sum;
    float score = sqrt(pitem_conf[1] * pitem_iou[0]);
    if (score <= confThreshold)
    {
        return;
    }

    int index = atomicAdd(dst + dy * dstArea, 1);

    //int index = atomicAdd(&(dst + dy * dstWidth)[0], 1);
    if (index >= topK) // dstHeight : topK
    {
        return;
    }

    // bbox
    float* pitem_loc = srcLoc + dy * srcLocArea + dx * srcLocWidth;
    pitem_loc[0] = priorBoxes[4 * dx] + pitem_loc[0] * variances[0] * priorBoxes[4 * dx + 2];
    pitem_loc[1] = priorBoxes[4 * dx + 1] + pitem_loc[1] * variances[0] * priorBoxes[4 * dx + 3];
    pitem_loc[2] = priorBoxes[4 * dx + 2] * expf(pitem_loc[2] * variances[1]);
    pitem_loc[3] = priorBoxes[4 * dx + 3] * expf(pitem_loc[3] * variances[1]);

    // xywh2xyxy
    pitem_loc[0] -= pitem_loc[2] / 2;
    pitem_loc[1] -= pitem_loc[3] / 2;
    pitem_loc[2] += pitem_loc[0];
    pitem_loc[3] += pitem_loc[1];

    pitem_loc[0] *= srcImgWidth;
    pitem_loc[1] *= srcImgHeight;
    pitem_loc[2] *= srcImgWidth;
    pitem_loc[3] *= srcImgHeight;

    // 5 landmarks
    pitem_loc[4] = (priorBoxes[4 * dx] + pitem_loc[4] * variances[0] * priorBoxes[4 * dx + 2]) * srcImgWidth;
    pitem_loc[6] = (priorBoxes[4 * dx] + pitem_loc[6] * variances[0] * priorBoxes[4 * dx + 2]) * srcImgWidth;
    pitem_loc[8] = (priorBoxes[4 * dx] + pitem_loc[8] * variances[0] * priorBoxes[4 * dx + 2]) * srcImgWidth;
    pitem_loc[10] = (priorBoxes[4 * dx] + pitem_loc[10] * variances[0] * priorBoxes[4 * dx + 2]) * srcImgWidth;
    pitem_loc[12] = (priorBoxes[4 * dx] + pitem_loc[12] * variances[0] * priorBoxes[4 * dx + 2]) * srcImgWidth;

    pitem_loc[5] = (priorBoxes[4 * dx + 1] + pitem_loc[5] * variances[0] * priorBoxes[4 * dx + 3]) * srcImgHeight;
    pitem_loc[7] = (priorBoxes[4 * dx + 1] + pitem_loc[7] * variances[0] * priorBoxes[4 * dx + 3]) * srcImgHeight;
    pitem_loc[9] = (priorBoxes[4 * dx + 1] + pitem_loc[9] * variances[0] * priorBoxes[4 * dx + 3]) * srcImgHeight;
    pitem_loc[11] = (priorBoxes[4 * dx + 1] + pitem_loc[11] * variances[0] * priorBoxes[4 * dx + 3]) * srcImgHeight;
    pitem_loc[13] = (priorBoxes[4 * dx + 1] + pitem_loc[13] * variances[0] * priorBoxes[4 * dx + 3]) * srcImgHeight;

    // get dst
    float* pitem_dst = dst + dy * dstArea + index * dstWidth + 1;  // note: not dx but index

    // method1
    //memcpy(pitem_dst, pitem_loc, srcLocWidth * sizeof(float)); // dstWidth:14 + 1
    //pitem_dst[14] = score;

    // method2
    /*pitem_dst[0] = pitem_loc[0];
    pitem_dst[1] = pitem_loc[1];
    pitem_dst[2] = pitem_loc[2];
    pitem_dst[3] = pitem_loc[3];
    pitem_dst[4] = pitem_loc[4];
    pitem_dst[5] = pitem_loc[5];
    pitem_dst[6] = pitem_loc[6];
    pitem_dst[7] = pitem_loc[7];
    pitem_dst[8] = pitem_loc[8];
    pitem_dst[9] = pitem_loc[9];
    pitem_dst[10] = pitem_loc[10];
    pitem_dst[11] = pitem_loc[11];
    pitem_dst[12] = pitem_loc[12];
    pitem_dst[13] = pitem_loc[13];
    pitem_dst[14] = score;*/

    // xyxy + landmarks + score -> // 17: xyxy(4) + score(1) + class(1) + keepflag(1) + landmarks(10)
    // xyxy
    pitem_dst[0] = pitem_loc[0];
    pitem_dst[1] = pitem_loc[1];
    pitem_dst[2] = pitem_loc[2];
    pitem_dst[3] = pitem_loc[3];
    // score
    pitem_dst[4] = score;
    // class
    pitem_dst[5] = 1; // 1:face, 0:no-face
    // feepflag
    pitem_dst[6] = 1; // 1:keep, 0: ignore
    // landmarks
    pitem_dst[7] = pitem_loc[4];
    pitem_dst[8] = pitem_loc[5];
    pitem_dst[9] = pitem_loc[6];
    pitem_dst[10] = pitem_loc[7];
    pitem_dst[11] = pitem_loc[8];
    pitem_dst[12] = pitem_loc[9];
    pitem_dst[13] = pitem_loc[10];
    pitem_dst[14] = pitem_loc[11];
    pitem_dst[15] = pitem_loc[12];
    pitem_dst[16] = pitem_loc[13];
}

void decodeLibFaceDetDevice(float* minSizes, float* feat_hw, float* priorBoxes, float* variances,
    int srcImgWidth, int srcImgHeight,
    float confThreshold, int batchSize, int srcHeight,
    float* srcLoc, int srcLocWidth,
    float* srcConf, int srcConfWidth,
    float* srcIou, int srcIouWidth,
    float* dst, int dstWidth, int dstHeight)
{
    dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
    dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
        (batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

    // todo
    int src_loc_area = srcHeight * srcLocWidth;   // 18984*14
    int src_conf_area = srcHeight * srcConfWidth; // 18984*2
    int src_iou_area = srcHeight * srcIouWidth;   // 18984*1
    int dst_area = dstHeight * dstWidth + 1;      // topK*17 + 1

    decode_face_det_device_kernel << < grid_size, block_size, 0, nullptr >> > (
        minSizes, feat_hw, priorBoxes, variances,
        srcImgWidth, srcImgHeight,
        confThreshold, batchSize, srcHeight,
        srcLoc, srcLocWidth, src_loc_area,
        srcConf, srcConfWidth, src_conf_area,
        srcIou, srcIouWidth, src_iou_area,
        dst, dstWidth, dstHeight, dst_area);
}
