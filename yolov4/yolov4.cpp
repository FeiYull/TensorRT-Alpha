#include"yolov4.h"
#include"decode_yolov4.h"

YOLOV4::YOLOV4(const utils::InitParameter& param) :yolo::YOLO(param)
{
}

YOLOV4::~YOLOV4()
{
}

void YOLOV4::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
#if 0 // valid
    {
        float* phost = new float[m_param.batch_size * m_output_area];
        float* pdevice = m_output_src_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            checkRuntime(cudaMemcpy(phost, pdevice + j * m_output_area, sizeof(float) * m_output_area, cudaMemcpyDeviceToHost));
            cv::Mat prediction(m_total_objects, m_param.num_class + 4, CV_32FC1, phost);
        }
        delete[] phost;
    }
#endif // 0

    // decode
    yolov4::decodeDevice(m_param, m_output_src_device, 4 + m_param.num_class, m_total_objects, m_output_area,
        m_output_objects_device, m_output_objects_width, m_param.topK);
#if 0 // valid
    {
        float* phost = new float[m_param.batch_size * (1 + m_output_objects_width * m_param.topK)];
        float* pdevice = m_output_objects_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            /* cv::imshow("srcimg", imgsBatch[j]);
             cv::waitKey(0);*/
            checkRuntime(cudaMemcpy(phost, pdevice + j * (1 + m_output_objects_width * m_param.topK),
                sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
            int num_candidates = phost[0];
            cv::Mat prediction(m_param.topK, m_output_objects_width, CV_32FC1, phost + 1);
        }
        delete[] phost;
    }
#endif // 0

    // nms
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1);
#if 0 // valid
    {
        float* phost = new float[m_param.batch_size * (1 + m_output_objects_width * m_param.topK)];
        float* pdevice = m_output_objects_device;
        for (size_t j = 0; j < imgsBatch.size(); j++)
        {
            /*cv::imshow("srcimg", imgsBatch[j]);
            cv::waitKey(0);*/
            checkRuntime(cudaMemcpy(phost, pdevice + j * (1 + m_output_objects_width * m_param.topK),
                sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
            int num_candidates_objects = phost[0];
            cv::Mat prediction(m_param.topK, m_output_objects_width, CV_32FC1, phost + 1);
        }
        delete[] phost;
    }
#endif // 0

    // copy result from gpu to cpu
    checkRuntime(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * (1 + 7 * m_param.topK), cudaMemcpyDeviceToHost));

    // transform to source image coordinate
    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        for (size_t i = 0; i < num_boxes; i++)
        {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag)
            {
                // yolov4 
                float x_lt = m_dst2src.v0 * ptr[0] * m_param.dst_w + m_dst2src.v1 * ptr[1] * m_param.dst_h + m_dst2src.v2; // left & top
                float y_lt = m_dst2src.v3 * ptr[0] * m_param.dst_w + m_dst2src.v4 * ptr[1] * m_param.dst_h + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] * m_param.dst_w + m_dst2src.v1 * ptr[3] * m_param.dst_h + m_dst2src.v2; // right & bottom
                float y_rb = m_dst2src.v3 * ptr[2] * m_param.dst_w + m_dst2src.v4 * ptr[3] * m_param.dst_h + m_dst2src.v5;

                m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4], (int)ptr[5]);
            }
        }

    }
}