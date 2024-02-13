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
    yolov4::decodeDevice(m_param, m_output_src_device, 4 + m_param.num_class, m_total_objects, m_output_area,
        m_output_objects_device, m_output_objects_width, m_param.topK);
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1);
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
                float x_lt = m_dst2src.v0 * ptr[0] * m_param.dst_w + m_dst2src.v1 * ptr[1] * m_param.dst_h + m_dst2src.v2;
                float y_lt = m_dst2src.v3 * ptr[0] * m_param.dst_w + m_dst2src.v4 * ptr[1] * m_param.dst_h + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] * m_param.dst_w + m_dst2src.v1 * ptr[3] * m_param.dst_h + m_dst2src.v2;
                float y_rb = m_dst2src.v3 * ptr[2] * m_param.dst_w + m_dst2src.v4 * ptr[3] * m_param.dst_h + m_dst2src.v5;

                m_objectss[bi].emplace_back(x_lt, y_lt, x_rb, y_rb, ptr[4], (int)ptr[5]);
            }
        }

    }
}