#include"yolov8_pose.h"
#include"decode_yolov8_pose.h"

YOLOv8Pose::YOLOv8Pose(const utils::InitParameter& param) :yolo::YOLO(param),
m_nkpts(17)
{
    m_skeleton = { cv::Point2i(16, 14), cv::Point2i(14, 12), cv::Point2i(17, 15),
                   cv::Point2i(15, 13), cv::Point2i(12, 13), cv::Point2i(6, 12),
                   cv::Point2i(7, 13),  cv::Point2i(6, 7),   cv::Point2i(6, 8),
                   cv::Point2i(7, 9),   cv::Point2i(8, 10),  cv::Point2i(9, 11),
                   cv::Point2i(2, 3),   cv::Point2i(1, 2),   cv::Point2i(1, 3),
                   cv::Point2i(2, 4),   cv::Point2i(3, 5),   cv::Point2i(4, 6),
                   cv::Point2i(5, 7) };

    m_kpt_color = { cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),
                    cv::Scalar(0, 255, 0),    cv::Scalar(0, 255, 0),    cv::Scalar(255, 128, 0),
                    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
                    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(51, 153, 255),
                    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
                    cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255) };

    m_limb_color = { cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255), cv::Scalar(51, 153, 255),
                    cv::Scalar(51, 153, 255), cv::Scalar(255,  51, 255),cv::Scalar(255,  51, 255),
                    cv::Scalar(255,  51, 255),cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),
                    cv::Scalar(255, 128, 0),  cv::Scalar(255, 128, 0),  cv::Scalar(255, 128,   0),
                    cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),
                    cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),  cv::Scalar(0, 255,   0),
                    cv::Scalar(0, 255,   0) };
 
    m_output_objects_device = nullptr;
    m_output_objects_width = 58; // xywhc + points * 17 = 56  -> left, top, right, bottom, confidence, class, keepflag + points * 17 = 58
    int output_objects_size = param.batch_size * (1 + param.topK * m_output_objects_width); // 1: count
    CHECK(cudaMalloc(&m_output_objects_device, output_objects_size * sizeof(float)));
    m_output_objects_host = new float[output_objects_size];
    m_objectss.resize(param.batch_size);
}

YOLOv8Pose::~YOLOv8Pose()
{
    CHECK(cudaFree(m_output_objects_device));
    CHECK(cudaFree(m_output_src_device));
    CHECK(cudaFree(m_output_src_transpose_device));
    delete[] m_output_objects_host;
    m_output_src_device = nullptr;
}

bool YOLOv8Pose::init(const std::vector<unsigned char>& trtFile)
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
    m_output_dims = this->m_context->getBindingDimensions(1);
    m_total_objects = m_output_dims.d[2];
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
    CHECK(cudaMalloc(&m_output_src_transpose_device, m_param.batch_size * m_output_area * sizeof(float)));
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

void YOLOv8Pose::preprocess(const std::vector<cv::Mat>& imgsBatch)
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

void YOLOv8Pose::postprocess(const std::vector<cv::Mat>& imgsBatch)
{
    yolov8pose::transposeDevice(m_param, m_output_src_device, m_total_objects, 56, m_total_objects * 56,
        m_output_src_transpose_device, 56, m_total_objects);
    yolov8pose::decodeDevice(m_param, m_output_src_transpose_device, 56, m_total_objects, m_output_area,
        m_output_objects_device, m_output_objects_width, m_param.topK);
    nmsDeviceV1(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1);

    //nmsDeviceV2(m_param, m_output_objects_device, m_output_objects_width, m_param.topK, m_param.topK * m_output_objects_width + 1, m_output_idx_device, m_output_conf_device);
    CHECK(cudaMemcpy(m_output_objects_host, m_output_objects_device, m_param.batch_size * sizeof(float) * (1 + m_output_objects_width * m_param.topK), cudaMemcpyDeviceToHost));
}

void YOLOv8Pose::reset()
{
    CHECK(cudaMemset(m_output_objects_device, 0, sizeof(float) * m_param.batch_size * (1 + m_output_objects_width * m_param.topK)));
}

void YOLOv8Pose::showAndSave(const std::vector<std::string>& classNames, const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch, float* avg_times)
{
    if (!m_param.is_show && !m_param.is_save)
        return;
    cv::Scalar color = cv::Scalar(0, 255, 0);
    cv::Point bbox_points[1][4];
    const cv::Point* bbox_point0[1] = { bbox_points[0] };
    int num_points[] = { 4 };

    for (size_t bi = 0; bi < imgsBatch.size(); bi++)
    {
        int num_boxes = std::min((int)(m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1))[0], m_param.topK);
        for (size_t i = 0; i < num_boxes; i++)
        {
            float* ptr = m_output_objects_host + bi * (m_param.topK * m_output_objects_width + 1) + m_output_objects_width * i + 1;
            int keep_flag = ptr[6];
            if (keep_flag)
            {
                int label = (int)ptr[5];
                color = utils::Colors::color80[label];
                float x_lt = m_dst2src.v0 * ptr[0] + m_dst2src.v1 * ptr[1] + m_dst2src.v2;
                float y_lt = m_dst2src.v3 * ptr[0] + m_dst2src.v4 * ptr[1] + m_dst2src.v5;
                float x_rb = m_dst2src.v0 * ptr[2] + m_dst2src.v1 * ptr[3] + m_dst2src.v2;
                float y_rb = m_dst2src.v3 * ptr[2] + m_dst2src.v4 * ptr[3] + m_dst2src.v5;
                cv::rectangle(imgsBatch[bi], cv::Point(x_lt, y_lt), cv::Point(x_rb, y_rb), color, 2, cv::LINE_AA);
                cv::String det_info = m_param.class_names[label] + " " + cv::format("%.4f", ptr[4]);
                bbox_points[0][0] = cv::Point(x_lt, y_lt);
                bbox_points[0][1] = cv::Point(x_lt + det_info.size() * m_param.char_width, y_lt);
                bbox_points[0][2] = cv::Point(x_lt + det_info.size() * m_param.char_width, y_lt - m_param.det_info_render_width);
                bbox_points[0][3] = cv::Point(x_lt, y_lt - m_param.det_info_render_width);
                cv::fillPoly(imgsBatch[bi], bbox_point0, num_points, 1, color);
                cv::putText(imgsBatch[bi], det_info, bbox_points[0][0], cv::FONT_HERSHEY_DUPLEX, m_param.font_scale, cv::Scalar(255, 255, 255), 1, cv::LINE_AA); 
                cv::rectangle(imgsBatch[bi], cv::Point(x_lt, y_lt), cv::Point(x_rb, y_rb), color, 1, cv::LINE_AA);
                float* pkpt = ptr + 7;
                for (size_t pi = 0; pi < m_nkpts; pi++)
                {
                    float conf = pkpt[pi * 3 + 2];
                    if (conf < 0.5f)
                        continue;
                    pkpt[pi * 3] = m_dst2src.v0 * pkpt[pi * 3] + m_dst2src.v1 * pkpt[pi * 3 + 1] + m_dst2src.v2;
                    pkpt[pi * 3 + 1] = m_dst2src.v3 * pkpt[pi * 3] + m_dst2src.v4 * pkpt[pi * 3 + 1] + m_dst2src.v5;
                    if (pkpt[pi * 3] >= (float)m_param.src_w || pkpt[pi * 3] < 0 ||
                        pkpt[pi * 3 + 1] >= (float)m_param.src_h || pkpt[pi * 3 + 1] < 0)
                        continue;
                    cv::circle(imgsBatch[bi], cv::Size2i(int(pkpt[pi * 3 + 0]), int(pkpt[pi * 3 + 1])), 5, m_kpt_color[pi], -1, cv::LINE_AA);
                }
                for (size_t ki = 0; ki < m_skeleton.size(); ki++)
                {
                    float conf1 = pkpt[(m_skeleton[ki].x - 1) * 3 + 2];
                    float conf2 = pkpt[(m_skeleton[ki].y - 1) * 3 + 2];
                    if (conf1 < 0.5f || conf2 < 0.5f)
                        continue;

                    int x0 = int(pkpt[(m_skeleton[ki].x - 1) * 3 + 0]);
                    int y0 = int(pkpt[(m_skeleton[ki].x - 1) * 3 + 1]);

                    int x1 = int(pkpt[(m_skeleton[ki].y - 1) * 3 + 0]);
                    int y1 = int(pkpt[(m_skeleton[ki].y - 1) * 3 + 1]);
                    cv::Point2i pos1(x0, y0);
                    cv::Point2i pos2(x1, y1);

                    if (pos1.x % imgsBatch[bi].cols == 0 || pos1.y % imgsBatch[bi].rows == 0 || pos1.x < 0 or pos1.y < 0)
                        continue;
                    if (pos2.x % imgsBatch[bi].cols == 0 || pos2.y % imgsBatch[bi].rows == 0 || pos2.x < 0 or pos2.y < 0)
                        continue;
                    cv::line(imgsBatch[bi], pos1, pos2, m_limb_color[ki], 2, cv::LINE_AA);
                }
            }
        }
        cv::putText(imgsBatch[bi], "preprocess  time =" + cv::format("%.3f", avg_times[0]) + "ms", cv::Point(100, 100), cv::FONT_HERSHEY_DUPLEX, m_param.font_scale + 0.3, cv::Scalar(255, 0, 0), 1, cv::LINE_AA);
        cv::putText(imgsBatch[bi], "inference   time =" + cv::format("%.3f", avg_times[1]) + "ms", cv::Point(100, 135), cv::FONT_HERSHEY_DUPLEX, m_param.font_scale + 0.3, cv::Scalar(0, 255, 0), 1, cv::LINE_AA);
        cv::putText(imgsBatch[bi], "postprocess time =" + cv::format("%.3f", avg_times[2]) + "ms", cv::Point(100, 170), cv::FONT_HERSHEY_DUPLEX, m_param.font_scale + 0.3, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
        if (m_param.is_show)
        {
            cv::imshow(m_param.winname, imgsBatch[bi]);
            cv::waitKey(cvDelayTime);
        }
        if (m_param.is_save)
        {
            cv::imwrite(m_param.save_path + utils::getTimeStamp() + ".jpg", imgsBatch[bi]);
        }
    }
}