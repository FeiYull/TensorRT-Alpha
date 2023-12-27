#pragma once
#include"../utils/yolo.h"
#include"../utils/utils.h"
class YOLO_NAS : public yolo::YOLO
{
public:
	YOLO_NAS(const utils::InitParameter& param);
	~YOLO_NAS();
	virtual bool init(const std::vector<unsigned char>& trtFile);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);

private:
	float* m_input_resize_padding_device;
	cv::Size m_resize_shape;
	int m_pad_top;
	int m_pad_left;
};