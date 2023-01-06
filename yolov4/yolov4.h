#pragma once
#include"../utils/yolo.h"
#include"../utils/utils.h"
class YOLOV4 : public yolo::YOLO
{
public:
	YOLOV4(const utils::InitParameter& param);
	~YOLOV4();
	virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
};