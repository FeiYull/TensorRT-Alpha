#pragma once
#include"../utils/yolo.h"
#include"../utils/kernel_function.h"

class YOLOX : public yolo::YOLO
{
public:
	YOLOX(const utils::InitParameter& param);
	~YOLOX();
	virtual bool init(const std::vector<unsigned char>& trtFile);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
private:
	float* m_input_resize_without_padding_device;
	int m_resized_w;
	int m_resized_h;
};
void copyWithPaddingDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, float paddingValue);