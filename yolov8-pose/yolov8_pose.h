#pragma once
#include"../utils/yolo.h"
#include"../utils/utils.h"
class YOLOv8Pose : public yolo::YOLO
{
public:
	YOLOv8Pose(const utils::InitParameter& param);
	~YOLOv8Pose();
	virtual bool init(const std::vector<unsigned char>& trtFile);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual void reset();

public:
	void showAndSave(const std::vector<std::string>& classNames,
		const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch, float* avg_times);

private:
	float* m_output_src_transpose_device;
	float* m_output_objects_device;
	float* m_output_objects_host;
	int m_output_objects_width;

	const size_t m_nkpts;
	std::vector<cv::Point2i> m_skeleton;
	std::vector<cv::Scalar> m_kpt_color;
	std::vector<cv::Scalar> m_limb_color;
};