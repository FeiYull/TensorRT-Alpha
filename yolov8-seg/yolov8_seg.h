#pragma once
#include<Eigen/Dense>
#include<opencv2/core/eigen.hpp>
#include"../utils/yolo.h"
#include"../utils/utils.h"
class YOLOv8Seg : public yolo::YOLO
{
public:
	YOLOv8Seg(const utils::InitParameter& param);
	~YOLOv8Seg();
	virtual bool init(const std::vector<unsigned char>& trtFile);
	virtual void preprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual bool infer();
	virtual void postprocess(const std::vector<cv::Mat>& imgsBatch);
	virtual void reset();

public:
	void showAndSave(const std::vector<std::string>& classNames,
		const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch);

private:
	float* m_output_src_transpose_device;
	float* m_output_seg_device; // eg:116 * 8400, 116=4+80+32
	float* m_output_objects_device;

	float* m_output_seg_host;
	float* m_output_objects_host;

	int m_output_objects_width; // 39 = 32 + 7, 7:left, top, right, bottom, confidence, class, keepflag; 
	int m_output_src_width; // 116 = 4+80+32, 4:xyxy; 80:coco label; 32:seg
	nvinfer1::Dims m_output_seg_dims;
	int m_output_obj_area;
	int m_output_seg_area;
	int m_output_seg_w;
	int m_output_seg_h;

	cv::Mat m_mask160; 
	Eigen::MatrixXf m_mask_eigen160;
	cv::Rect m_thresh_roi160;
	cv::Rect m_thresh_roisrc;
	float m_downsample_scale;
	cv::Mat m_mask_src;
	cv::Mat m_img_canvas;
};