// pch.cpp: 与预编译标头对应的源文件
#include"./utils/yolo.h"
#include "pch.h"
#include"./yolov8/yolov8.h"
// 当使用预编译的头时，需要使用此源文件，编译才能成功。

void getAimsInfo(const std::vector<std::vector<utils::Box>>& objectss, float(*res_array)[6])
{
	for (const auto& objects : objectss)
	{
		for (const auto& box : objects)
		{
			res_array[0][0] = box.left;
			res_array[0][1] = box.top;
			res_array[0][2] = box.right;
			res_array[0][3] = box.bottom;
			res_array[0][4] = box.label;
			res_array[0][5] = box.confidence;

			++res_array;
		}
	}
}


// c++ code

void* Init(
	const char* trt_file_path,
	int src_w,
	int src_h,
	float conf_thresh,
	float iou_thresh,
	int num_class
)

{
	// parameters
	utils::InitParameter param;

	param.input_output_names = { "images",  "output0" };
	param.batch_size = 1;
	param.src_h = src_h;
	param.src_w = src_w;
	param.dst_h = 640;
	param.dst_w = 640;
	param.iou_thresh = iou_thresh;
	param.conf_thresh = conf_thresh;
	param.num_class = num_class;

	YOLOV8* yolov8 = new YOLOV8(param);

	std::vector<unsigned char> trt_file = utils::loadModel(trt_file_path);
	if (trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		return nullptr;
	}

	if (!yolov8->init(trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
		return nullptr;
	}
	yolov8->check();
	return yolov8;
}


// 2. img inference 
void Detect(void* yolo, int rows, int cols, unsigned char* src_data, float(*res_array)[6])

{
	YOLOV8* yolov8 = (YOLOV8*)yolo;
	
	cv::Mat frame = cv::Mat(rows, cols, CV_8UC3, src_data);

	std::vector<cv::Mat> imgs_batch(1, frame.clone());

	yolov8->reset();

	yolov8->copy(imgs_batch);

	utils::DeviceTimer d_t1; yolov8->preprocess(imgs_batch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; yolov8->infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; yolov8->postprocess(imgs_batch); float t3 = d_t3.getUsedTime();

	sample::gLogInfo << 
		"preprocess time = " << t1 << "; "
		"infer time = " << t2 << "; "
		"postprocess time = " << t3 << std::endl;

	getAimsInfo(yolov8->getObjectss(), res_array);	
}
