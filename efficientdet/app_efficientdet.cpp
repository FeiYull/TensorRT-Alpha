#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"efficientdet.h"

void setParameters(utils::InitParameter& initParameters)
{
	initParameters.class_names = utils::dataSets::coco91;
	initParameters.num_class = 91; // for coco91

	initParameters.batch_size = 8;
	// efficientdet d0
	initParameters.dst_h = 512;
	initParameters.dst_w = 512;

	// efficientdet d1
	/*initParameters.dst_h = 640;
	initParameters.dst_w = 640;*/

	// efficientdet d2
	/*initParameters.dst_h = 768;
	initParameters.dst_w = 768;*/

	// efficientdet d3
	/*initParameters.dst_h = 896;
	initParameters.dst_w = 896;*/

	// efficientdet d4 d5 d6 ...

	initParameters.input_output_names = { "input",  "num_detections", "detection_boxes", "detection_scores", "detection_classes", };
	initParameters.conf_thresh = 0.45f; 
	initParameters.topK = 100; // be determined while exporting the onnx!
	initParameters.save_path = "";
}

void task(EfficientDet& efficient_det, const utils::InitParameter& param, std::vector<cv::Mat>& imgsBatch, const int& delayTime, const int& batchi, 
	const bool& isShow, const bool& isSave)
{
	efficient_det.copy(imgsBatch);
	utils::DeviceTimer d_t1; efficient_det.preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; efficient_det.infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; efficient_det.postprocess(imgsBatch); float t3 = d_t3.getUsedTime();
	sample::gLogInfo << "preprocess time = " << t1 / param.batch_size << "; "
		"infer time = " << t2 / param.batch_size << "; "
		"postprocess time = " << t3 / param.batch_size << std::endl;
	if(isShow)
		utils::show(efficient_det.getObjectss(), param.class_names, delayTime, imgsBatch);
	if(isSave)
		utils::save(efficient_det.getObjectss(), param.class_names, param.save_path, imgsBatch, param.batch_size, batchi);
	efficient_det.reset();
}

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv,
		{
			"{model 	|| tensorrt model file	}"
			"{size      || image (h, w), eg: 640}"
			"{batch_size|| batch size           }"
			"{video     || video's path			}"
			"{img       || image's path			}"
			"{cam_id    || camera's device id	}"
			"{show      || if show the result	}"
			"{savePath  || save path, can be ignore}"
		});
	// parameters
	utils::InitParameter param;
	setParameters(param);
	// path
	std::string model_path = "../../data/efficientdet/efficientdet0.trt";
	std::string video_path = "../../data/people.mp4";
	std::string image_path = "../../data/6406403.jpg";
	int camera_id = 0; // camera' id

	// get input
	utils::InputStream source;
	//source = utils::InputStream::IMAGE;
	source = utils::InputStream::VIDEO;
	//source = utils::InputStream::CAMERA;

	// update params from command line parser
	int size = -1; // w or h
	int batch_size = 8;
	bool is_show = false;
	bool is_save = false;
	if(parser.has("model"))
	{
		model_path = parser.get<std::string>("model");
		sample::gLogInfo << "model_path = " << model_path << std::endl;
	}
	if(parser.has("size"))
	{
		size = parser.get<int>("size");
		sample::gLogInfo << "size = " << size << std::endl;
		param.dst_h = param.dst_w = size;
	}
	if(parser.has("batch_size"))
	{
		batch_size = parser.get<int>("batch_size");
		sample::gLogInfo << "batch_size = " << batch_size << std::endl;
		param.batch_size = batch_size;
	}
	if(parser.has("video"))
	{
		source = utils::InputStream::VIDEO;
		video_path = parser.get<std::string>("video");
		sample::gLogInfo << "video_path = " << video_path << std::endl;
	}
	if(parser.has("img"))
	{
		source = utils::InputStream::IMAGE;
		image_path = parser.get<std::string>("img");
		sample::gLogInfo << "image_path = " << image_path << std::endl;
	}
	if(parser.has("cam_id"))
	{
		source = utils::InputStream::CAMERA;
		camera_id = parser.get<int>("cam_id");
		sample::gLogInfo << "camera_id = " << camera_id << std::endl;
	}
	if(parser.has("show"))
	{
		is_show = true;
		sample::gLogInfo << "is_show = " << is_show << std::endl;
	}
	if(parser.has("savePath"))
	{
		is_save = true;
		param.save_path = parser.get<std::string>("savePath");
		sample::gLogInfo << "save_path = " << param.save_path << std::endl;
	}

	int total_batches = 0;
	int delay_time = 1;
	cv::VideoCapture capture;
	if (!setInputStream(source, image_path, video_path, camera_id,
		capture, total_batches, delay_time, param))
	{
		sample::gLogError << "read the input data errors!" << std::endl;
		return -1;
	}
	
	EfficientDet efficient_det(param);

	// read model
	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
	if (trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		return -1;
	}
	// init model
	if (!efficient_det.init(trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
		return -1;
	}
	efficient_det.check();
	cv::Mat frame;
	std::vector<cv::Mat> imgs_batch;
	imgs_batch.reserve(param.batch_size);
	sample::gLogInfo << imgs_batch.capacity() << std::endl;
	int batchi = 0;
	while (capture.isOpened())
	{
		if (batchi >= total_batches && source != utils::InputStream::CAMERA)
		{
			break;
		}
		if (imgs_batch.size() < param.batch_size) 
		{
			if (source != utils::InputStream::IMAGE)
			{
				capture.read(frame);
			}
			else
			{
				frame = cv::imread(image_path);
			}

			if (frame.empty())
			{
				sample::gLogWarning << "no more video or camera frame" << std::endl;
				task(efficient_det, param, imgs_batch, delay_time, batchi, is_show, is_save);
				imgs_batch.clear(); 
				batchi++;
				break;
			}
			else
			{
				imgs_batch.emplace_back(frame.clone());
			}

		}
		else // infer
		{
			task(efficient_det, param, imgs_batch, delay_time, batchi, is_show, is_save);
			imgs_batch.clear();
			batchi++;
		}
	}
	return  -1;
}

