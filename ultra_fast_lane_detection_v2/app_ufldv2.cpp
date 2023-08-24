#include"ufldv2.h"

void setParameters(UFLD_Params& initParameters)
{
	//initParameters.class_names = utils::dataSets::coco80;
	//initParameters.num_class = 80; // for coco

	initParameters.batch_size = 8;

	// culane
	// 
	//num_row = 72
	//num_col = 81
	//self.row_anchor = np.linspace(0.42, 1, num_row)
	//self.col_anchor = np.linspace(0, 1, num_col)

	//initParameters.dst_h = 320;
	//initParameters.dst_w = 1600;
	//initParameters.crop_ratio = 0.6f;

	// tusimple
	// 
	//num_row = 56
	//num_col = 41
	//self.row_anchor = np.linspace(160, 710, num_row) / 720
	//self.col_anchor = np.linspace(0, 1, num_col)

	initParameters.dst_h = 320;
	initParameters.dst_w = 800;
	initParameters.crop_ratio = 0.8f;




	initParameters.resize_height = int(initParameters.dst_h / initParameters.crop_ratio);
	initParameters.resize_width = initParameters.dst_w;

	initParameters.input_output_names = { "images",  "output0", "output1", "output2", "output3" };
	//initParameters.conf_thresh = 0.25f;
	//initParameters.iou_thresh = 0.45f;

	initParameters.means[0] = 0.485;
	initParameters.means[1] = 0.456;
	initParameters.means[2] = 0.406;
	initParameters.stds[0] = 0.229;
	initParameters.stds[1] = 0.224;
	initParameters.stds[2] = 0.225;

	initParameters.save_path = "";
}

void task(UFLDV2& ufld, const UFLD_Params& param, std::vector<cv::Mat>& imgsBatch, const int& delayTime, const int& batchi,
	const bool& isShow, const bool& isSave)
{
	ufld.copy(imgsBatch);
	utils::DeviceTimer d_t1; ufld.preprocess(imgsBatch);  float t1 = d_t1.getUsedTime();
	utils::DeviceTimer d_t2; ufld.infer();				  float t2 = d_t2.getUsedTime();
	utils::DeviceTimer d_t3; ufld.postprocess(imgsBatch); float t3 = d_t3.getUsedTime();
	sample::gLogInfo << "preprocess time = " << t1 / param.batch_size << "; "
		"infer time = " << t2 / param.batch_size << "; "
		"postprocess time = " << t3 / param.batch_size << std::endl;

}

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv,
		{
			"{model 	|| tensorrt model file	   }"
			"{batch_size|| batch size              }"
			"{video     || video's path			   }"
			"{img       || image's path			   }"
			"{cam_id    || camera's device id	   }"
			"{show      || if show the result	   }"
			"{savePath  || save path, can be ignore}"
		});

	/************************************************************************************************
	* init
	*************************************************************************************************/
	// parameters
	UFLD_Params param;
	setParameters(param);
	// path
	/*std::string model_path = "../../data/yolov8/yolov8n.trt";
	std::string video_path = "../../data/people.mp4";
	std::string image_path = "../../data/bus.jpg";*/
	std::string model_path = ""; // ............................................................
	std::string video_path = ""; // ............................................................
	std::string image_path = ""; // ............................................................

	// camera' id
	int camera_id = 0;

	// get input
	utils::InputStream source;
	source = utils::InputStream::IMAGE;
	//source = utils::InputStream::VIDEO;
	//source = utils::InputStream::CAMERA;

	// update params from command line parser
	int size = -1; // w or h
	int batch_size = 8;
	bool is_show = false;
	bool is_save = false;
	if (parser.has("model"))
	{
		model_path = parser.get<std::string>("model");
		sample::gLogInfo << "model_path = " << model_path << std::endl;
	}
	/*if (parser.has("size"))
	{
		size = parser.get<int>("size");
		sample::gLogInfo << "size = " << size << std::endl;
		param.dst_h = param.dst_w = size;
	}*/
	if (parser.has("batch_size"))
	{
		batch_size = parser.get<int>("batch_size");
		sample::gLogInfo << "batch_size = " << batch_size << std::endl;
		param.batch_size = batch_size;
	}
	if (parser.has("video"))
	{
		source = utils::InputStream::VIDEO;
		video_path = parser.get<std::string>("video");
		sample::gLogInfo << "video_path = " << video_path << std::endl;
	}
	if (parser.has("img"))
	{
		source = utils::InputStream::IMAGE;
		image_path = parser.get<std::string>("img");
		sample::gLogInfo << "image_path = " << image_path << std::endl;
	}
	if (parser.has("cam_id"))
	{
		source = utils::InputStream::CAMERA;
		camera_id = parser.get<int>("cam_id");
		sample::gLogInfo << "camera_id = " << camera_id << std::endl;
	}
	if (parser.has("show"))
	{
		is_show = true;
		sample::gLogInfo << "is_show = " << is_show << std::endl;
	}
	if (parser.has("savePath"))
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
	UFLDV2 ufld(param);

	// read model
	std::vector<unsigned char> trt_file = utils::loadModel(model_path);
	if (trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		return -1;
	}
	// init model
	if (!ufld.init(trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
		return -1;
	}
	ufld.check();
	/************************************************************************************************
	* recycle
	*************************************************************************************************/
	cv::Mat frame;
	std::vector<cv::Mat> imgs_batch;
	imgs_batch.reserve(param.batch_size);
	sample::gLogInfo << imgs_batch.capacity() << std::endl;
	int i = 0; // debug
	int batchi = 0;
	while (capture.isOpened())
	{
		if (batchi >= total_batches && source != utils::InputStream::CAMERA)
		{
			break;
		}
		if (imgs_batch.size() < param.batch_size) // get input
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
				task(ufld, param, imgs_batch, delay_time, batchi, is_show, is_save);
				imgs_batch.clear(); // clear
				//sample::gLogInfo << imgs_batch.capacity() << std::endl;
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
			task(ufld, param, imgs_batch, delay_time, batchi, is_show, is_save);
			imgs_batch.clear(); // clear
			//sample::gLogInfo << imgs_batch.capacity() << std::endl;
			batchi++;
		}
	}
	return  -1;
}