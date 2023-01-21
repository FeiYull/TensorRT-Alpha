#include"../utils/common_include.h"
#include"../utils/utils.h"
#include"facemesh.h"

void setParameters(utils::InitParameter& faceMeshParam)
{
	faceMeshParam.batch_size = 4;
	faceMeshParam.dst_h = 192;
	faceMeshParam.dst_w = 192;
	faceMeshParam.input_output_names = { "image",  "preds", "confs" };
	faceMeshParam.scale = 127.5f; // img = img.float() / 127.5 - 1.0
	faceMeshParam.means[0] = 1.f;
	faceMeshParam.means[1] = 1.f;
	faceMeshParam.means[2] = 1.f;
	faceMeshParam.save_path = "";
}

int main(int argc, char** argv)
{
	cv::CommandLineParser parser(argc, argv,
		{
			"{model 	|| tensorrt model file	}"
			"{size      || image (h, w), eg: 640}"
			"{imgs_dir  || 4 images dir  		}"
			"{show      || if show the result	}"
			"{savePath  || save path, can be ignore}"
		});

	/************************************************************************************************
	* init
	*************************************************************************************************/
	// parameters
	utils::InitParameter param;
	setParameters(param);

	// path
	std::string model_path = "../../data/facemesh/facemesh.trt";
	std::string image_dir = "../../data/";
	
	// update params from command line parser
	int size = -1; // w or h
	int batch_size = 4;
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
	if(parser.has("imgs_dir"))
	{
		image_dir = parser.get<std::string>("imgs_dir");
		sample::gLogInfo << "image_dir = " << image_dir << std::endl;
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

	FaceMesh face_mesh(param);
	// read model
	std::vector<unsigned char> face_mesh_trt_file = utils::loadModel(model_path);
	if (face_mesh_trt_file.empty())
	{
		sample::gLogError << "trt_file is empty!" << std::endl;
		return -1;
	}

	// init model
	if (!face_mesh.init(face_mesh_trt_file))
	{
		sample::gLogError << "initEngine() ocur errors!" << std::endl;
		return -1;
	}
	face_mesh.check();
	
	std::vector<cv::Mat> imgs_batch;
	std::vector<cv::Mat> imgs_src;
	imgs_batch.reserve(4);
	imgs_src.reserve(4);
	std::string imgs_path[4] = {image_dir + "face1.jpg", 
								image_dir + "face2.jpg", 
								image_dir + "face3.jpg", 
								image_dir + "face5.jpg" };
	for (int i = 0; i < 4; ++i)
	{
		cv::Mat img_src = cv::imread(imgs_path[i]);
		imgs_batch.emplace_back(img_src);
		imgs_src.emplace_back(img_src);
	}

	assert(imgs_batch.size() == param.batch_size);

	// facemesh
	face_mesh.resize(imgs_batch); // resize to 192 * 192
	face_mesh.copy(imgs_batch);
	face_mesh.preprocess(imgs_batch);
	face_mesh.infer();
	face_mesh.postprocess(imgs_batch);
	std::vector<std::vector<cv::Point2f>> land_markss = face_mesh.getLandMarkss();

	assert(land_markss.size() == imgs_src.size());
	for (size_t bi = 0; bi < land_markss.size(); bi++)
	{
		for (size_t i = 0; i < 468; i++)
		{
			cv::circle(imgs_src[bi], cv::Point2i(land_markss[bi][i].x, land_markss[bi][i].y), 1, cv::Scalar(0, 255, 0), 1 , cv::LINE_AA, 0);
		}
		if(is_show)
		{
			cv::imshow("vis", imgs_src[bi]);
			cv::waitKey(0);
		}
		if(is_save)
		{
			cv::imwrite(param.save_path + std::to_string(bi) + ".jpg", imgs_src[bi]);
		}
	}

	face_mesh.reset();
	return  -1;
}