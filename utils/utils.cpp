#include"../utils/utils.h"
/************************************************************************************************
* struct
*************************************************************************************************/


/************************************************************************************************
* function
*************************************************************************************************/
void utils::saveBinaryFile(float* vec, size_t len, const std::string& file)
{
	std::ofstream  out(file, std::ios::out | std::ios::binary);
	if (!out.is_open())
		return;
	out.write((const char*)vec, sizeof(float) * len);
	out.close();
}

std::vector<uint8_t> utils::readBinaryFile(const std::string& file) 
{

	std::ifstream in(file, std::ios::in | std::ios::binary);
	if (!in.is_open())
		return {};

	in.seekg(0, std::ios::end);
	size_t length = in.tellg();

	std::vector<uint8_t> data;
	if (length > 0) {
		in.seekg(0, std::ios::beg);
		data.resize(length);

		in.read((char*)&data[0], length);
	}
	in.close();
	return data;
}


std::vector<unsigned char> utils::loadModel(const std::string& file)
{
    std::ifstream in(file, std::ios::in | std::ios::binary);
    if (!in.is_open())
    {
        return {};
    }
    in.seekg(0, std::ios::end);
    size_t length = in.tellg();

    std::vector<uint8_t> data;
    if (length > 0)
    {
        in.seekg(0, std::ios::beg);
        data.resize(length);
        in.read((char*)&data[0], length);
    }
    in.close();
    return data;
}
//std::string getSystemTimeStr()
//{
//	SYSTEMTIME st = { 0 };
//	GetLocalTime(&st);
//	/*std::string t = (st.wYear) + st.wMonth + st.wDay + st.wHour + st.wMinute + st.wMilliseconds;*/
//	std::string t =
//		std::to_string(st.wYear) + "." +
//		std::to_string(st.wMonth) + "." +
//		std::to_string(st.wDay) + "." +
//		std::to_string(st.wHour) + "." +
//		std::to_string(st.wMinute) + "." +
//		std::to_string(st.wSecond) + "." +
//		std::to_string(st.wMilliseconds);
//
//	return t;
//}

std::string utils::getSystemTimeStr()
{
	return std::to_string(std::rand()); 
}

bool utils::setInputStream(const utils::InputStream& source, const std::string& imagePath, const std::string& videoPath, const int& cameraID,
	cv::VideoCapture& capture, int& totalBatches, int& delayTime, utils::InitParameter& param)
{
	int total_frames = 0;
	switch (source)
	{
	case utils::InputStream::IMAGE:
		capture.open(imagePath); //cv::CAP_IMAGES : !< OpenCV Image Sequence (e.g. img_%02d.jpg)
		param.batch_size = 1;
		total_frames = 1;
		totalBatches = 1;
		//numFillFrames = 0;
		delayTime = 0;
		break;
	case utils::InputStream::VIDEO:
		capture.open(videoPath);
		total_frames = capture.get(cv::CAP_PROP_FRAME_COUNT);
		totalBatches = (total_frames % param.batch_size == 0) ?
			(total_frames / param.batch_size) : (total_frames / param.batch_size + 1);
		//numFillFrames = param.batch_size - total_frames % param.batch_size;
		break;
	case utils::InputStream::CAMERA:
		capture.open(cameraID);
		total_frames = INT_MAX;
		totalBatches = INT_MAX;
		//numFillFrames = 0;
		break;
	default:
		break;
	}
	param.src_h = capture.get(cv::CAP_PROP_FRAME_HEIGHT);
	param.src_w = capture.get(cv::CAP_PROP_FRAME_WIDTH);

	return capture.isOpened();
}

void utils::show(const std::vector<std::vector<utils::Box>>& objectss, const std::vector<std::string>& classNames,
	const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch)
{
	//for (size_t bi = 0; bi < objectss.size(); bi++)
	for (size_t bi = 0; bi < imgsBatch.size(); bi++)
	{
		if (!objectss.empty())
		{
			for (auto& box : objectss[bi])
			{
				cv::rectangle(imgsBatch[bi], cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
				cv::putText(imgsBatch[bi], cv::format("%.4f", box.confidence), cv::Point(box.left, box.top - 3), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				cv::putText(imgsBatch[bi], classNames[box.label], cv::Point(box.left, box.top + 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				if (!box.land_marks.empty()) // for facial landmarks
				{
					for (auto& pt:box.land_marks)
					{
						cv::circle(imgsBatch[bi], pt, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
					}
				}
			}
		}
		
		//cv::Mat img = imgsBatch[bi];
		cv::imshow("image", imgsBatch[bi]);
		cv::waitKey(cvDelayTime);
	}

}

void utils::save(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& classNames,
	const std::string& savePath, std::vector<cv::Mat>& imgsBatch, const int& batchSize, const int& batchi)
{
	//for (size_t bi = 0; bi < objectss.size(); bi++)
	for (size_t bi = 0; bi < imgsBatch.size(); bi++)
	{
		if (!objectss.empty())
		{
			for (auto& box : objectss[bi])
			{
				cv::rectangle(imgsBatch[bi], cv::Point(box.left, box.top), cv::Point(box.right, box.bottom), cv::Scalar(0, 255, 0), 2, cv::LINE_AA);
				cv::putText(imgsBatch[bi], cv::format("%.4f", box.confidence), cv::Point(box.left, box.top - 3), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				cv::putText(imgsBatch[bi], classNames[box.label], cv::Point(box.left, box.top + 12), cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(0, 0, 255), 1, cv::LINE_AA);
				if (!box.land_marks.empty())
				{
					for (auto& pt : box.land_marks)
					{
						cv::circle(imgsBatch[bi], pt, 1, cv::Scalar(255, 255, 255), 1, cv::LINE_AA, 0);
					}
				}
			}
		}
		
		int imgi = batchi * batchSize + bi;
		cv::imwrite(savePath + "_" + std::to_string(imgi) + ".jpg", imgsBatch[bi]);
		cv::waitKey(1); // waitting for writting imgs 
	}
}

/************************************************************************************************
* class
*************************************************************************************************/
utils::HostTimer::HostTimer()
{
    t1 = std::chrono::steady_clock::now();
}

float utils::HostTimer::getUsedTime()
{
    t2 = std::chrono::steady_clock::now();
    std::chrono::duration<double> time_used = std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
    return(1000 * time_used.count()); // ms
}

utils::HostTimer::~HostTimer()
{
}

utils::DeviceTimer::DeviceTimer()
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start);
}

float utils::DeviceTimer::getUsedTime()
{
	cudaEventRecord(end);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

utils::DeviceTimer::DeviceTimer(cudaStream_t stream)
{
	cudaEventCreate(&start);
	cudaEventCreate(&end);
	cudaEventRecord(start, stream);
}

float utils::DeviceTimer::getUsedTime(cudaStream_t stream)
{
	cudaEventRecord(end, stream);
	cudaEventSynchronize(end);
	float total_time;
	cudaEventElapsedTime(&total_time, start, end);
	return total_time;
}

utils::DeviceTimer::~DeviceTimer()
{
	cudaEventDestroy(start);
	cudaEventDestroy(end);
}