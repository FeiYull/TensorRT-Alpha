#pragma once
#include"../utils/common_include.h"


namespace utils 
{
    /************************************************************************************************
    * array
    *************************************************************************************************/
    namespace dataSets
    {
        const std::vector<std::string> coco80 = {
            "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"
        };

        const std::vector<std::string> face2 = {"non-face", "face"};
    }



    /************************************************************************************************
    * struct
    *************************************************************************************************/
    struct InitParameter
    {
        int num_class{ 80 }; // coco 
        std::vector<std::string> class_names;
        //std::vector<std::vector<float>> min_sizes;
        std::vector<std::string> input_output_names;

        bool dynamic_batch{ true };
        size_t batch_size;
        int src_h, src_w; // size of source image eg:unknow * unknow
        int dst_h, dst_w; // size of net's input, eg:640*640

        float scale{ 255.f };
        float means[3] = { 0.f, 0.f, 0.f };
        float stds[3] = { 1.f, 1.f, 1.f };

        float iou_thresh;
        float conf_thresh;

        int topK{ 1000 };
        std::string save_path;
    };


    // legacy
    struct CandidateObject
    {
        float mBboxAndkeyPoints[14]; // bbox:[x y w h] +  5 facial key points:[x1 y1 x2 y2 ...x5 y5]
        float mScore;
        bool  mIsGood;
        CandidateObject()
        {
            std::fill_n(mBboxAndkeyPoints, 14, FLT_MAX);
            mScore = FLT_MAX;
            mIsGood = true;
        }
        CandidateObject(float* bboxAndkeyPoints, float score, bool isGood) :
            mScore(score),
            mIsGood(isGood)
        {
            memcpy(mBboxAndkeyPoints, bboxAndkeyPoints, 14 * sizeof(float));
        }
    };

    struct Box
    {
        float left, top, right, bottom, confidence;
        int label;
        std::vector<cv::Point2i> land_marks;

        Box() = default;
        Box(float left, float top, float right, float bottom, float confidence, int label) :
            left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) {}

        Box(float left, float top, float right, float bottom, float confidence, int label, int numLandMarks) :
            left(left), top(top), right(right), bottom(bottom), confidence(confidence), label(label) 
        {
            land_marks.reserve(numLandMarks);
        }
    };

    enum class InputStream { IMAGE, VIDEO, CAMERA };

    enum class ColorMode { RGB, GRAY };

    struct AffineMat
    {
        float v0, v1, v2;
        float v3, v4, v5;
    };

    /************************************************************************************************
    * function
    *************************************************************************************************/
    void saveBinaryFile(float* vec, size_t len, const std::string& file);

    std::vector<uint8_t> readBinaryFile(const std::string& file);

    std::vector<unsigned char> loadModel(const std::string& file);

    std::string getSystemTimeStr();

    bool setInputStream(const InputStream& source, const std::string& imagePath, const std::string& videoPath, const int& cameraID,
        cv::VideoCapture& capture, int& totalBatches, int& delayTime, InitParameter& param);

    
    void show(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& classNames,
        const int& cvDelayTime, std::vector<cv::Mat>& imgsBatch);

    void save(const std::vector<std::vector<Box>>& objectss, const std::vector<std::string>& classNames,
        const std::string& savePath, std::vector<cv::Mat>& imgsBatch, const int& batchSize, const int& batchi);
    

    /************************************************************************************************
    * class
    *************************************************************************************************/

    class HostTimer
    {
    public:
        HostTimer();
        float getUsedTime(); // while timing for cuda code, add "cudaDeviceSynchronize();" before this
        ~HostTimer();

    private:
        std::chrono::steady_clock::time_point t1;
        std::chrono::steady_clock::time_point t2;
    };


    class DeviceTimer
    {
    public:
        DeviceTimer();
        float getUsedTime();
        // overload
        DeviceTimer(cudaStream_t ctream);
        float getUsedTime(cudaStream_t ctream);

        ~DeviceTimer();

    private:
        cudaEvent_t start, end;
    };
}
