#pragma once
#include"../utils/common_include.h"
#include"../utils/utils.h"

#define checkRuntime(op)  __check_cuda_runtime((op), #op, __FILE__, __LINE__)

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line);

#define BLOCK_SIZE 8

//note: resize rgb with padding
void resizeDevice(const int& batch_size, float* src, int src_width, int src_height,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, utils::AffineMat matrix);

//overload:resize rgb with padding, but src's type is uin8
void resizeDevice(const int& batch_size, unsigned char* src, int src_width, int src_height,
    float* dst, int dstWidth, int dstHeight,
    float paddingValue, utils::AffineMat matrix);

// overload: resize rgb/gray without padding
void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight,
    utils::ColorMode mode, utils::AffineMat matrix);

void bgr2rgbDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

void normDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight,
    utils::InitParameter norm_param);

void hwc2chwDevice(const int& batch_size, float* src, int srcWidth, int srcHeight,
    float* dst, int dstWidth, int dstHeight);

void decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight);

// nms fast
void nmsDeviceV1(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea);

// nms sort
void nmsDeviceV2(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea,
    int* idx, float* conf);