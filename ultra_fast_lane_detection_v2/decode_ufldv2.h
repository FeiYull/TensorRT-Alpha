#pragma once
#include"../utils/utils.h"
#include"../utils/kernel_function.h"

namespace ufld
{
	// python:dst = src[yLow:yHigh, xLow:xHigh] -> [yLow, yHigh), [xLow, xHigh)
	void cropDevice(int batchSize, int xLow, int xHigh, int yLow, int yHigh,
		float* src, int srcWidth, int srcHeight, 
		float* dst, int dstWidth, int dstHeight);



	//void decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight);
	//void transposeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight);
}
