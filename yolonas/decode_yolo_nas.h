#pragma once
#include"../utils/utils.h"
#include"../utils/kernel_function.h"

namespace yolo_nas
{
	void decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcLength, float* dst, int dstWidth, int dstHeight);
}
