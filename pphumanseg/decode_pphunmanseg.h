#pragma once
#include"../utils/utils.h"
#include"../utils/common_include.h"

namespace pphunmanseg
{
	void decodeDevice(int batchSize, float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight);
}