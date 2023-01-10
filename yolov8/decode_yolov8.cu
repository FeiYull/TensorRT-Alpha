#include "decode_yolov8.h"

__global__ void decode_yolov8_device_kernel(int batch_size, int  num_class, int topK, float conf_thresh,
	float* src, int srcWidth, int srcHeight, int srcArea,
	float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; // "srcArea" dim
	int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
	if (dx >= srcHeight || dy >= batch_size)
	{
		return;
	}
	float* pitem = src + dy * srcArea + dx * srcWidth;

	// find max Pr(Classi/Object)
	//float* class_confidence = pitem + 5;  // Pr(Class0/Object)
	float* class_confidence = pitem + 4;    // Pr(Class0/Object)
	float confidence = *class_confidence++; // Pr(Class1/Object)
	int label = 0;
	for (int i = 1; i < num_class; ++i, ++class_confidence)
	{
		if (*class_confidence > confidence)
		{
			confidence = *class_confidence;
			label = i;
		}
	}
	//confidence *= objectness; // Pr(Class0/Object) * Pr(Object)
	if (confidence < conf_thresh)
	{
		return;
	}

	// parray:count, box1, box2, box3(count:)
	// parray[0]:count
	// atomicAdd -> count += 1
	// atomicAdd: return old_count
	//int index = atomicAdd(dst + dy * dstArea, 1);
	//assert(dy == 1);

	int index = atomicAdd(dst + dy * dstArea, 1);

	if (index >= topK)
	{
		return;
	}
	// xywh -> xyxy
	float cx = *pitem++;
	float cy = *pitem++;
	float width = *pitem++;
	float height = *pitem++;

	float left = cx - width * 0.5f;
	float top = cy - height * 0.5f;
	float right = cx + width * 0.5f;
	float bottom = cy + height * 0.5f;

	/*float left = cx;
	float top = cy;
	float right = width;
	float bottom = height;*/
	
	//float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left; // todo
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;


	*pout_item++ = confidence;
	*pout_item++ = label;
	*pout_item++ = 1;// 1 = keep, 0 = ignore
}

void yolov8::decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = 1 + dstWidth * dstHeight;

	decode_yolov8_device_kernel << < grid_size, block_size, 0, nullptr >> > (param.batch_size, param.num_class, param.topK, param.conf_thresh,
		src, srcWidth, srcHeight, srcArea,
		dst, dstWidth, dstHeight, dstArea);
}


__global__ void transpose_device_kernel(int batch_size,
	float* src, int srcWidth, int srcHeight, int srcArea,
	float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; // "srcArea" dim
	int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
	if (dx >= dstHeight || dy >= batch_size)
	{
		return;
	}
	float* p_dst_row = dst + dy * dstArea + dx * dstWidth; // row = dx
	float* p_src_col = src + dy * srcArea + dx; // col = dx

	for (int i = 0; i < dstWidth; i++)
	{
		p_dst_row[i] = p_src_col[i * srcWidth];
	}
}

/*
	src:
				8400 ->
	84	  x1 x2  ...... x8400
	 |    y1 y2  ...... y8400
	 V	  w1 w2  ...... w8400
		  h1 h2  ...... h8400
		  c0  .
		  c1  .
		  c2  .
		  .   .
		  .   .
		  .
		  c79

*/
void yolov8::transposeDevice(utils::InitParameter param, 
float* src, int srcWidth, int srcHeight, int srcArea, 
float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = dstWidth * dstHeight;

	transpose_device_kernel << < grid_size, block_size, 0, nullptr >> > (param.batch_size,
		src, srcWidth, srcHeight, srcArea,
		dst, dstWidth, dstHeight, dstArea);
}


