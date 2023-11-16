#include "decode_yolov4.h"

__global__ void decode_yolov4_device_kernel(int batch_size, int  num_class, int topK, float conf_thresh,
									float* src, int srcWidth, int srcHeight, int srcArea, 
									float* dst, int dstWidth, int dstHeight, int dstArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx >= srcHeight || dy >= batch_size)
	{
		return;
	}
	float* pitem = src + dy * srcArea + dx * srcWidth;
	float* class_confidence = pitem + 4;
	float confidence = *class_confidence++;
	int label = 0;
	for (int i = 1; i < num_class; ++i, ++class_confidence)
	{
		if (*class_confidence > confidence)
		{
			confidence = *class_confidence;
			label = i;
		}
	}
	if (confidence < conf_thresh)
	{
		return;
	}
	int index = atomicAdd(dst + dy * dstArea, 1);
	if (index >= topK)
	{
		return;
	}
	float cx = *pitem++;
	float cy = *pitem++;
	float width = *pitem++;
	float height = *pitem++;

	float left = cx;
	float top = cy;
	float right = width;
	float bottom = height;
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left; 
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;
	*pout_item++ = confidence;
	*pout_item++ = label;
	*pout_item++ = 1;
}

static __device__ float box_iou(
	float aleft, float atop, float aright, float abottom,
	float bleft, float btop, float bright, float bbottom
) {
	float cleft = max(aleft, bleft);
	float ctop = max(atop, btop);
	float cright = min(aright, bright);
	float cbottom = min(abottom, bbottom);

	float c_area = max(cright - cleft, 0.0f) * max(cbottom - ctop, 0.0f);
	if (c_area == 0.0f)
		return 0.0f;

	float a_area = max(0.0f, aright - aleft) * max(0.0f, abottom - atop);
	float b_area = max(0.0f, bright - bleft) * max(0.0f, bbottom - btop);
	return c_area / (a_area + b_area - c_area);
}

void yolov4::decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = 1 + dstWidth * dstHeight;
	
	decode_yolov4_device_kernel << < grid_size, block_size, 0, nullptr >> >(param.batch_size, param.num_class, param.topK, param.conf_thresh,
																	 src, srcWidth, srcHeight, srcArea, 
																	 dst, dstWidth, dstHeight, dstArea);
}

