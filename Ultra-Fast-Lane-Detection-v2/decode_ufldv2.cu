#include "decode_ufldv2.h"

__global__
void crop_device_kernel(int batchSize, int xLow, int xHigh, int yLow, int yHigh,
	float* src, int srcWidth, int srcHeight, int srcArea, int srcVolume,
	float* dst, int dstWidth, int dstHeight, int dstArea, int dstVolume)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < dstArea && dy < batchSize)
	{
		int dst_y = dx / dstWidth; // dst row
		int dst_x = dx % dstWidth; // dst col
		float* pdst = dst + dy * dstVolume + dst_y * dstWidth * 3 + dst_x * 3;
		int src_y = dst_y + yLow; // base address + offset
		int src_x = dst_x + xLow;
		if (src_y < yHigh && src_x < xHigh)
		{
			float* psrc = src + dy * srcVolume + src_y * srcWidth * 3 + src_x * 3;
			pdst[0] = psrc[0];
			pdst[1] = psrc[1];
			pdst[2] = psrc[2];
		}
	}
}

void ufld::cropDevice(int batchSize, int xLow, int xHigh, int yLow, int yHigh, 
	float* src, int srcWidth, int srcHeight, 
	float* dst, int dstWidth, int dstHeight)
{
	assert(xLow < srcWidth  && xHigh <= srcWidth && 
		   yLow < srcHeight && yHigh <= srcHeight);
	assert((xHigh - xLow) == dstWidth);
	assert((yHigh - yLow) == dstHeight);

	int src_area = srcHeight * srcWidth;
	int dst_area = dstHeight * dstWidth;

	int src_volume = 3 * srcHeight * srcWidth;
	int dst_volume = 3 * dstHeight * dstWidth;

	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
	crop_device_kernel << < grid_size, block_size, 0, nullptr >> > (batchSize, xLow, xHigh, yLow, yHigh,
		src, srcWidth, srcHeight, src_area, src_volume,
		dst, dstWidth, dstHeight, dst_area, dst_volume);
}
