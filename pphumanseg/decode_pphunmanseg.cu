#include"decode_pphunmanseg.h"
#include"../utils/kernel_function.h"

__global__
void decode_pphunmanseg_device_kernel(int batch_size,
	float* src, int src_width, int src_height, int src_area, int src_volum,
	float* dst, int dst_width, int dst_height, int dst_area, int dst_volum)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x; // "srcArea" dim
	int dy = blockDim.y * blockIdx.y + threadIdx.y; // "batch size" dim
	if (dx >= dst_area || dy >= batch_size)
	{
		return;
	}
	// img(x, y)'index in channel 1 = dy * src_volum + dx + src_area: 
	// img(x, y)'index in channel 0 = dy * src_volum + dx
	dst[dy * dst_volum + dx] = (src[dy * src_volum + dx + src_area] > src[dy * src_volum + dx] ? 1.f : 0.f);
}
// python: result = np.argmax(output[0], axis=1).astype(np.uint8)
void pphunmanseg::decodeDevice(int batchSize, float* src, int srcWidth, int srcHeight, float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int src_area  = srcWidth * srcHeight;
	int src_volum = srcWidth * srcHeight * 2;
	int dst_area  = dstWidth * dstHeight;
	int dst_volum = dstWidth * dstHeight * 1;
	decode_pphunmanseg_device_kernel << < grid_size, block_size, 0, nullptr >> > (batchSize, 
		src, srcWidth, srcHeight, src_area, src_volum,
		dst, dstWidth, dstHeight, dst_area, dst_volum);
}