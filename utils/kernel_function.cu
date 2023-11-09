#include"../utils/kernel_function.h"
#include<math.h>

bool __check_cuda_runtime(cudaError_t code, const char* op, const char* file, int line) {
	if (code != cudaSuccess) {
		const char* err_name = cudaGetErrorName(code);
		const char* err_message = cudaGetErrorString(code);
		printf("runtime error %s:%d  %s failed. \n  code = %s, message = %s\n", file, line, op, err_name, err_message);
		return false;
	}
	return true;
}

__device__ 
void affine_project_device_kernel(utils::AffineMat* matrix, int x, int y, float* proj_x, float* proj_y)
{
	*proj_x = matrix->v0 * x + matrix->v1 * y + matrix->v2;
	*proj_y = matrix->v3 * x + matrix->v4 * y + matrix->v5;
}

__global__ 
void resize_rgb_padding_device_kernel(float* src, int src_width, int src_height, int src_area, int src_volume,
	float* dst, int dst_width, int dst_height, int dst_area, int dst_volume,
	int batch_size, float padding_value, utils::AffineMat matrix)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < dst_area && dy < batch_size)
	{
		int dst_y = dx / dst_width; 
		int dst_x = dx % dst_width; 
		float src_x = 0;
		float src_y = 0;
		affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);
		float c0 = padding_value, c1 = padding_value, c2 = padding_value;
		if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
		{
		}
		else
		{
			int y_low = floorf(src_y);
			int x_low = floorf(src_x);
			int y_high = y_low + 1;   
			int x_high = x_low + 1;   
			float const_values[] = { padding_value, padding_value, padding_value };
			float ly = src_y - y_low;
			float lx = src_x - x_low;
			float hy = 1 - ly;
			float hx = 1 - lx;
			float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; 
			float* v1 = const_values;
			float* v2 = const_values;
			float* v3 = const_values;
			float* v4 = const_values;

			if (y_low >= 0)
			{
				if (x_low >= 0)
					v1 = src + dy * src_volume + y_low * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v2 = src + dy * src_volume + y_low * src_width * 3 + x_high * 3;
			}

			if (y_high < src_height)
			{
				if (x_low >= 0)
					v3 = src + dy * src_volume + y_high * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v4 = src + dy * src_volume + y_high * src_width * 3 + x_high * 3;
			}
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}
		float* pdst = dst + dy * dst_volume + dst_y * dst_width * 3 + dst_x * 3;
		pdst[0] = c0;
		pdst[1] = c1;
		pdst[2] = c2;
	}
}

__global__
void resize_rgb_padding_device_kernel(unsigned char* src, int src_width, int src_height, int src_area, int src_volume,
	float* dst, int dst_width, int dst_height, int dst_area, int dst_volume,
	int batch_size, float padding_value, utils::AffineMat matrix)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < dst_area && dy < batch_size)
	{
		int dst_y = dx / dst_width;
		int dst_x = dx % dst_width;
		float src_x = 0;
		float src_y = 0;
		affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);
		float c0 = padding_value, c1 = padding_value, c2 = padding_value;
		if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
		{
		}
		else
		{
			int y_low = floorf(src_y); 
			int x_low = floorf(src_x); 
			int y_high = y_low + 1;
			int x_high = x_low + 1;
			unsigned char const_values[] = { 
				(unsigned char)padding_value, 
				(unsigned char)padding_value, 
				(unsigned char)padding_value }; 
			float ly = src_y - y_low;
			float lx = src_x - x_low;
			float hy = 1 - ly;
			float hx = 1 - lx;
			float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			unsigned char* v1 = const_values;
			unsigned char* v2 = const_values;
			unsigned char* v3 = const_values;
			unsigned char* v4 = const_values;
			if (y_low >= 0)
			{
				if (x_low >= 0)
					v1 = src + dy * src_volume + y_low * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v2 = src + dy * src_volume + y_low * src_width * 3 + x_high * 3;
			}
			if (y_high < src_height)
			{
				if (x_low >= 0)
					v3 = src + dy * src_volume + y_high * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v4 = src + dy * src_volume + y_high * src_width * 3 + x_high * 3;
			}
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}
		float* pdst = dst + dy * dst_volume + dst_y * dst_width * 3 + dst_x * 3;
		pdst[0] = c0;
		pdst[1] = c1;
		pdst[2] = c2;
	}
}
__global__
void resize_rgb_without_padding_device_kernel(float* src, int src_width, int src_height, int src_area, int src_volume,
	float* dst, int dst_width, int dst_height, int dst_area, int dst_volume,
	int batch_size, utils::AffineMat matrix)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < dst_area && dy < batch_size)
	{
		int dst_y = dx / dst_width;
		int dst_x = dx % dst_width;
		float src_x = 0;
		float src_y = 0;
		affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);
		float default_val = 114.f;
		float c0 = default_val, c1 = default_val, c2 = default_val;
		if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
		{
		}
		else
		{
			int y_low = floorf(fmaxf(src_y, 0.f)); 
			int x_low = floorf(fmaxf(src_x, 0.f)); 
			int y_high = min(y_low + 1, src_height - 1);
			int x_high = min(x_low + 1, src_width - 1);
			float const_values[] = { default_val, default_val, default_val };
			float ly = src_y - y_low;
			float lx = src_x - x_low;
			float hy = 1 - ly;
			float hx = 1 - lx;
			float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx; 
			float* v1 = const_values;
			float* v2 = const_values;
			float* v3 = const_values;
			float* v4 = const_values;

			if (y_low >= 0)
			{
				if (x_low >= 0)
					v1 = src + dy * src_volume + y_low * src_width * 3 + x_low * 3;

				if (x_high < src_width) 
					v2 = src + dy * src_volume + y_low * src_width * 3 + x_high * 3;
			}

			if (y_high < src_height)
			{
				if (x_low >= 0)
					v3 = src + dy * src_volume + y_high * src_width * 3 + x_low * 3;

				if (x_high < src_width)
					v4 = src + dy * src_volume + y_high * src_width * 3 + x_high * 3;
			}
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
			c1 = floorf(w1 * v1[1] + w2 * v2[1] + w3 * v3[1] + w4 * v4[1] + 0.5f);
			c2 = floorf(w1 * v1[2] + w2 * v2[2] + w3 * v3[2] + w4 * v4[2] + 0.5f);
		}
		float* pdst = dst + dy * dst_volume + dst_y * dst_width * 3 + dst_x * 3;
		pdst[0] = c0;
		pdst[1] = c1;
		pdst[2] = c2;
	}
}

__global__
void resize_gray_without_padding_device_kernel(float* src, int src_width, int src_height, int src_area, 
	float* dst, int dst_width, int dst_height, int dst_area, 
	int batch_size, utils::AffineMat matrix)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < dst_area && dy < batch_size)
	{
		int dst_y = dx / dst_width;
		int dst_x = dx % dst_width;
		float src_x = 0;
		float src_y = 0;
		affine_project_device_kernel(&matrix, dst_x, dst_y, &src_x, &src_y);
		float default_val = 114.f;
		float c0 = default_val;
		if (src_x < -1 || src_x >= src_width || src_y < -1 || src_y >= src_height)
		{
		}
		else
		{
			int y_low = floorf(fmaxf(src_y, 0.f));
			int x_low = floorf(fmaxf(src_x, 0.f));
			int y_high = min(y_low + 1, src_height - 1);
			int x_high = min(x_low + 1, src_width - 1);
			float const_values[] = { default_val};
			
			float ly = src_y - y_low;
			float lx = src_x - x_low;
			float hy = 1 - ly;
			float hx = 1 - lx;
			float w1 = hy * hx, w2 = hy * lx, w3 = ly * hx, w4 = ly * lx;
			
			float* v1 = const_values;
			float* v2 = const_values;
			float* v3 = const_values;
			float* v4 = const_values;

			if (y_low >= 0)
			{
				if (x_low >= 0)
					v1 = src + dy * src_area + y_low * src_width * 1 + x_low * 1;

				if (x_high < src_width) 
					v2 = src + dy * src_area + y_low * src_width * 1 + x_high * 1;
			}

			if (y_high < src_height)
			{
				if (x_low >= 0)
					v3 = src + dy * src_area + y_high * src_width * 1 + x_low * 1;

				if (x_high < src_width)
					v4 = src + dy * src_area + y_high * src_width * 1 + x_high * 1;
			}
			c0 = floorf(w1 * v1[0] + w2 * v2[0] + w3 * v3[0] + w4 * v4[0] + 0.5f);
		}
		float* pdst = dst + dy * dst_area + dst_y * dst_width * 1 + dst_x * 1;
		pdst[0] = c0;
	}
}

__global__ 
void bgr2rgb_device_kernel(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{
		int ch = dx % 3;
		assert(ch < 3);

		switch (ch)
		{
		case 0:
			dst[dy * img_volume + dx] = src[dy * img_volume + dx + 2];
			return;
		case 1:
			dst[dy * img_volume + dx] = src[dy * img_volume + dx];
			return;
		case 2:
			dst[dy * img_volume + dx] = src[dy * img_volume + dx - 2];
			return;
		}
	}
}

static __device__  
float norm_device(float val, float s, float mean, float std)
{
	return ((val / s) - mean) / std;
}

__global__ 
void norm_device_kernel(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume,
	float scale,
	float mean0, float mean1, float mean2,
	float std0, float std1, float std2)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{
		int ch = dx % 3;
		assert(ch < 3);

		switch (ch)
		{
		case 0:
			dst[dy * img_volume + dx] = norm_device(src[dy * img_volume + dx], scale, mean0, std0);
			break;
		case 1:
			dst[dy * img_volume + dx] = norm_device(src[dy * img_volume + dx], scale, mean1, std1);
			break;
		case 2:
			dst[dy * img_volume + dx] = norm_device(src[dy * img_volume + dx], scale, mean2, std2);
			break;
		}
	}
}

__global__ void hwc2chw_device_kernel(float* src, float* dst,
	int batch_size, int img_height, int img_width, int img_area, int img_volume)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dx < img_volume && dy < batch_size)
	{

		int ch = dx / img_area;
		assert(ch < 3);
		int sub_idx = dx % img_area;
		int row = sub_idx / img_width;
		int col = sub_idx % img_width;

		int dx_ = row * (img_width * 3) + col * 3 + ch;
		dst[dy * img_volume + dx] = src[dy * img_volume + dx_];
	}
}

void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight,
	float paddingValue, utils::AffineMat matrix)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int src_volume = 3 * srcHeight * srcWidth;
	int src_area = srcHeight * srcWidth;

	int dst_volume = 3 * dstHeight * dstWidth;
	int dst_area = dstHeight * dstWidth;

	resize_rgb_padding_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, srcWidth, srcHeight, src_area, src_volume,
		dst, dstWidth, dstHeight, dst_area, dst_volume,
		batchSize, paddingValue, matrix);
}

void resizeDevice(const int& batchSize, unsigned char* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight,
	float paddingValue, utils::AffineMat matrix)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int src_volume = 3 * srcHeight * srcWidth;
	int src_area = srcHeight * srcWidth;

	int dst_volume = 3 * dstHeight * dstWidth;
	int dst_area = dstHeight * dstWidth;

	resize_rgb_padding_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, srcWidth, srcHeight, src_area, src_volume,
		dst, dstWidth, dstHeight, dst_area, dst_volume,
		batchSize, paddingValue, matrix);
}

void resizeDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight, 
	utils::ColorMode mode, utils::AffineMat matrix)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int src_area = srcHeight * srcWidth;
	int dst_area = dstHeight * dstWidth;

	int src_volume = 3 * srcHeight * srcWidth;
	int dst_volume = 3 * dstHeight * dstWidth;
	
	switch (mode)
	{
	case utils::ColorMode::RGB:
		resize_rgb_without_padding_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, srcWidth, srcHeight, src_area, src_volume,
			dst, dstWidth, dstHeight, dst_area, dst_volume,
			batchSize, matrix);
		return;
	case utils::ColorMode::GRAY:
		resize_gray_without_padding_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, srcWidth, srcHeight, src_area, 
			dst, dstWidth, dstHeight, dst_area, batchSize, matrix);
		return;
	}
}

void bgr2rgbDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int img_volume = 3 * srcHeight * srcWidth;
	int img_area = srcHeight * srcWidth;
	int img_height = srcHeight;
	int img_width = srcWidth;
	bgr2rgb_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, dst, batchSize, img_height, img_width, img_area, img_volume);
}

void normDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight,
	utils::InitParameter param)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);

	int img_volume = 3 * srcHeight * srcWidth;
	int img_area = srcHeight * srcWidth;
	int img_height = srcHeight;
	int img_width = srcWidth;
	norm_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, dst, batchSize, img_height, img_width, img_area, img_volume,
		param.scale, param.means[0], param.means[1], param.means[2], param.stds[0], param.stds[1], param.stds[2]);
}

void hwc2chwDevice(const int& batchSize, float* src, int srcWidth, int srcHeight,
	float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((dstWidth * dstHeight * 3 + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(batchSize + BLOCK_SIZE - 1) / BLOCK_SIZE);


	int img_volume = 3 * srcHeight * srcWidth;
	int img_area = srcHeight * srcWidth;
	int img_height = srcHeight;
	int img_width = srcWidth;
	hwc2chw_device_kernel << < grid_size, block_size, 0, nullptr >> > (src, dst, batchSize, img_height, img_width, img_area, img_volume);
}

__global__ 
void decode_yolo_device_kernel(int batch_size, int  num_class, int topK, float conf_thresh,
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
	float objectness = pitem[4];
	if (objectness < conf_thresh)
	{
		return;
	}
	float* class_confidence = pitem + 5;
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
	confidence *= objectness;
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
	float left = cx - width * 0.5f;
	float top = cy - height * 0.5f;
	float right = cx + width * 0.5f;
	float bottom = cy + height * 0.5f;
	float* pout_item = dst + dy * dstArea + 1 + index * dstWidth;
	*pout_item++ = left;
	*pout_item++ = top;
	*pout_item++ = right;
	*pout_item++ = bottom;
	*pout_item++ = confidence;
	*pout_item++ = label;
	*pout_item++ = 1;
}

static __device__ 
float box_iou(
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

__global__ 
void nms_fast_kernel(int topK, int batch_size, float iou_thresh,
	float* src, int srcWidth, int srcHeight, int srcArea)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dy >= batch_size)
	{
		return;
	}
	float* p_count = src + dy * srcArea;
	int count = min(int(p_count[0]), topK);
	
	if (dx >= count)
	{
		return;
	}
	float* pcurrent = src + dy * srcArea + 1 + dx * srcWidth;
	for (int i = 0; i < count; ++i) 
	{
		float* pitem = src + dy * srcArea + 1 + i * srcWidth; 
		if (i == dx || pcurrent[5] != pitem[5]) 
			continue;

		if (pitem[4] >= pcurrent[4])
		{
			if (pitem[4] == pcurrent[4] && i < dx) 
				continue;

			float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
				pitem[0], pitem[1], pitem[2], pitem[3]);

			if (iou > iou_thresh)
			{
				pcurrent[6] = 0;
				return;
			}
		}
	}
}

__global__
void get_key_val_kernel(int batchSize, float* src, int srcWidth, int srcHeight, int srcArea,
	int* idx, float* conf)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dy >= batchSize || dx >= srcHeight)
	{
		return;
	}
	int* p_idx_row    = idx  + dy * srcHeight + dx;
	float* p_conf_row = conf + dy * srcHeight + dx;
	p_idx_row[0] = dx;
	float* p_src_row = src + dy * srcArea + 1 + dx * srcWidth; 
	p_conf_row[0] = p_src_row[4];
}

__global__
void nms_sort_kernel(int topK, int batch_size, float iou_thresh,
	float* src, int srcWidth, int srcHeight, int srcArea,
	int* idx)
{
	int dx = blockDim.x * blockIdx.x + threadIdx.x;
	int dy = blockDim.y * blockIdx.y + threadIdx.y;
	if (dy >= batch_size)
	{
		return;
	}
	float* p_count = src + dy * srcArea;
	int count = min(int(p_count[0]), topK);

	if (dx >= count)
	{
		return;
	}
	int* p_idx1 = idx + dy * srcHeight + dx;
	float* pcurrent = src + dy * srcArea + 1 + p_idx1[0] * srcWidth; 
	
	for (int i = (dx + 1); i < count; ++i) 
	{
		int* p_idx2 = idx + dy * srcHeight + i;
		float* pitem = src + dy * srcArea + 1 + p_idx2[0] * srcWidth; 
		
		if (abs(pcurrent[5] - pitem[5]) > 1e-3) 
			continue;
		float iou = box_iou(pcurrent[0], pcurrent[1], pcurrent[2], pcurrent[3],
			pitem[0], pitem[1], pitem[2], pitem[3]);

		if (iou > iou_thresh)
		{
			pitem[6] = 0; 
		}
	}
}

void decodeDevice(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, float* dst, int dstWidth, int dstHeight)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((srcHeight + BLOCK_SIZE - 1) / BLOCK_SIZE,
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	int dstArea = 1 + dstWidth * dstHeight;

	decode_yolo_device_kernel << < grid_size, block_size, 0, nullptr >> > (param.batch_size, param.num_class, param.topK, param.conf_thresh,
		src, srcWidth, srcHeight, srcArea,
		dst, dstWidth, dstHeight, dstArea);
}

void nmsDeviceV1(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((param.topK + BLOCK_SIZE - 1) / BLOCK_SIZE, 
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);

	nms_fast_kernel << < grid_size, block_size, 0, nullptr >> > (param.topK, param.batch_size, param.iou_thresh,
		src, srcWidth, srcHeight, srcArea);
}

void nmsDeviceV2(utils::InitParameter param, float* src, int srcWidth, int srcHeight, int srcArea, 
	int* idx, float* conf)
{
	dim3 block_size(BLOCK_SIZE, BLOCK_SIZE);
	dim3 grid_size((param.topK + BLOCK_SIZE - 1) / BLOCK_SIZE, 
		(param.batch_size + BLOCK_SIZE - 1) / BLOCK_SIZE);
	get_key_val_kernel << < grid_size, block_size, 0, nullptr >> > (param.batch_size, src, srcWidth, srcHeight, srcArea, idx, conf);
	for (size_t i = 0; i < param.batch_size; i++)
	{
		int* p_idx     = idx + i * srcHeight;
		float* p_conf = conf + i * srcHeight;
		thrust::sort_by_key(thrust::device, p_conf, p_conf + srcHeight, p_idx, thrust::greater<float>());
	}
	nms_sort_kernel << < grid_size, block_size, 0, nullptr >> > (param.topK, param.batch_size, param.iou_thresh,
		src, srcWidth, srcHeight, srcArea, idx);
}
