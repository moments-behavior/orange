#include "color_conversion_gpu.h"

__global__ void rgba2rgb_kernel(unsigned char* dest, unsigned char* src, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x < width) && (y < height)) {
        *(dest + ((y * width * 3) + (x * 3))) = *(src + ((y * width * 4) + (x * 4)));            
        *(dest + ((y * width * 3) + (x * 3)) + 1) = *(src + ((y * width * 4) + (x * 4)) + 1);
        *(dest + ((y * width * 3) + (x * 3)) + 2) = *(src + ((y * width * 4) + (x * 4)) + 2);
    }
}



void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    rgba2rgb_kernel << <num_blocks, threads_per_block, 0, stream>> > (dest, src, width, height);
}


__global__ void rgba2bgr_kernel(unsigned char* dest, unsigned char* src, int width, int height)
{
    int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if ((x < width) && (y < height)) {
        *(dest + ((y * width * 3) + (x * 3))) = *(src + ((y * width * 4) + (x * 4)) + 2);            
        *(dest + ((y * width * 3) + (x * 3)) + 1) = *(src + ((y * width * 4) + (x * 4)) + 1);
        *(dest + ((y * width * 3) + (x * 3)) + 2) = *(src + ((y * width * 4) + (x * 4)));
    }
}



void rgba2bgr_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream)
{
    dim3 threads_per_block(32, 32);
    dim3 num_blocks((width + threads_per_block.x -1) / threads_per_block.x, (height + threads_per_block.y -1) / threads_per_block.y);
    rgba2bgr_kernel << <num_blocks, threads_per_block, 0, stream>> > (dest, src, width, height);
}