#ifndef __CUDA_BAYER_H__
#define __CUDA_BAYER_H__
#include <cuda.h>
#include <iostream>
#include <nppi.h>
#include <cuda_runtime.h>
cudaError_t cudaBayerToRGBA( uint8_t* input, uchar3* output, size_t width, size_t height);
#endif