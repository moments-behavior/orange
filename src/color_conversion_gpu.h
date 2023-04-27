#ifndef ORANGE_COLORCONVERSIONGPU
#define ORANGE_COLORCONVERSIONGPU

#include <iostream>
#include <cuda_runtime.h>

void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
void rgba2bgr_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);

#endif