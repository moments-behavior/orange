#ifndef KERNEL_H
#define KERNEL_H
#include <cuda.h>
#include <cuda_runtime_api.h>

void GSPRINT4521_Convert(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift);
void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
void rgba2bgr_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
#endif // KERNEL_H
