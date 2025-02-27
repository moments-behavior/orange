#ifndef KERNEL_H
#define KERNEL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "types.h"
#include <chrono>
#include <iostream>

void GSPRINT4521_Convert(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift);
void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
void rgba2bgr_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
void gpu_draw_cicles(unsigned char* src, int width, int height, float* d_points, int num_points, cudaStream_t stream);
void gpu_draw_box(unsigned char* src, int width, int height, float* d_points, cudaStream_t stream);
void gpu_draw_rat_pose(unsigned char* src, int width, int height, float* d_points, unsigned int* d_skeleton, cudaStream_t stream);
void gpu_draw_ring(unsigned char* src, int width, int height, float* d_points, float scale, cudaStream_t stream);
#endif // KERNEL_H