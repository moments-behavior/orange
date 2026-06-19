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
void gpu_draw_box(unsigned char* src, int width, int height, float* d_points, int label_id, cudaStream_t stream);
void gpu_draw_rat_pose(unsigned char* src, int width, int height, float* d_points, unsigned int* d_skeleton, cudaStream_t stream, int num_channels);

// Sum a mono8 image over a strided grid (every `stride`-th pixel in x and y)
// into *d_sum (device scalar; zeroed by this call) on `stream`. Used for a cheap
// average-brightness estimate that stays off the encoder's default stream. Row
// pitch is assumed to equal `width` (tightly packed, as frame_original.d_orig
// is). Sample count = ceil(width/stride) * ceil(height/stride).
void launch_brightness_sum(const unsigned char* d_img, int width, int height,
                           int stride, unsigned long long* d_sum, cudaStream_t stream);
#endif // KERNEL_H
