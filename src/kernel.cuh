// src/kernel.cuh
#ifndef KERNEL_H
#define KERNEL_H
#include <cuda.h>
#include <cuda_runtime_api.h>
#include "types.h"
#include <chrono>
#include <iostream>
#include "common.hpp" // For pose::Object

void GSPRINT4521_Convert(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift);
void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
void rgba2bgr_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);
void gpu_draw_cicles(unsigned char* src, int width, int height, float* d_points, int num_points, cudaStream_t stream);
void gpu_draw_box(unsigned char* src, int width, int height, const pose::Object* d_detections, int num_objects, cudaStream_t stream);
void gpu_draw_rat_pose(unsigned char* src, int width, int height, float* d_points, unsigned int* d_skeleton, cudaStream_t stream, int num_channels);

void gpu_crop_and_resize(
    const unsigned char* d_src,
    unsigned char* d_dst_bgr,
    int src_width, int src_height,
    pose::Rect detection_rect,
    int dst_width, int dst_height,
    cudaStream_t stream
);

void launch_mono_to_rgb_kernel(unsigned char* dst_rgb, const unsigned char* src_mono, int width, int height, cudaStream_t stream);

#endif // KERNEL_H