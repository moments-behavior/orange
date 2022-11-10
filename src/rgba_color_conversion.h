#pragma once
#include <iostream>
#include <cuda_runtime.h>

void rgba2rgb_convert(unsigned char* dest, unsigned char* src, int width, int height, cudaStream_t stream);