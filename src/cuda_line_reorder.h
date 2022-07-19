#pragma once
#include <iostream>
#include <cuda_runtime.h>

void GSPRINT4521_Convert(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift);
