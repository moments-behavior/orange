#include <iostream>
#include <cuda_runtime.h>

/* Sensor process data:
    *Every group of 16 lines, change order to (0, 4, 8, 12), (1, 5, 9, 13), (2, 6, 10, 14), (3, 7, 11, 15).
    *Then lines interleave:
    *0                        N
    *1                        0
    *2  top/bot interleave    N - 1
    *3  ==================>   1
    *                         ....
    *N
    *
**/
__global__ void GSPRINT4521_ConvertKernel(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift)
{
    //int x = threadIdx.x;  // 1
    //printf("threadIdx: x: %d y: %d\n", threadIdx.x, threadIdx.y);
    int row = blockIdx.x;  // 0 - (heigh - 1)
    const unsigned char* ptrS = src + row * strideS;

    bool isBot = !(row & 0x01); // line comes bot, top, bot, top
    int rowInHalf = row >> 1;  // top / bot interleave
    int rowBase16 = rowInHalf & 0xFFF0; // base row of group-16-line, 0, 16, 32, 48, ..., same for top and bottom

    int idGroup4 = (rowInHalf & 0xC) >> 2;  // 0 - 3 of group id in a group-16-line, start of each group-4-col
    int idInGroup4 = rowInHalf & 0x03;  // 0 - 3 in a group-4-col
    int lineInGroup16 = idGroup4 + (idInGroup4 << 2);  // idInGroup4 << 2 is idInGroup4 * 4

    int lineD = lineInGroup16 + rowBase16;

    if (isBot) {
        lineD = height - 1 - lineD;
    }
    unsigned char* ptrD = dest + lineD * strideD;

    if(leftShift == 0) { // not leftshift, just memcpy
        memcpy(ptrD, ptrS, strideS);
    } else {  // need left shift for 16 bit data
        unsigned short* ptrS16 = (unsigned short*)ptrS;
        unsigned short* ptrD16 = (unsigned short*)ptrD;
        for (int i = 0; i < width; i++) {
            unsigned short s = *ptrS16;
            *ptrD16 = (s << leftShift);
            ptrD16++;
            ptrS16++;
        }
    }
}

void GSPRINT4521_Convert(unsigned char* dest, const unsigned char* src, int width, int height, int strideS, int strideD, int leftShift)
{
    dim3 threadPerBlock(1, 1);  // 1 thread per line    
//height = <blocksPerGrid>, 1 dimension, blockIdx.x == blockIdx.y == 0 ... (height - 1)
//threadPerBlock(1,2) = 2 dimensions, threadIdx.x == 0, threadIdx.y = 0 ... 1
    GSPRINT4521_ConvertKernel << <height, threadPerBlock >> > (dest, src, width, height, strideS, strideD, leftShift);  // this opens height blocks, 2 thread per block
    cudaError_t err = cudaDeviceSynchronize();
    if(err != cudaSuccess) printf("GSPRINT4521_ConvertKernel failed: %s\n", cudaGetErrorString(err));
}