#ifndef ORANGE_BUFFERUTILS
#define ORANGE_BUFFERUTILS

#include <cuda.h>
#include "NvCodecUtils.h"
#include <opencv2/opencv.hpp>            

struct PictureBuffer{
    unsigned char* frame;
    int frame_number;
    bool available_to_write;
};


void GetImage(CUdeviceptr dpSrc, uint8_t* pDst, int nWidth, int nHeight);
void clear_buffer_with_constant_image(unsigned char* image_pt, int width, int height);
void print_one_display_buffer(unsigned char* image_pt, int width, int height, int channels);

#endif