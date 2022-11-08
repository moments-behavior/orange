#include "buffer_utils.h"

void GetImage(CUdeviceptr dpSrc, uint8_t* pDst, int nWidth, int nHeight)
{
    CUDA_MEMCPY2D m = { 0 };
    m.WidthInBytes = nWidth;
    m.Height = nHeight;
    m.srcMemoryType = CU_MEMORYTYPE_DEVICE;
    m.srcDevice = (CUdeviceptr)dpSrc;
    m.srcPitch = m.WidthInBytes;
    m.dstMemoryType = CU_MEMORYTYPE_HOST;
    m.dstDevice = (CUdeviceptr)(m.dstHost = pDst);
    m.dstPitch = m.WidthInBytes;
    cuMemcpy2D(&m);
}

void clear_buffer_with_constant_image(unsigned char* image_pt, int width, int height) {
    int counter = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            *(image_pt + counter) = 45;
            *(image_pt + counter + 1) = 85;
            *(image_pt + counter + 2) = 255;
            *(image_pt + counter + 3) = 255;
            counter += 4;
        }
    }
}

void print_one_display_buffer(unsigned char* image_pt, int width, int height, int channels) {
    int counter = 0;
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            for (int k = 0; k < channels; k++) {
                printf("%x ", *(image_pt + counter));
                counter++;
            }
            printf("  ");
        }
        printf("\n");
    }
}