
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"
#include <iostream>
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nppi.h>
#include "thread.h"

int main()
{

    const char *my_picture = "camera_test.bmp";
    int width = 3208;
    int height = 2200;
    int channel = 1;
    unsigned char *picture_memory = stbi_load(my_picture, &width, &height, &channel, 1);
    channel = 1;

    unsigned char *d_orig;
    int size_pic = width * height * channel * sizeof(unsigned char);
    cudaMalloc((void **)&d_orig, size_pic);

    cudaError_t result = cudaMemcpy(d_orig, picture_memory, size_pic, cudaMemcpyHostToDevice);
     
    if (result != cudaSuccess)
    {
        printf("Cuda Error");
    }

    unsigned char *d_debayer;
    cudaMalloc((void **)&d_debayer, size_pic * 3);

    // debayer
    NppiSize size;
    size.width = width;
    size.height = height;

    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = width;
    roi.height = height;

    NppiBayerGridPosition grid;
    grid = NPPI_BAYER_RGGB;

    float start_time = tick();
    const NppStatus npp_result = nppiCFAToRGB_8u_C1C3R(d_orig,
                                                       width * sizeof(unsigned char),
                                                       size,
                                                       roi,
                                                       d_debayer,
                                                       width * sizeof(uchar3),
                                                       grid,
                                                       NPPI_INTER_UNDEFINED);
    
    // not sure if it is just lighting fast for one frame, or ..., how to time it, write a clocl macro? 
    cudaDeviceSynchronize();
    float end_time = tick();
    float time_diff = end_time - start_time;
    
    
    printf("time for debayer (us): %.6f", time_diff * 1.0e6);

    if (npp_result != 0)
    {
        printf("\nNPP error %d \n", npp_result);
    }

    unsigned char *host_back = (unsigned char *)malloc(size_pic * 3);
    result = cudaMemcpy(host_back, d_debayer, size_pic*3, cudaMemcpyDeviceToHost);
    if (result != cudaSuccess)
    {
        printf("Cuda Error");
    }

    stbi_write_bmp("picture_memory.bmp", width, height, 3, host_back);
}