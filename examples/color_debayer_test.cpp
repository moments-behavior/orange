#include <cuda.h>
#include <iostream>
#include <nppi.h>
#include <cuda_runtime.h>
#include <iostream>
#include "helper_cuda.h"

inline int cudaDeviceInit(int argc, const char **argv)
{
    int deviceCount;
    checkCudaErrors(cudaGetDeviceCount(&deviceCount));

    if (deviceCount == 0)
    {
        std::cerr << "CUDA error: no devices supporting CUDA." << std::endl;
        exit(EXIT_FAILURE);
    }

    int dev = findCudaDevice(argc, argv);

    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, dev);
    std::cerr << "cudaSetDevice GPU" << dev << " = " << deviceProp.name << std::endl;

    checkCudaErrors(cudaSetDevice(dev));

    return dev;
}

cudaError_t cuda_bayer_to_rgba( uint8_t* input, uchar3* output, size_t width, size_t height)
{
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
    Npp8u nAlpha = 1;

    const NppStatus result = nppiCFAToRGBA_8u_C1AC4R(input, width * sizeof(uint8_t), size, roi, 
                                                   (uint8_t*)output, width * sizeof(uchar3),
                                                   grid, NPPI_INTER_UNDEFINED, nAlpha);

    if( result != 0 )
    {
        printf("cudaBayerToRGB() NPP error %\n", result);
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

int main(int argc, char *argv[]){

    // find a bayer image input, and test this function
    printf("%s Starting...\n\n", argv[0]);
    std::string sFilename;
    char *filePath {"frame_069.raw"};

    cudaDeviceInit(argc, (const char **)argv);


}