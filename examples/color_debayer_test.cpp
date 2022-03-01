#include <cuda.h>
#include <iostream>
#include <nppi.h>
#include <cuda_runtime.h>
#include <iostream>
#include "helper_cuda.h"
#include "../UtilNPP/ImagesNPP.h"
#include "../UtilNPP/ImagesCPU.h"
#include "../UtilNPP/ImageIO.h"
#define STB_IMAGE_WRITE_IMPLEMENTATION
#include "stb_image_write.h"


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

bool printfNPPinfo(int argc, char *argv[])
{
    // const NppLibraryVersion *libVer   = nppGetLibVersion();
    // printf("NPP Library Version %d.%d.%d\n", libVer->major, libVer->minor, libVer->build);

    int driverVersion, runtimeVersion;
    cudaDriverGetVersion(&driverVersion);
    cudaRuntimeGetVersion(&runtimeVersion);

    printf("  CUDA Driver  Version: %d.%d\n", driverVersion / 1000, (driverVersion % 100) / 10);
    printf("  CUDA Runtime Version: %d.%d\n", runtimeVersion / 1000, (runtimeVersion % 100) / 10);

    // Min spec is SM 1.0 devices
    bool bVal = checkCudaCapabilities(1, 0);
    return bVal;
}

cudaError_t cuda_bayer_to_rgba(uint8_t *input, uint8_t *output, size_t width, size_t height)
{
    NppiSize size;
    size.width = width;
    size.height = height;

    NppiRect roi;
    roi.x = 0;
    roi.y = 0;
    roi.width = width-1;
    roi.height = height-1;

    NppiBayerGridPosition grid;
    grid = NPPI_BAYER_GRBG; //NPPI_BAYER_RGGB;
    
    Npp8u nAlpha = 255;

    printf("\nsize of uchar4 %ld", sizeof(uchar4));
    printf("\nsize of uint8_t %ld", sizeof(uint8_t));
    const NppStatus result = nppiCFAToRGBA_8u_C1AC4R(input, width * sizeof(uint8_t), size, roi,
                                                     output, width * sizeof(uchar4),
                                                     grid, NPPI_INTER_UNDEFINED, nAlpha);

    if (result != 0)
    {
        printf("\ncudaBayerToRGB() NPP error %d \n", result);
        return cudaErrorUnknown;
    }
    return cudaSuccess;
}

int main(int argc, char *argv[])
{

    // find a bayer image input, and test this function
    printf("%s Starting...\n\n", argv[0]);

    try
    {

        cudaDeviceInit(argc, (const char **)argv);

        std::string sFilename;
        char *filePath {"camera_test.bmp"};

        sFilename = filePath;
        // if we specify the filename at the command line, then we only test sFilename[0].
        int file_errors = 0;
        std::ifstream infile(sFilename.data(), std::ifstream::in);

        if (infile.good())
        {
            std::cout << "boxFilterNPP opened: <" << sFilename.data() << "> successfully!" << std::endl;
            file_errors = 0;
            infile.close();
        }
        else
        {
            std::cout << "boxFilterNPP unable to open: <" << sFilename.data() << ">" << std::endl;
            file_errors++;
            infile.close();
        }

        if (file_errors > 0)
        {
            exit(EXIT_FAILURE);
        }

        // declare a host image object for an 8-bit grayscale image
        npp::ImageCPU_8u_C1 oHostSrc;
        // load gray-scale image from disk
        npp::loadImage(sFilename, oHostSrc);

        // loading works
        stbi_write_bmp("test_loading.bmp", 3208, 2200, 1, oHostSrc.data());

        // declare a device image and copy construct from the host image,
        // i.e. upload host to device
        npp::ImageNPP_8u_C1 oDeviceSrc(oHostSrc);

        NppiSize oSrcSize = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};
        // create struct with ROI size
        NppiSize oSizeROI = {(int)oDeviceSrc.width(), (int)oDeviceSrc.height()};

        // allocate device image of appropriately reduced size
        npp::ImageNPP_8u_C4 oDeviceDst(oSizeROI.width, oSizeROI.height);

        //printf("\nprint value: %s", oDeviceSrc->data()(0, 0));
        cuda_bayer_to_rgba((uint8_t *) oDeviceSrc.data(), (uint8_t *) oDeviceDst.data(), oSizeROI.width, oSizeROI.height);

        // declare a host image for the result
        npp::ImageCPU_8u_C4 oHostDst(oDeviceDst.size() );
        // and copy the device result data into it
        printf("pitch: %u\n", oHostDst.pitch());
        oDeviceDst.copyTo(oHostDst.data(), oHostDst.pitch());
        // for(int i = 0; i < oHostDst.height(); i++) {
        //     for (int j = 0; j < oHostDst.width(); j++) {
        //         printf("%u ", *(oHostDst.data(j, i)));
        //         printf("%u ", *(oHostDst.data(j, i) + (1 * sizeof(Npp8u))));
        //         printf("%u, ", *(oHostDst.data(j, i) + (2 * sizeof(Npp8u))));
        //         printf("%u, ", *(oHostDst.data(j, i) + (3 * sizeof(Npp8u))));
        //     } 
        //     printf("\n");
        // }
        //printf("\ntry to print this %s", (char*) oHostDst.data());
        // printf("\nPitch oHostSrc: %d", oHostSrc.pitch());
        // printf("\nPitch oHostDst: %d\n", oHostDst.pitch());

        // Npp8u *color_test = (Npp8u *)calloc(10 * 10 * 4,  sizeof(Npp8u));
        // *color_test = 255;
        // *(color_test + (3 * sizeof(Npp8u))) = 255;
        // stbi_write_bmp("color_test.bmp", 10, 10, 4, (void *) color_test);
        stbi_write_bmp("test.bmp", (int)oHostDst.width(), (int)oHostDst.height(), 4, (void *) oHostDst.data());
        nppiFree(oDeviceSrc.data());
        nppiFree(oDeviceDst.data());

        exit(EXIT_SUCCESS);
    }
    catch (npp::Exception &rException)
    {
        std::cerr << "Program error! The following exception occurred: \n";
        std::cerr << rException << std::endl;
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
    }
    catch (...)
    {
        std::cerr << "Program error! An unknow type of exception occurred. \n";
        std::cerr << "Aborting." << std::endl;

        exit(EXIT_FAILURE);
        return -1;
    }

    return 0;
}
