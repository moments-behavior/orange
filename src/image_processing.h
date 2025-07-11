// src/image_processing.h
#ifndef ORANGE_IMGAE_PROCESSING
#define ORANGE_IMGAE_PROCESSING

#include "NvEncoder/NvCodecUtils.h"
#include <nppi.h>
#include "video_capture.h"
#include "common.hpp"      // For pose::Object
#include <vector>         // For std::vector
#include <atomic>         // For std::atomic
#include <cuda_runtime.h> // For cudaEvent_t


struct FrameGPU
{
    unsigned char *d_orig;
    int size_pic;
};

struct FrameCPU
{
    unsigned char *frame;
    int size_pic;
};

struct Debayer
{
    unsigned char *d_debayer;
    NppiSize size;
    Npp8u nAlpha;
    NppiRect roi;
    NppiBayerGridPosition grid;
};

struct FrameProcess
{
    FrameGPU frame_original;
    Debayer debayer;
    unsigned char *d_convert;
    FrameCPU frame_cpu;
};

static inline void initialize_cpu_frame(FrameCPU *cpu_buffer, CameraParams *camera_params)
{
    int size_pic = camera_params->width * camera_params->height * 3 * sizeof(unsigned char);
    cpu_buffer->frame = (unsigned char *)malloc(size_pic);
}

static inline void initalize_gpu_frame(FrameGPU *frame_original, CameraParams *camera_params)
{
    frame_original->size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char);
    ck(cudaMalloc((void **)&frame_original->d_orig, frame_original->size_pic));
}

static inline void initialize_gpu_debayer(Debayer *debayer, CameraParams *camera_params)
{
    int output_channels = 4;
    int size_pic = camera_params->width * camera_params->height * sizeof(unsigned char) * output_channels;
    cudaMalloc((void **)&debayer->d_debayer, size_pic);
    cudaMemset(debayer->d_debayer, 0xFF, size_pic);

    debayer->size.width = camera_params->width;
    debayer->size.height = camera_params->height;
    debayer->nAlpha = 255;
    debayer->roi.x = 0;
    debayer->roi.y = 0;
    debayer->roi.width = camera_params->width;
    debayer->roi.height = camera_params->height;
    if (camera_params->need_reorder) {
        // 100G camera
        debayer->grid = NPPI_BAYER_GRBG;
    } else if (camera_params->pixel_format.compare("BayerRG8")==0) {
        debayer->grid = NPPI_BAYER_RGGB;
    } else {
        debayer->grid = NPPI_BAYER_GBRG;
    }
}

static inline void debayer_frame_gpu(CameraParams *camera_params, FrameGPU *frame_original, Debayer *debayer)
{
    const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(frame_original->d_orig,
                                                         camera_params->width * sizeof(unsigned char),
                                                         debayer->size,
                                                         debayer->roi,
                                                         debayer->d_debayer,
                                                         camera_params->width * sizeof(uchar4),
                                                         debayer->grid,
                                                         NPPI_INTER_UNDEFINED,
                                                         debayer->nAlpha);
    if (npp_result != 0)
    {
        std::cout << "\nNPP error %d \n"
                  << npp_result << std::endl;
    }
}

static inline void duplicate_channel_gpu(CameraParams *camera_params, FrameGPU *frame_original, Debayer *debayer)
{
    const NppStatus npp_result = nppiDup_8u_C1AC4R(
        frame_original->d_orig,
        camera_params->width * sizeof(unsigned char),
        debayer->d_debayer,
        camera_params->width * sizeof(uchar4),
        debayer->size);

    if (npp_result != 0)
    {
        std::cout << "\nNPP error %d \n"
                  << npp_result << std::endl;
    }
}


#endif