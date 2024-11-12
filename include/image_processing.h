#ifndef ORANGE_IMAGE_PROCESSING
#define ORANGE_IMAGE_PROCESSING

#include "camera_params.h"
#include "NvEncoder/NvCodecUtils.h"
#include <nppi.h>
#include <cuda_runtime_api.h>
#include <iostream>

namespace evt {

// Define structs for frames
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

typedef struct {
	void* imagePtr; // source image buffer
	size_t bufferSize; // size of imagePtr in bytes
	int width;
	int height;
	int pixelFormat;
    unsigned long long timestamp;
    unsigned long long frame_id;
    uint64_t timestamp_sys;
} WORKER_ENTRY;

// CPU Frame Initialization
static inline void initializeCPUFrame(FrameCPU* cpu_buffer, const CameraParams* camera_params) {
    int size_pic = camera_params->width * camera_params->height * 3 * sizeof(unsigned char);
    cpu_buffer->frame = (unsigned char*)malloc(size_pic);
}

// GPU Frame Initialization
static inline void initializeGPUFrame(FrameGPU* frame_original, const CameraParams* camera_params) {
    frame_original->size_pic = camera_params->width * camera_params->height * 1 * sizeof(unsigned char);
    ck(cudaMalloc((void**)&frame_original->d_orig, frame_original->size_pic));
}

// Debayer Initialization for GPU
static inline void initializeGPUDebayer(Debayer* debayer, const CameraParams* camera_params) {
    int output_channels = 4;
    int size_pic = camera_params->width * camera_params->height * sizeof(unsigned char) * output_channels;
    cudaMalloc((void**)&debayer->d_debayer, size_pic);
    cudaMemset(debayer->d_debayer, 0xFF, size_pic);

    debayer->size.width = camera_params->width;
    debayer->size.height = camera_params->height;
    debayer->nAlpha = 255;
    debayer->roi.x = 0;
    debayer->roi.y = 0;
    debayer->roi.width = camera_params->width;
    debayer->roi.height = camera_params->height;
    if (camera_params->need_reorder) {
        debayer->grid = NPPI_BAYER_GRBG;
    } else if (camera_params->pixel_format.compare("BayerRG8") == 0) {
        debayer->grid = NPPI_BAYER_RGGB;
    } else {
        debayer->grid = NPPI_BAYER_GBRG;
    }
}

// Debayering GPU Frame
static inline void debayerFrameGPU(const CameraParams* camera_params, FrameGPU* frame_original, Debayer* debayer) {
    const NppStatus npp_result = nppiCFAToRGBA_8u_C1AC4R(frame_original->d_orig,
                                                         camera_params->width * sizeof(unsigned char),
                                                         debayer->size,
                                                         debayer->roi,
                                                         debayer->d_debayer,
                                                         camera_params->width * sizeof(uchar4),
                                                         debayer->grid,
                                                         NPPI_INTER_UNDEFINED,
                                                         debayer->nAlpha);
    if (npp_result != 0) {
        std::cerr << "NPP error " << npp_result << std::endl;
    }
}

// Duplicate Channel for Monochrome Images (GPU)
static inline void duplicateChannelGPU(const CameraParams* camera_params, FrameGPU* frame_original, Debayer* debayer) {
    const NppStatus npp_result = nppiDup_8u_C1AC4R(
        frame_original->d_orig,
        camera_params->width * sizeof(unsigned char),
        debayer->d_debayer,
        camera_params->width * sizeof(uchar4),
        debayer->size);

    if (npp_result != 0) {
        std::cerr << "NPP error " << npp_result << std::endl;
    }
}

} // namespace evt

#endif // ORANGE_IMAGE_PROCESSING
