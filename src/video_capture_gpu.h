#ifndef ORANGE_VIDEO_CAPTURE_GPU
#define ORANGE_VIDEO_CAPTURE_GPU

#include "camera.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nppi.h>
#include "NvEncoder/NvEncoderCuda.h"
#include "Utils/NvEncoderCLIOptions.h"
#include "Utils/NvCodecUtils.h"

template <class EncoderClass>
void InitializeEncoder(EncoderClass &pEnc, NvEncoderInitParam encodeCLIOptions, NV_ENC_BUFFER_FORMAT eFormat);
void aquire_frames_gpu_encode(Emergent::CEmergentCamera *camera, Emergent::CEmergentFrame *frame_recv, int num_frames, CameraParams camera_params);


#endif 