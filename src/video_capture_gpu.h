#ifndef ORANGE_VIDEO_CAPTURE_GPU
#define ORANGE_VIDEO_CAPTURE_GPU
#include "NvEncoder/NvEncoderCuda.h"
#include "NvEncoder/NvEncoderCLIOptions.h"
#include "NvEncoder/NvCodecUtils.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <nppi.h>
#include "thread.h"
#include "camera.h"
#include <iostream>

void aquire_frames_gpu_encode(Emergent::CEmergentCamera *camera, Emergent::CEmergentFrame *frame_recv, int num_frames, CameraParams camera_params);

#endif 