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

void aquire_frames_gpu_encode(Emergent::CEmergentCamera *camera, Emergent::CEmergentFrame *frame_recv, CameraParams* camera_params, const char *encoder_str, int* key_num_ptr, PTPParams* ptp_params, string folder_name, unsigned char* d_debayer, bool* encode_flag, bool* capture_pause);
// void aquire_frames_gpu_encode(Emergent::CEmergentCamera *camera, Emergent::CEmergentFrame *frame_recv, CameraParams* camera_params, const char *encoder_str, int* key_num_ptr, PTPParams* ptp_params, string folder_name, unsigned char* display_buffer, bool* encode_flag, bool* capture_pause, unsigned char** cuda_buffer, cudaGraphicsResource_t* cuda_resource, size_t *cuda_pbo_storage_buffer_size, GLuint *pbo, GLuint *texture);

#endif 