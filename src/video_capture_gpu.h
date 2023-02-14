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
#include "FFmpegWriter.h"
#include <fstream>

struct CameraControl
{
    bool streaming = false;
    bool record_video = false;
    bool pause_recording = false;
};

struct CameraState
{
    int camera_return = 0;
    unsigned short id_prev = 0;
    unsigned short dropped_frames = 0;
    unsigned int frames_recd = 0;
    int frame_count = 0;
};

struct Debayer
{
    unsigned char *d_debayer;
    NppiSize size;
    Npp8u nAlpha;
    NppiRect roi;
    NppiBayerGridPosition grid;
};

struct FrameGPU
{
    unsigned char *d_orig;
    int size_pic;
};

struct Writer
{
    string video_file;
    string metadata_file;
    FFmpegWriter *video;
    ofstream* metadata;
};

struct EncoderContext
{
    NV_ENC_BUFFER_FORMAT eFormat;
    NvEncoderInitParam encodeCLIOptions;
    CUcontext cuContext;
    int num_frame_encode;
    std::vector<std::vector<uint8_t>> vPacket;
    NvEncoderCuda *pEnc;
};


struct PTPState 
{
    int ptp_offset;
    int ptp_offset_sum=0;
    int ptp_offset_prev=0;
    unsigned int ptp_time_low;
    unsigned int ptp_time_high;
    unsigned int ptp_time_plus_delta_to_start_low;
    unsigned int ptp_time_plus_delta_to_start_high;
    unsigned long long ptp_time_delta_sum = 0;
    unsigned long long ptp_time_delta;
    unsigned long long ptp_time;
    unsigned long long ptp_time_prev;
    unsigned long long ptp_time_countdown;
    unsigned long long frame_ts; 
    unsigned long long frame_ts_prev;
    unsigned long long frame_ts_delta;
    unsigned long long frame_ts_delta_sum = 0;
    unsigned long long ptp_time_plus_delta_to_start;
    char ptp_status[100];
    unsigned long ptp_status_sz_ret;
    unsigned int ptp_time_plus_delta_to_start_uint;
};

void syc_aquisition(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_control, PTPParams* ptp_params);
void sync_aquire_frames_gpu_encode(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_control, unsigned char *display_buffer, string encoder_setup, string folder_name, PTPParams* ptp_params);
void aquire_frames_gpu(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_state, unsigned char *display_buffer);
void headless_slave_aquire_frames_gpu_encode(CameraEmergent *ecam, CameraParams *camera_params, CameraControl *camera_control, string encoder_setup, string folder_name, PTPParams* ptp_params);
#endif