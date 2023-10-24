#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE
#include "thread.h"
#include "camera.h"
#include <iostream>
#include <fstream>
#include "SyncDisplay.h"

struct CameraControl
{
    bool open = false;
    bool subscribe = false;
    bool stop_record = false;
    bool record_video = false;
    bool sync_camera = false;
};

struct CameraEachSelect
{
    bool stream_on = true;
};

struct CameraState
{
    int camera_return = 0;
    unsigned short id_prev = 0;
    unsigned short dropped_frames = 0;
    unsigned int frames_recd = 0;
    unsigned long long frame_count = 0;
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

void aquire_frames(CameraEmergent *ecam, CameraParams *camera_params, CameraEachSelect* camera_select, CameraControl* camera_control, unsigned char *display_buffer, std::string encoder_setup, std::string folder_name, PTPParams* ptp_params, SyncDisplay* sync_manager);
#endif