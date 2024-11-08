#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE
#include "thread.h"
#include "camera.h"
#include <iostream>
#include <fstream>
#include "network_base.h"
#include "camera_params.h"

enum PictureSaveState {
    State_Frame_Idle = 0, 
    State_Write_New_Frame = 1
};

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
    bool yolo = false;
    PictureSaveState frame_save_state = State_Frame_Idle;
    int frame_save_idx = 0; 
    bool selected_to_save = false;
    const char* picture_save_folder;
    const char* yolo_model;
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

void report_statistics(CameraParams *camera_params, CameraState *CameraState, double time_diff);
void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam);
void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params, CameraParams *camera_params, CameraEmergent *ecam, unsigned int delay_in_second);
void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam);
#endif