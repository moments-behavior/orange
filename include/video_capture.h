#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE

#include <cstdint>
#include <string>
#include <iostream>
#include <fstream>
#include "network_base.h"
#include "camera_params.h"
#include "ptp_manager.h"

// Forward declarations
struct CameraControl;
struct CameraEachSelect;
struct CameraParams;
struct CameraEmergent;

struct PTPParams {
    unsigned long long ptp_global_time{0}; 
    unsigned long long ptp_stop_time{0};
    uint64_t ptp_counter{0};
    uint64_t ptp_stop_counter{0};
    bool network_sync{false};
    bool ptp_start_reached{false};
    bool ptp_stop_reached{false};
    bool network_set_stop_ptp{false};
    bool network_set_start_ptp{false};
};

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

// Function declarations
void report_statistics(CameraParams *camera_params, CameraState *camera_state, double time_diff);
void show_ptp_offset(evt::PTPState *ptp_state, CameraEmergent *ecam);
void start_ptp_sync(evt::PTPState *ptp_state, PTPParams *ptp_params, CameraParams *camera_params, CameraEmergent *ecam, unsigned int delay_in_second);
void grab_frames_after_countdown(evt::PTPState *ptp_state, CameraEmergent *ecam);

#endif // ORANGE_VIDEO_CAPTURE