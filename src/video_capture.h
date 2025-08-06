#ifndef ORANGE_VIDEO_CAPTURE
#define ORANGE_VIDEO_CAPTURE
#include "camera.h"
#include "network_base.h"
#include "utils.h"
#include <atomic>

enum PictureState {
    State_Frame_Idle,
    State_Copy_New_Frame,
    State_Frame_Copy_Done,
    State_Frame_Detection_Ready
};

struct CameraControl {
    bool open = false;
    bool subscribe = false;
    bool stop_record = false;
    bool record_video = false;
    bool sync_camera = false;
    bool trigger_mode = false;
};

enum DetectMode {
    Detect_OFF,
    Detect2D_GLThread,
    Detect2D_Standoff,
    Detect3D_Standoff
};
constexpr const char *DetectModeNames[] = {"OFF", "2DGLThread", "2DStandoff",
                                           "3DStandoff"};
struct CameraEachSelect {
    bool stream_on = true;
    bool record = true;
    int downsample = 1;
    std::atomic<PictureState> frame_save_state;
    std::string frame_save_format;
    std::string frame_save_name;
    int pictures_counter = 0;
    bool selected_to_save = false;
    std::string picture_save_folder;
    std::string yolo_model;
    DetectMode detect_mode = Detect_OFF;
    int idx2d = 0;
    int idx3d = 0;
    int total_standoff_detector = 0;
    std::atomic<PictureState> frame_detect_state;
    FPSEstimator encoder_fps_estimator;
    FPSEstimator capture_fps_estimator;
    int dropped_frames = 0;
    CameraEachSelect()
        : frame_save_state(State_Frame_Idle),
          frame_detect_state(State_Frame_Idle) {}
};

struct CameraState {
    int camera_return = 0;
    unsigned short id_prev = 0;
    unsigned short dropped_frames = 0;
    unsigned int frames_recd = 0;
    unsigned long long frame_count = 0;
};

struct PTPState {
    int ptp_offset;
    int ptp_offset_sum = 0;
    int ptp_offset_prev = 0;
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

void report_statistics(CameraParams *camera_params, CameraState *camera_state,
                       double time_diff);
void show_ptp_offset(PTPState *ptp_state, CameraEmergent *ecam);
void start_ptp_sync(PTPState *ptp_state, PTPParams *ptp_params,
                    CameraParams *camera_params, CameraEmergent *ecam,
                    unsigned int delay_in_second);
void grab_frames_after_countdown(PTPState *ptp_state, CameraEmergent *ecam);
bool try_start_timer();
bool try_stop_timer();
void acquire_frames(CameraEmergent *ecam, CameraParams *camera_params,
                    CameraEachSelect *camera_select,
                    CameraControl *camera_control,
                    unsigned char *display_buffer, std::string encoder_setup,
                    std::string folder_name, PTPParams *ptp_params);
#endif
