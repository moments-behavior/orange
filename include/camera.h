#ifndef ORANGE_CAMERA
#define ORANGE_CAMERA

#ifndef  EMERGENT_SDK
#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <EvtParamAttribute.h>
#include <gigevisiondeviceinfo.h>
#endif

#include <emergentcameradef.h>
#include <emergentgigevisiondef.h>
#include <EvtParamAttribute.h>
#include <unistd.h>
#include <string>
#include <algorithm>
#include <vector>
#include <numeric>

std::string get_evt_error_string(EVT_ERROR error);

#define check_camera_errors(err, camera_serial) __check_camera_errors(err, camera_serial, __FILE__, __LINE__)

inline void __check_camera_errors(EVT_ERROR err, const char *camera_serial, const char *file, const int line) {
  if (EVT_SUCCESS != err) {
    std::string error_string;
    error_string = get_evt_error_string(err);
    const char*  errorStr = error_string.c_str();
    fprintf(stderr,
            "%s checkCameraErrors() Driver API error = %04d \"%s\" from file <%s>, "
            "line %i.\n",
            camera_serial, err, errorStr, file, line);
    throw(EXIT_FAILURE);
  }
}


struct CameraEmergent{
    Emergent::CEmergentCamera camera;
    Emergent::CEmergentFrame* evt_frame;
    Emergent::CEmergentFrame frame_recv;
    Emergent::CEmergentFrame frame_reorder;
};

struct PTPParams{
    unsigned long long ptp_global_time; 
    unsigned long long ptp_stop_time;
    uint64_t ptp_counter;
    uint64_t ptp_stop_counter;
    bool network_sync = false;
    bool ptp_start_reached = false;
    bool ptp_stop_reached = false;
    bool network_set_stop_ptp = false;
    bool network_set_start_ptp = false;
};

// void print_camera_device_struct(GigEVisionDeviceInfo* device_info, int camera_idx);
// void configure_factory_defaults(Emergent::CEmergentCamera* camera, CameraParams *camera_params);
// void close_camera(Emergent::CEmergentCamera* camera, CameraParams *camera_params);
// void open_camera_with_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams* camera_params);
// void allocate_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, CameraParams* camera_params, int buffer_size);
// void set_frame_buffer(Emergent::CEmergentFrame* evt_frame, CameraParams* camera_params);
// void destroy_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, int buffer_size, CameraParams *camera_params);
// void ptp_camera_sync(Emergent::CEmergentCamera* camera, CameraParams *camera_params);
// void ptp_sync_off(Emergent::CEmergentCamera *camera, CameraParams *camera_params);
// void quick_print_camera(GigEVisionDeviceInfo* device_info, int camera_idx);
// void print_camera_device_struct(GigEVisionDeviceInfo* device_info, int camera_idx);
// unsigned long long get_current_PTP_time(Emergent::CEmergentCamera* camera);
// void test_gpo_manual_toggle(Emergent::CEmergentCamera* camera);
// void change_camera_ip_persistent(GigEVisionDeviceInfo* device_info, Emergent::CEmergentCamera* camera, const char* new_ip, CameraParams *camera_params);
// void update_gain_value(Emergent::CEmergentCamera* camera, int gain_val, CameraParams* camera_params);
// void update_exposure_value(Emergent::CEmergentCamera* camera, int exposure_val, CameraParams* camera_params);
// void update_exposure_framerate_value(Emergent::CEmergentCamera *camera, int exposure_val, int* frame_rate_val, CameraParams *camera_params);
// void update_frame_rate_value(Emergent::CEmergentCamera* camera, int frame_rate_val, CameraParams* camera_params);
// void update_width_value(Emergent::CEmergentCamera* camera, int width_val, CameraParams* camera_params);
// void update_height_value(Emergent::CEmergentCamera* camera, int height_val, CameraParams* camera_params);
// void update_offsetX_value(Emergent::CEmergentCamera* camera, int OFFSET_X_VAL, CameraParams* camera_params);
// void update_offsetY_value(Emergent::CEmergentCamera* camera, int OFFSET_Y_VAL, CameraParams* camera_params);
// void update_focus_value(Emergent::CEmergentCamera* camera, int focus_value, CameraParams* camera_params);
// int scan_cameras(int max_cameras, GigEVisionDeviceInfo *device_info);
// void allocate_frame_reorder_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_reorder, CameraParams* camera_params);
// void camera_open_stream(Emergent::CEmergentCamera* camera, CameraParams *camera_params);
// void sort_cameras_ip(GigEVisionDeviceInfo *device_info, GigEVisionDeviceInfo *sorted_device_info, int cam_count);
// void get_senstemp_value(Emergent::CEmergentCamera *camera, CameraParams *camera_params);
#endif