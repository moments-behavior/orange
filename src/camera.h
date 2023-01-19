#ifndef ORANGE_CAMERA
#define ORANGE_CAMERA

#include "camera_driver_helper.h"
#include <emergentcameradef.h>
#include <emergentgigevisiondef.h>
#include <EvtParamAttribute.h>
#include <unistd.h>


struct CameraParams{
    unsigned int width;
    unsigned int height;
    unsigned int frame_rate;
    unsigned int gain;
    unsigned int exposure;
    unsigned int focus;
    string pixel_format;
    string color_temp;
    int gpu_id;
    int camera_id;
    int num_cameras;
    bool gpu_direct;
    bool need_reorder;
    unsigned int gain_max; 
    unsigned int gain_min;
    unsigned int gain_inc;
    unsigned int exposure_max; 
    unsigned int exposure_min;
    unsigned int exposure_inc;
    unsigned int frame_rate_max; 
    unsigned int frame_rate_min;
    unsigned int frame_rate_inc;
    unsigned int width_max; 
    unsigned int width_min;
    unsigned int width_inc;
    unsigned int height_max; 
    unsigned int height_min;
    unsigned int height_inc;
    unsigned int offsetx_max; 
    unsigned int offsetx_min;
    unsigned int offsetx_inc;
    unsigned int offsety_max; 
    unsigned int offsety_min;
    unsigned int offsety_inc;
    unsigned int focus_max; 
    unsigned int focus_min;
    unsigned int focus_inc;
}; 


struct PTPParams{
    unsigned long long ptp_global_time; 
    uint64_t ptp_counter;
};


CameraParams create_camera_params(unsigned int width, unsigned int height, unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp, int camera_id, int gpu_id, int num_cameras, bool gpu_direct);
void configure_factory_defaults(Emergent::CEmergentCamera* camera);
void close_camera(Emergent::CEmergentCamera* camera);
void open_camera_with_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams* camera_params);
void allocate_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, CameraParams* camera_params, int buffer_size);
void set_frame_buffer(Emergent::CEmergentFrame* evt_frame, CameraParams* camera_params);
void destroy_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, int buffer_size);
void ptp_camera_sync(Emergent::CEmergentCamera* camera);
void quick_print_camera(GigEVisionDeviceInfo* device_info, int camera_idx);
unsigned long long get_current_PTP_time(Emergent::CEmergentCamera* camera);
void test_gpo_manual_toggle(Emergent::CEmergentCamera* camera);
void change_camera_ip_persistent(GigEVisionDeviceInfo* device_info, Emergent::CEmergentCamera* camera, const char* new_ip);
void update_gain_value(Emergent::CEmergentCamera* camera, int gain_val, CameraParams* camera_params);
void update_exposure_value(Emergent::CEmergentCamera* camera, int exposure_val, CameraParams* camera_params);
void update_frame_rate_value(Emergent::CEmergentCamera* camera, int frame_rate_val, CameraParams* camera_params);
void update_width_value(Emergent::CEmergentCamera* camera, int width_val, CameraParams* camera_params);
void update_height_value(Emergent::CEmergentCamera* camera, int height_val, CameraParams* camera_params);
void update_offsetX_value(Emergent::CEmergentCamera* camera, int OFFSET_X_VAL, CameraParams* camera_params);
void update_offsetY_value(Emergent::CEmergentCamera* camera, int OFFSET_Y_VAL, CameraParams* camera_params);
void update_focus_value(Emergent::CEmergentCamera* camera, int focus_value, CameraParams* camera_params);
#endif