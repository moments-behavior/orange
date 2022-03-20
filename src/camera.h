#ifndef ORANGE_CAMERA
#define ORANGE_CAMERA

#include "camera_driver_helper.h"
#include <emergentcameradef.h>
#include <emergentgigevisiondef.h>
#include <unistd.h>

struct CameraParams{
    unsigned int width;
    unsigned int height;
    unsigned int frame_rate;
    unsigned int gain;
    unsigned int exposure;
    string pixel_format;
    string color_temp;
    int camera_id;
    int num_cameras;
}; 


struct PTPParams{
    unsigned long long ptp_global_time; 
    uint64_t ptp_counter;
};


CameraParams create_camera_params(unsigned int width, unsigned int height, unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp, int camera_id, int num_cameras);
int check_cameras(int max_cameras, GigEVisionDeviceInfo *device_info, GigEVisionDeviceInfo *ordered_device_info);
void configure_factory_defaults(Emergent::CEmergentCamera* camera);
void close_camera(Emergent::CEmergentCamera* camera);
void open_camera_with_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params);
void allocate_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, CameraParams camera_params, int buffer_size);
void set_frame_buffer(Emergent::CEmergentFrame* evt_frame, CameraParams camera_params);
void destroy_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, int buffer_size);
void ptp_camera_sync(Emergent::CEmergentCamera* camera);
void quick_print_camera(GigEVisionDeviceInfo* device_info, int camera_idx);
void print_camera_device_struct(GigEVisionDeviceInfo* device_info, int camera_idx);
unsigned long long get_current_PTP_time(Emergent::CEmergentCamera* camera);
void test_gpo_manual_toggle(Emergent::CEmergentCamera* camera);
int get_camera_id(char* ip_address);
void change_camera_ip(GigEVisionDeviceInfo* device_info, int camera_idx, const char* new_ip);
void change_camera_ip_persistent(GigEVisionDeviceInfo* device_info, Emergent::CEmergentCamera* camera, const char* new_ip);
void set_temporary_camera_ip(GigEVisionDeviceInfo* device_info, int num_camera);
void set_rigroom_camera_ip_persistent(GigEVisionDeviceInfo* device_info, Emergent::CEmergentCamera* camera, int num_camera);
void set_ip_persistent_with_open_close_camera(GigEVisionDeviceInfo* device_info, int num_camera);
#endif