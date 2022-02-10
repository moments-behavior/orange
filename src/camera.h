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
}; 

// struct Camera{
//     Emergent::CEmergentCamera* emergent_cam;
//     CameraParams params;
//     Emergent::CEmergentFrame* evtFrame; 
//     Emergent::CEmergentFrame* evtFrameRecv; 
// };

CameraParams create_camera_params(unsigned int width, unsigned int height, unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp);
int get_number_cameras(int max_cameras, GigEVisionDeviceInfo* device_info);
void configure_factory_defaults(Emergent::CEmergentCamera* camera);
void close_camera(Emergent::CEmergentCamera* camera);
void open_camera_with_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params);
void allocate_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, CameraParams camera_params, int buffer_size);
void set_frame_buffer(Emergent::CEmergentFrame* evt_frame, CameraParams camera_params);
void destroy_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, int buffer_size);
void quick_print_camera(GigEVisionDeviceInfo* device_info, int camera_idx);
int get_camera_id(char* ip_address);
#endif