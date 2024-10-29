#ifndef ORANGE_CAMERA_MANAGER
#define ORANGE_CAMERA_MANAGER
#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include "video_capture.h"
#include <thread>

const std::string current_date_time();
void start_one_camera(CameraParams camera_params, GigEVisionDeviceInfo* device_info, int* key_num_ptr, string folder_name, PTPParams* ptp_params);
void camera_manager(); 
#endif