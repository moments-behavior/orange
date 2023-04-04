#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "project.h"


int main(int argc, char *argv[])
{
    int max_cameras = 16;
    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);
    std::cout << "no of cameras: " << cam_count << std::endl;
    int num_cameras = atoi(argv[1]);

    for (unsigned int i = 0; i < count; i++) {
        change_camera_ip_persistent(device_info[i], , const char* new_ip)
    }
}