#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "project.h"

int main(int argc, char **args)
{
    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<thread> camera_threads;

    CameraControl *camera_control = new CameraControl;
    
    int num_cameras = 4;
    cameras_params = new CameraParams[num_cameras];
    ecams = new CameraEmergent[num_cameras];

    for (int i = 0; i < num_cameras; i++)
    {
        init_25G_camera_params(&cameras_params[i], i, num_cameras, 2000, 3000, i, 100);
        string multicast_ip = "239.255.255." + std::to_string(i);
        string iface_address = "192.168.1." + std::to_string(2*i + 20);
        ecams[i].camera.multicastAddress = multicast_ip.c_str(); 
        ecams[i].camera.ifaceAddress = iface_address.c_str();
        ecams[i].camera.portMulticast = 60646 + i;    
    }
    
    int ReturnVal = 0;
    ReturnVal = EVT_CameraOpenStream(&ecams[0].camera);
    if(ReturnVal != 0)
    {
        printf("EVT_CameraOpenStream: Error\n");
        return ReturnVal;
    }

    allocate_frame_buffer(&ecams[0].camera, ecams[0].evt_frame, &cameras_params[0], 100);
    if (cameras_params[0].need_reorder && cameras_params[0].gpu_direct)
    {
        allocate_frame_reorder_buffer(&ecams[0].camera, &ecams[0].frame_reorder, &cameras_params[0]);
    }
    set_frame_buffer(&ecams[0].frame_recv, &cameras_params[0]);

    string folder_string = current_date_time();
    string folder_name = "/home/user/Videos/" + folder_string;

    // Creating a directory to save recorded video;
    if (mkdir(folder_name.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return 0;
    }
    else
    {
        std::cout << "Recorded video saves to : " << folder_name << std::endl;
    }

    string encoder_setup = "-preset p1 -fps " + to_string(cameras_params[0].frame_rate);
    
    camera_control->record_video = true; 
    camera_control->streaming = true;

    for (int i = 0; i < num_cameras; i++)
    {
        camera_threads.push_back(std::thread(&headless_slave_aquire_frames_gpu_encode, &ecams[i], &cameras_params[i], camera_control, encoder_setup, folder_name));
    }

    getchar();
    camera_control->streaming = false;
    for (auto &t : camera_threads)
        t.join();

    for (int i = 0; i < num_cameras; i++)
    {
        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, 100);
        EVT_CameraCloseStream(&ecams[i].camera);
    }

    return 0;
}