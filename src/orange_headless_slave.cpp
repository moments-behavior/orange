#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "project.h"

int main(int argc, char *argv[])
{
    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<thread> camera_threads;

    CameraControl *camera_control = new CameraControl;
    PTPParams* ptp_params = new PTPParams{0, 0};

    int num_cameras = atoi(argv[1]);
    std::cout << "number of cameras: " << num_cameras << std::endl;

    cameras_params = new CameraParams[num_cameras];
    ecams = new CameraEmergent[num_cameras];

    for (int i = 0; i < num_cameras; i++)
    {
        init_25G_camera_params(&cameras_params[i], i, num_cameras, 2000, 3000, i, 10);
        
        string multicast_ip = "239.255.255." + std::to_string(i);
        // string iface_address = "192.168.1." + std::to_string(2*i + 20);
        string iface_address = "192.168.1.21";

        std::cout << "Ip: " << multicast_ip << ", iface_ip: " << iface_address << std::endl;
        cameras_params[i].camera_name.append(multicast_ip);
        ecams[i].camera.multicastAddress = multicast_ip.c_str(); 
        ecams[i].camera.ifaceAddress = iface_address.c_str();
        ecams[i].camera.portMulticast = 60646 + i;    
    
        int ReturnVal = 0;
        ReturnVal = EVT_CameraOpenStream(&ecams[i].camera);
        if(ReturnVal != 0)
        {
            std::cout << "Camera" << i << "EVT_CameraOpenStream: Error" << std::endl;
            return ReturnVal;
        }
    
        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], 100);
        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
        {
            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
        }
        set_frame_buffer(&ecams[i].frame_recv, &cameras_params[i]);
    }
    
     
    string folder_string = current_date_time();
    string folder_name = "/home/jinyao/Videos/" + folder_string;

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
        camera_threads.push_back(std::thread(&headless_slave_aquire_frames_gpu_encode, &ecams[i], &cameras_params[i], camera_control, encoder_setup, folder_name, ptp_params));
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