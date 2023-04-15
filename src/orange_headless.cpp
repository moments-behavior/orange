#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "project.h"

int main(int argc, char *argv[])
{
    int max_cameras = 20;
    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);
    std::cout << "available no of cameras: " << cam_count << std::endl;
    int num_cameras = atoi(argv[1]);
    int frame_rate = atoi(argv[2]);
    int frame_buffer_size = atoi(argv[3]);

    std::cout << "number of cameras: " << num_cameras << std::endl;
    std::cout << "frame_rate: " << frame_rate << std::endl;
    std::cout << "frame_buffer_size: " << frame_buffer_size << std::endl;


    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    CameraControl *camera_control = new CameraControl;
    PTPParams* ptp_params = new PTPParams{0, 0};

    cameras_params = new CameraParams[num_cameras];
    ecams = new CameraEmergent[num_cameras];


    for (int i = 0; i < num_cameras; i++)
    {
        int gpu_id = i%4;
        init_7MP_camera_params_color(&cameras_params[i], i, num_cameras, 2000, 1500, gpu_id, frame_rate); 
        open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);

        int ReturnVal = 0;  
        ReturnVal = EVT_CameraOpenStream(&ecams[i].camera);
        if(ReturnVal != 0)
        {
            std::cout << "Camera" << i << ": EVT_CameraOpenStream: Error" << std::endl;
            return ReturnVal;
        }
        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], frame_buffer_size);
        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
        {
            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
        }
        set_frame_buffer(&ecams[i].frame_recv, &cameras_params[i]);
    }

    camera_control->record_video = true; 
    camera_control->subscribe = true;
    camera_control->stream = false;
    camera_control->m_slave = false;
    camera_control->sync_camera = true;
    string encoder_setup = "-preset p1 -fps " + to_string(cameras_params[0].frame_rate);
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

    for (int i = 0; i < num_cameras; i++)
    {
        ptp_camera_sync(&ecams[i].camera);
    }


    for (int i = 0; i < num_cameras; i++)
    {
        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, nullptr, encoder_setup, folder_name, ptp_params));
    }

    getchar();
    camera_control->subscribe = false;
    for (auto &t : camera_threads)
        t.join();

    for (int i = 0; i < num_cameras; i++)
    {
        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, 50);
        EVT_CameraCloseStream(&ecams[i].camera);
    }

    return 0;
}
