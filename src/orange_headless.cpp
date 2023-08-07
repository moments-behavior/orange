#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "project.h"
#include <filesystem>
#include <iostream>
#include "utils.h"

int main(int argc, char *argv[])
{


    int max_cameras = 20;
    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);
    std::cout << "available no of cameras: " << cam_count << std::endl;

    if (argc < 6)
    {
        std::cout << "Missing arguments. Use: sudo ./targets2/orange_headless num_cameras frame_rate frame_buffer_size num_gpus sync_bool capture_only" << std::endl; 
        return 1;
    }

    int num_cameras = atoi(argv[1]);
    int frame_rate = atoi(argv[2]);
    int frame_buffer_size = atoi(argv[3]);
    int num_gpus = atoi(argv[4]);

    bool capture_only = false;
    if (argc = 7) {
        capture_only = true;
    }

    bool sync_flag; 
    if (strcmp(argv[5], "true") == 0) {
        sync_flag = true;
    } else {
        sync_flag = false;
    }

    std::cout << "number of cameras: " << num_cameras << std::endl;
    std::cout << "frame_rate: " << frame_rate << std::endl;
    std::cout << "frame_buffer_size: " << frame_buffer_size << std::endl;

    std::filesystem::path cwd = std::filesystem::current_path();
    std::cout << "current directory: " << cwd << std::endl; 
    std::string delimiter = "/";
    // find user name
    std::vector<std::string> tokenized_path = string_split (cwd, delimiter);
    
    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    CameraControl *camera_control = new CameraControl;
    PTPParams* ptp_params = new PTPParams{0, 0};

    try {
        cameras_params = new CameraParams[num_cameras];
        ecams = new CameraEmergent[num_cameras];


        for (int i = 0; i < num_cameras; i++)
        {
            int gpu_id = i%num_gpus;
            init_7MP_camera_params_color(&cameras_params[i], i, num_cameras, 2000, 3000, gpu_id, frame_rate); 
            open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);

            int ReturnVal = 0;  
            ReturnVal = EVT_CameraOpenStream(&ecams[i].camera);
            if(ReturnVal != 0)
            {
                std::cout << "Camera" << i << ": EVT_CameraOpenStream: Error" << std::endl;
                return ReturnVal;
            }
            ecams[i].evt_frame = new Emergent::CEmergentFrame[frame_buffer_size];
            allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], frame_buffer_size);
            if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
            {
                allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
            }
        }

        camera_control->record_video = true; 
        camera_control->subscribe = true;
        camera_control->stream = false;
        camera_control->sync_camera = sync_flag;
        camera_control->capture_only = capture_only;
        std::string encoder_setup = "-codec h264 -preset p1 -fps " + std::to_string(cameras_params[0].frame_rate);
        std::string folder_string = current_date_time();
        std::string folder_name = "/home/" + tokenized_path[2] + "/Videos/" + folder_string;

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

        
        if (camera_control->sync_camera) {
        for (int i = 0; i < num_cameras; i++)
            {
                ptp_camera_sync(&ecams[i].camera);
            }
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
            destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, frame_buffer_size);
            delete[] ecams[i].evt_frame;
            EVT_CameraCloseStream(&ecams[i].camera);
            close_camera(&ecams[i].camera);
        }
    }
    catch(const char* msg) {
        std::cerr << msg << std::endl;
        for (int i = 0; i < num_cameras; i++)
        {
            EVT_CameraCloseStream(&ecams[i].camera);
            close_camera(&ecams[i].camera);
        }


    }

    return 0;
}
