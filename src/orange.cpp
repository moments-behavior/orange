#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include "video_capture.h"
#include <thread>

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    return buf;
}


void start_one_camera(CameraParams camera_params, GigEVisionDeviceInfo* device_info, int* key_num_ptr, string folder_name)
{
    int buffer_size {30};
    Emergent::CEmergentCamera camera;
    Emergent::CEmergentFrame evt_frame[buffer_size]; 
    
    string encoder_setup = "-preset p1 -fps " + to_string(camera_params.frame_rate);
    const char *encoder_str = encoder_setup.c_str();

    // determine which gpu to use
    int gpu_idx = 0;
    const char* nic_ip = device_info->nic.ip4Address;    
    if(strcmp("192.168.2.20", nic_ip) == 0)
    {
        gpu_idx = 1;
    }

    try{        
        char* camera_ip = device_info->currentIp;
        string file_name = folder_name + "/" + camera_ip + ".mp4"; 
        const char *output_file= file_name.c_str();

        open_camera_with_params(&camera, device_info, camera_params); 

        // sync 
        // test_gpo_manual_toggle(&camera);
        // ptp_camera_sync(&camera);

        allocate_frame_buffer(&camera, evt_frame, camera_params, buffer_size);

        Emergent::CEmergentFrame frame_recv;
        set_frame_buffer(&frame_recv, camera_params);
        aquire_frames_gpu_encode(&camera, &frame_recv, camera_params, output_file, encoder_str, gpu_idx, key_num_ptr);
        destroy_frame_buffer(&camera, evt_frame, buffer_size);
        close_camera(&camera);
    }
    catch(int &ex)
    {
        printf("\nError...");
        destroy_frame_buffer(&camera, evt_frame, buffer_size);
        close_camera(&camera);
    }
}


int main(int argc, char **args) 
{
    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    GigEVisionDeviceInfo ordered_device_info[max_cameras];

    int num_cameras = 2;

    int cam_count;
    cam_count = check_cameras(max_cameras, device_info, ordered_device_info);
    if (cam_count < num_cameras) 
    {
        printf("Missing cameras...Exit\n");
        return 0;
    }


    set_ip_persistent_with_open_close_camera(device_info, num_cameras);

    // popular change to camera settings 
    unsigned int width {3208}; 
    unsigned int height {2200};
    unsigned int frame_rate {100};
    unsigned int gain {1000}; 
    unsigned int exposure {4000};
    string pixel_format = "BayerRG8"; 
    string color_temp = "CT_2800K";

    CameraParams camera_params = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp);
    std::vector<thread> camera_threads;

    // esc to exit 
    int key_num;
    int* key_num_ptr = &key_num;  


    string folder_string = current_date_time();
    //string folder_name = "/home/red/Videos/" + folder_string;
    string folder_name = "/mnt/md129/videos/" + folder_string;
    
    // Creating a directory to save recorded video;
    if (mkdir(folder_name.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return 0;
    }
    else
        std::cout << "Recorded video saves to : " << folder_name << std::endl;

    

    int selected_cameras[] = {0, 1, 3, 5};  
    int camera_id {0};
    for(int i = 0; i < num_cameras; i++)
    {
        camera_id = selected_cameras[i];
        camera_threads.push_back(std::thread(&start_one_camera, camera_params, &ordered_device_info[camera_id], key_num_ptr, folder_name));
    }

    // main thread event loop
    while (true){
        key_num = getchar();
        if(key_num == 27)
            {
                std::cout << "ESC pressed. Quit program." << std::endl;
                break;
            }
    }

    // wait for threads to join
    for (auto& t : camera_threads)
            t.join();
    

    std::cout << folder_name << std::endl;
    return 0;
}