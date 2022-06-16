#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
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


int main(int argc, char **args) 
{
    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    GigEVisionDeviceInfo ordered_device_info[max_cameras];

    int num_cameras = 3;

    int cam_count;
    cam_count = check_cameras(max_cameras, device_info, ordered_device_info);
    if (cam_count < num_cameras) 
    {
        printf("Missing cameras...Exit\n");
        return 0;
    }

    // popular change to camera settings 
    unsigned int width {3208}; 
    unsigned int height {2200};
    unsigned int frame_rate {30};
    unsigned int gain {1000}; 
    unsigned int exposure {4000};
    string pixel_format = "BayerRG8"; 
    string color_temp = "CT_2800K";

    std::vector<thread> camera_threads;

    // esc to exit 
    int key_num;
    int* key_num_ptr = &key_num;  


    string folder_string = current_date_time();
    //string folder_name = "/home/red/Videos/" + folder_string;
    //string folder_name = "/mnt/md129/videos/" + folder_string;
    string folder_name = "/home/user/Videos/" + folder_string;
    
    
    // Creating a directory to save recorded video;
    if (mkdir(folder_name.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return 0;
    }
    else
        std::cout << "Recorded video saves to : " << folder_name << std::endl;


    PTPParams* ptp_params = new PTPParams{0, 0};
    int camera_orders[] = {0, 1, 2, 5, 3, 4, 6};  
    int camera_gpus[] = {0, 0, 0, 0, 1, 1, 1, };
    
    
    int camera_id {0};
    int gpu_id {0};
    int buffer_size {100};

    CameraParams camera_params[num_cameras];

    for(int i = 0; i < num_cameras; i++)
    {
        camera_id = camera_orders[i];
        gpu_id = camera_gpus[camera_id];
        camera_params[i] = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp, camera_id, gpu_id, num_cameras);
    }

    // init camera resources 
    Emergent::CEmergentCamera camera[num_cameras];
    Emergent::CEmergentFrame evt_frame[num_cameras][buffer_size]; 

    string encoder_setup = "-preset p1 -fps " + to_string(frame_rate);
    const char *encoder_str = encoder_setup.c_str();

    for(int i = 0; i < num_cameras; i++)
    {
        string file_name = folder_name + "/Cam" + std::to_string(camera_params[i].camera_id) + ".mp4"; 
        const char *output_file= file_name.c_str();

        open_camera_with_params(&camera[i], &ordered_device_info[i], camera_params[i]); 

        // sync 
        ptp_camera_sync(&camera[i]);

        allocate_frame_buffer(&camera[i], evt_frame[i], camera_params[i], buffer_size);
        Emergent::CEmergentFrame frame_recv;
        set_frame_buffer(&frame_recv, camera_params[i]);

        //aquire_frames_gpu_encode(&camera, &frame_recv, camera_params, output_file, encoder_str, key_num_ptr, ptp_params, folder_name);
        destroy_frame_buffer(&camera[i], evt_frame[i], buffer_size);
        close_camera(&camera[i]);
    }

    // for(int i = 0; i < num_cameras; i++)
    // {
    //     camera_threads.push_back(std::thread(&start_one_camera, camera_params, &ordered_device_info[camera_id], key_num_ptr, folder_name, ptp_params));
    // }

    // main thread event loop
    // while (true){
    //     key_num = getchar();
    //     if(key_num == 27)
    //         {
    //             std::cout << "ESC pressed. Quit program." << std::endl;
    //             break;
    //         }
    // }

    // // wait for threads to join
    // for (auto& t : camera_threads)
    //         t.join();
    

    std::cout << folder_name << std::endl;
    return 0;
}