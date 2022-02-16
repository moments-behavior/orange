#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include "video_capture.h"
#include <thread>

void start_one_camera(CameraParams camera_params, GigEVisionDeviceInfo* device_info, int thread_id)
{
    int buffer_size {30};
    Emergent::CEmergentCamera camera;
    Emergent::CEmergentFrame evt_frame[buffer_size]; 

    string file_name = "video/camera_" + to_string(thread_id) + ".mp4"; 
    const char *output_file= file_name.c_str();
    
    string encoder_setup = "-preset p1 -fps " + to_string(camera_params.frame_rate);
    const char *encoder_str = encoder_setup.c_str();
    // std::cout << encoder_str << std::endl; 

    // determine which gpu to use
    int gpu_idx = 0;
    const char* nic_ip = device_info->nic.ip4Address;    
    if(strcmp("192.168.2.20", nic_ip) == 0)
    {
        gpu_idx = 1;
    }

    try{
        open_camera_with_params(&camera, device_info, camera_params);        
        allocate_frame_buffer(&camera, evt_frame, camera_params, buffer_size);

        Emergent::CEmergentFrame frame_recv;
        set_frame_buffer(&frame_recv, camera_params);

        int num_frames {1000};
        bool save_bmp_flag = true;
        //aquire_num_frames(&camera, &frame_recv, num_frames, camera_params, save_bmp_flag);
        //aquire_and_display(&camera, &frame_recv, camera_params);
        //aquire_and_encode_gstreamer(&camera, &frame_recv, num_frames, camera_params);
        //aquire_and_encode_ffmpeg(&camera, &frame_recv, num_frames, camera_params);
        aquire_frames_gpu_encode(&camera, &frame_recv, num_frames, camera_params, output_file, encoder_str, gpu_idx);

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
    if (!get_number_cameras(max_cameras, device_info)) 
    {
        return 0;
    }

    int num_cameras = 1;
    //set_rigroom_camera_ip(device_info, num_cameras);

    for (int camera_id = 0; camera_id < num_cameras; camera_id++)
    {
        print_camera_device_struct(device_info, camera_id);
    }

    // popular change to camera settings 
    unsigned int width {5120}; // TODO, make this parameters changeble, 3208 
    unsigned int height {4096}; // 2200
    unsigned int frame_rate {60};
    unsigned int gain {1000}; 
    unsigned int exposure {4000};
    //library support these two formats for now
    string pixel_format = "BayerGB8"; // "BayerRG8"; // "YUV422Packed"; 
    string color_temp = "CT_2800K";

    CameraParams camera_params = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp);

    std::vector<thread> camera_threads;

    for(int camera_id = 0; camera_id < num_cameras; camera_id++)
    {
        camera_threads.push_back(std::thread(&start_one_camera, camera_params, &device_info[camera_id], camera_id));
    }
    
    for (auto& t : camera_threads)
        t.join();
    
    return 0;
}