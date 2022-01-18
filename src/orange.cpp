#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include "video_capture.h"
#include <thread>

void start_one_camera(CameraParams camera_params, GigEVisionDeviceInfo* device_info)
{
    int buffer_size {30};
    Emergent::CEmergentCamera camera;
    Emergent::CEmergentFrame evt_frame[buffer_size]; 

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
        aquire_frames_gpu_encode(&camera, &frame_recv, num_frames, camera_params);

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
    
    // popular change to camera settings 
    unsigned int width {3208}; // TODO, make this parameters changeble
    unsigned int height {2200};
    unsigned int frame_rate {210};
    unsigned int gain {1000}; 
    unsigned int exposure {4000};
    //library support these two formats for now
    string pixel_format = "BayerRG8"; // "YUV422Packed"; 
    string color_temp = "CT_2800K";

    CameraParams camera_params = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp);
    std::thread camera_thread_0(&start_one_camera, camera_params, &device_info[0]);
    
    camera_thread_0.join(); 
    return 0;
}