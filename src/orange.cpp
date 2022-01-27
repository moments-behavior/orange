#include <iostream>
#include "camera.h"
#include "video_capture.h"

int main(int argc, char **args) 
{
    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    if (!get_number_cameras(max_cameras, device_info)) 
    {
        return 0;
    }

    int num_cameras = 4;
    for (int camera_id = 0; camera_id < num_cameras; camera_id++)
    {
        print_camera_device_struct(device_info, camera_id);
    }
   
    // popular change to camera settings 
    unsigned int width {3208}; // TODO, make this parameters changeble
    unsigned int height {2200};
    unsigned int frame_rate {100};
    unsigned int gain {3000}; 
    unsigned int exposure {5000};
    string pixel_format = "BayerRG8"; // "YUV422Packed"; library support these two formats for now
    string color_temp = "CT_3000K";

    Emergent::CEmergentCamera camera[num_cameras];
    CameraParams camera_params = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp);
    
    for (int camera_id = 0; camera_id < num_cameras; camera_id++)
    {
        open_camera_with_params(&camera[camera_id], &device_info[camera_id], camera_params);
    }

    // change ip address in persistent memory
    //set_rigroom_camera_ip_persistent(device_info, camera, num_cameras);

    for (int camera_id = 0; camera_id < num_cameras; camera_id++)
    {
        close_camera(&camera[camera_id]);
    }

    return 0;


    //**************************************
    // legacy
    // int buffer_size {30};
    // Emergent::CEmergentFrame evt_frame[buffer_size]; 
    // allocate_frame_buffer(&camera, evt_frame, camera_params, buffer_size);
    // Emergent::CEmergentFrame frame_recv;
    // set_frame_buffer(&frame_recv, camera_params);

    //int num_frames {1000};
    //aquire_num_frames(&camera, &frame_recv, num_frames);
    //aquire_and_display(&camera, &frame_recv, camera_params);
    //aquire_and_encode_gstreamer(&camera, &frame_recv, num_frames, camera_params);
    //aquire_and_encode_ffmpeg(&camera, &frame_recv, num_frames, camera_params);

    // clean 
    // destroy_frame_buffer(&camera, evt_frame, buffer_size);
    //*****************************************

}