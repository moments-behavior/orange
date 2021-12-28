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
    
    // popular change to camera settings 
    unsigned int width {3208}; // TODO, make this parameters changeble
    unsigned int height {2200};
    unsigned int frame_rate {100};
    unsigned int gain {3000}; 
    unsigned int exposure {5000};
    string pixel_format = "BayerRG8"; // "YUV422Packed";
    string color_temp = "CT_3000K";

    Emergent::CEmergentCamera camera;
    CameraParams camera_params = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp);
    open_camera_with_params(&camera, &device_info[0], camera_params);
    
    int buffer_size {30};
    Emergent::CEmergentFrame evt_frame[buffer_size]; 
    allocate_frame_buffer(&camera, evt_frame, camera_params, buffer_size);
    Emergent::CEmergentFrame frame_recv;
    set_frame_buffer(&frame_recv, camera_params);

    int num_frames {1000};
    //aquire_num_frames(&camera, &frame_recv, num_frames);
    //aquire_and_display(&camera, &frame_recv, camera_params);

    // encoding using ffmpeg
    FILE *encoder_pipe;
    stringstream str_stream;
    
    // ffmpeg encode
    string pix_fmt;
    if (camera_params.pixel_format == "YUV422Packed") pix_fmt = "yuv422p";
    else if (camera_params.pixel_format  == "BayerRG8") pix_fmt = "bayer_rggb8";    
    string input_string = "/usr/local/bin/ffmpeg -y -f rawvideo -pix_fmt " + pix_fmt + " -video_size " + to_string(camera_params.width) + "x" + to_string(camera_params.height) + " -framerate " + to_string(camera_params.frame_rate) + " -i - ";
    string output_string = " -c:v h264_nvenc -r " + to_string(camera_params.frame_rate) + " my_output.mp4";
    str_stream << (input_string + output_string);
    //printf("Debug: %s", str_stream.str().c_str());
    
    if ( !(encoder_pipe = popen(str_stream.str().c_str(), "w")) ) {
        printf("popen error");
        // think about more general error handling with camera
        exit(1);
    }

    aquire_and_encode_ffmpeg(&camera, &frame_recv, num_frames, camera_params, encoder_pipe);


    fflush(encoder_pipe);
    fclose(encoder_pipe);


    // clean 
    destroy_frame_buffer(&camera, evt_frame, buffer_size);
    close_camera(&camera);
    return 0;
}