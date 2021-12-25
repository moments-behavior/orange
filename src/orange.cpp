#include<iostream>
#include "camera.h"


int main(int argc, char **args) {

    const short SUCCESS {0};
    int ReturnVal = SUCCESS;

    short MAX_CAMERAS {10};
    struct GigEVisionDeviceInfo deviceInfo[MAX_CAMERAS];
    if (!get_number_cameras(MAX_CAMERAS, deviceInfo)) 
    {
        return 0;
    }
    
    // popular change to camera settings 
    unsigned int frame_rate {100};
    unsigned int gain {3000}; 
    unsigned int exposure {5000};
    string pixel_format = "YUV422Packed";
    string color_temp = "CT_3500K";

    // initialize number of cameras based on count, struct vector later
    Emergent::CEmergentCamera camera;
    CameraParams camera_params = create_camera_params(frame_rate, gain, exposure, pixel_format, color_temp);
    Emergent::CEmergentFrame evtFrame[30], evtFrameRecv;
    
    configure_factory_defaults(&camera);

    // use first camera 
    ReturnVal = set_camera_params(&camera, &deviceInfo[0], camera_params);
    
    // frame buffer
    

    // get frame


    // 
    
 
    if (ReturnVal!= SUCCESS)
    {
        close_camera(&camera);
        return 0;
    }


    close_camera(&camera);
    return 0;
}