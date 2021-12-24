#include<iostream>
#include "camera.h"


#define SUCCESS 0
#define MAX_CAMERAS 10

int main(int argc, char **args) {

    struct GigEVisionDeviceInfo deviceInfo[MAX_CAMERAS];
    if (!find_num_cameras(MAX_CAMERAS, deviceInfo)) {
        // take some fatal action
        return 0;
    }
    
    // Emergent::CEmergentCamera camera;
    // int ReturnVal = SUCCESS;
    // // may be pass this in as argument
    // u16 frame_rate = 100;
    // u16 gain = 3000; 
    // u16 exposure = 5000;

    // CameraParams camera_params = create_camera_params(frame_rate, gain, exposure);
    // Emergent::CEmergentFrame evtFrame[30], evtFrameRecv;
    // Camera camera = {camera_params, evtFrame[30], evtFrameRecv};

    return 0;
}