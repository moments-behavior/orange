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
    
    u16 frame_rate = 100;
    u16 gain = 3000; 
    u16 exposure = 5000;

    int ReturnVal = SUCCESS;

    Emergent::CEmergentCamera em_camera;
    CameraParams camera_params = create_camera_params(frame_rate, gain, exposure);
    Emergent::CEmergentFrame evtFrame[30], evtFrameRecv;
    //Camera camera = {em_camera, camera_params, evtFrame[30], evtFrameRecv};

    return 0;
}