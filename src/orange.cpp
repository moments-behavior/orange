#include<iostream>
#include "camera.h"


#define SUCCESS 0


int main(int argc, char **args) {
    Emergent::CEmergentCamera camera;
    int ReturnVal = SUCCESS;
    u16 frame_rate = 100;
    u16 gain = 3000; 
    u16 exposure = 5000;

    CameraParams camera_params = create_camera_params(frame_rate, gain, exposure);

}