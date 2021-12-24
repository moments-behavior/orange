#ifndef ORANGE_CAMERAS
#define ORANGE_CAMERAS

#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <EvtParamAttribute.h>
#include <gigevisiondeviceinfo.h>
#include "types.h"


struct CameraParams{
    u16 frame_rate, frame_rate_max, frame_rate_min, frame_rate_inc;
    u16 width_max, height_max; 
    u16 gain;
    u16 exposure;
}; 

struct Camera{
    Emergent::CEmergentCamera camera;
    CameraParams camera_params;
    Emergent::CEmergentFrame evtFrame; 
    Emergent::CEmergentFrame evtFrameRecv; 
};

int find_num_cameras(int max_cameras, GigEVisionDeviceInfo* deviceInfo);
CameraParams create_camera_params(u16 frame_rate, u16 gain, u16 exposure);
void configure_factory_defaults(Emergent::CEmergentCamera* camera);


#endif
