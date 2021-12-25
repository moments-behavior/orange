#ifndef ORANGE_CAMERAS
#define ORANGE_CAMERAS

#include <EmergentCameraAPIs.h>
#include <emergentframe.h>
#include <EvtParamAttribute.h>
#include <gigevisiondeviceinfo.h>


struct CameraParams{
    unsigned int frame_rate;
    unsigned int gain;
    unsigned int exposure;
    string pixel_format;
    string color_temp;
}; 

// struct Camera{
//     Emergent::CEmergentCamera* emergent_cam;
//     CameraParams params;
//     Emergent::CEmergentFrame* evtFrame; 
//     Emergent::CEmergentFrame* evtFrameRecv; 
// };

CameraParams create_camera_params(unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp);
int get_number_cameras(int max_cameras, GigEVisionDeviceInfo* deviceInfo);
void configure_factory_defaults(Emergent::CEmergentCamera* camera);
void close_camera(Emergent::CEmergentCamera* camera);
int set_camera(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params);


#endif
