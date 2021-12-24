#include "camera.h"

// important camera tuning parameters
CameraParams create_camera_params(u16 frame_rate, u16 gain, u16 exposure)
{
    CameraParams camera_params = {};
    camera_params.frame_rate = frame_rate;
    camera_params.gain = gain;
    camera_params.exposure = exposure;
    return camera_params;
}


// //A function to reset to factory defaults for running eSDK examples 
// void configure_factory_defaults(Emergent::CEmergentCamera* camera)
// {
//     unsigned int width_max, height_max, param_val_max;
//     const unsigned long enumBufferSize = 1000;
//     unsigned long enumBufferSizeReturn = 0;
//     char enumBuffer[enumBufferSize];

//     //Order is important as param max/mins get updated.
//     Emergent::EVT_CameraGetEnumParamRange(camera, "PixelFormat", enumBuffer, enumBufferSize, &enumBufferSizeReturn);
//     char* enumMember = strtok_s(enumBuffer, ",", &next_token);
//     Emergent::EVT_CameraSetEnumParam(camera,      "PixelFormat", enumMember);

//     Emergent::EVT_CameraSetUInt32Param(camera,    "FrameRate", 30);

//     Emergent::EVT_CameraSetUInt32Param(camera,    "OffsetX", 0);
//     Emergent::EVT_CameraSetUInt32Param(camera,    "OffsetY", 0);

//     Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max);
//     Emergent::EVT_CameraSetUInt32Param(camera,    "Width", width_max);

//     Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max);
//     Emergent::EVT_CameraSetUInt32Param(camera,    "Height", height_max);

//     Emergent::EVT_CameraSetEnumParam(camera,      "AcquisitionMode",        "Continuous");
//     Emergent::EVT_CameraSetUInt32Param(camera,    "AcquisitionFrameCount",  1);
//     Emergent::EVT_CameraSetEnumParam(camera,      "TriggerSelector",        "AcquisitionStart");
//     Emergent::EVT_CameraSetEnumParam(camera,      "TriggerMode",            "Off");
//     Emergent::EVT_CameraSetEnumParam(camera,      "TriggerSource",          "Software");
//     Emergent::EVT_CameraSetEnumParam(camera,      "BufferMode",             "Off");
//     Emergent::EVT_CameraSetUInt32Param(camera,    "BufferNum",              0);

//     Emergent::EVT_CameraGetUInt32ParamMax(camera, "GevSCPSPacketSize", &param_val_max);
//     Emergent::EVT_CameraSetUInt32Param(camera,    "GevSCPSPacketSize", param_val_max);

//     Emergent::EVT_CameraSetUInt32Param(camera,    "Gain", 256);
//     Emergent::EVT_CameraSetUInt32Param(camera,    "Offset", 0);

//     Emergent::EVT_CameraSetBoolParam(camera,      "LUTEnable", false);
//     Emergent::EVT_CameraSetBoolParam(camera,      "AutoGain", false);
// }


//Find all cameras in system.
int find_num_cameras(int max_cameras, GigEVisionDeviceInfo* deviceInfo)
{
    int cameras_found = 0;
    unsigned int listcam_buf_size = max_cameras;
    unsigned int count;
    
    Emergent::EVT_ListDevices(deviceInfo, &listcam_buf_size, &count);
    if(count==0)
    {
        printf("Enumerate Cameras: \tNo cameras found. Exiting program.\n");
        return 0;
    }
    else
    {
        printf("Found %d cameras. \n", count);
        return count;
    }

}