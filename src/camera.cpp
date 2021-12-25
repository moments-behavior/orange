#include "camera.h"

// important camera tuning parameters
CameraParams create_camera_params(unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp)
{
    CameraParams camera_params = {};
    camera_params.frame_rate = frame_rate;
    camera_params.gain = gain;
    camera_params.exposure = exposure;
    camera_params.pixel_format = pixel_format;
    camera_params.color_temp = color_temp;
    return camera_params;
}


//A function to reset to factory defaults for running eSDK examples 
void configure_factory_defaults(Emergent::CEmergentCamera* camera)
{
    unsigned int width_max, height_max, param_val_max;
    const unsigned long enumBufferSize = 1000;
    unsigned long enumBufferSizeReturn = 0;
    char enumBuffer[enumBufferSize];
    char* next_token;
    char* enumMember = strtok_s(enumBuffer, ",", &next_token);

    //Order is important as param max/mins get updated.
    Emergent::EVT_CameraGetEnumParamRange(camera, "PixelFormat", enumBuffer, enumBufferSize, &enumBufferSizeReturn);
    Emergent::EVT_CameraSetEnumParam(camera,      "PixelFormat", enumMember);

    Emergent::EVT_CameraSetUInt32Param(camera,    "FrameRate", 30);

    Emergent::EVT_CameraSetUInt32Param(camera,    "OffsetX", 0);
    Emergent::EVT_CameraSetUInt32Param(camera,    "OffsetY", 0);

    Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max);
    Emergent::EVT_CameraSetUInt32Param(camera,    "Width", width_max);

    Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max);
    Emergent::EVT_CameraSetUInt32Param(camera,    "Height", height_max);

    Emergent::EVT_CameraSetEnumParam(camera,      "AcquisitionMode",        "Continuous");
    Emergent::EVT_CameraSetUInt32Param(camera,    "AcquisitionFrameCount",  1);
    Emergent::EVT_CameraSetEnumParam(camera,      "TriggerSelector",        "AcquisitionStart");
    Emergent::EVT_CameraSetEnumParam(camera,      "TriggerMode",            "Off");
    Emergent::EVT_CameraSetEnumParam(camera,      "TriggerSource",          "Software");
    Emergent::EVT_CameraSetEnumParam(camera,      "BufferMode",             "Off");
    Emergent::EVT_CameraSetUInt32Param(camera,    "BufferNum",              0);

    Emergent::EVT_CameraGetUInt32ParamMax(camera, "GevSCPSPacketSize", &param_val_max);
    Emergent::EVT_CameraSetUInt32Param(camera,    "GevSCPSPacketSize", param_val_max);

    Emergent::EVT_CameraSetUInt32Param(camera,    "Gain", 256);
    Emergent::EVT_CameraSetUInt32Param(camera,    "Offset", 0);

    Emergent::EVT_CameraSetBoolParam(camera,      "LUTEnable", false);
    Emergent::EVT_CameraSetBoolParam(camera,      "AutoGain", false);
}


int set_camera_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params)
{
    //TODO: open camera using xml file after explored on camera settings
    //ReturnVal = EVT_CameraOpen(&camera, &deviceInfo[camera_index], XML_FILE);
    // TODO: macro the error message
    
    const short SUCCESS {0};
    int ReturnVal = SUCCESS;

    ReturnVal = EVT_CameraOpen(camera, device_info);      

    configure_factory_defaults(camera);

    unsigned int width_max, height_max;
    Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max);
    Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width" , &width_max);

    printf("Resolution: \t\t%d x %d\n", width_max, height_max); 

    const char* pixel_format = camera_params.pixel_format.c_str();
    ReturnVal = EVT_CameraSetEnumParam(camera, "PixelFormat", pixel_format);
    printf("PixelFormat: \t\t%s\n", pixel_format);
    if(ReturnVal != SUCCESS)
    {
        printf("EVT_CameraSetEnumParam: PixelFormat Error\n");
        return ReturnVal;
    }

    const char* color_temp = camera_params.color_temp.c_str();
    ReturnVal = EVT_CameraSetUInt32Param(camera, "Gain", camera_params.gain);
    ReturnVal = EVT_CameraSetUInt32Param(camera, "Exposure", camera_params.exposure);
    ReturnVal = EVT_CameraSetEnumParam(camera, "ColorTemp", color_temp);

    unsigned int frame_rate_max, frame_rate_min, frame_rate_inc;
    EVT_CameraGetUInt32ParamMax(camera, "FrameRate", &frame_rate_max);
    printf("FrameRate Max: \t\t%d\n", frame_rate_max);
    EVT_CameraGetUInt32ParamMin(camera, "FrameRate", &frame_rate_min);
    printf("FrameRate Min: \t\t%d\n", frame_rate_min);
    EVT_CameraGetUInt32ParamInc(camera, "FrameRate", &frame_rate_inc);
    printf("FrameRate Inc: \t\t%d\n", frame_rate_inc);

    if ((camera_params.frame_rate > frame_rate_min)and (camera_params.frame_rate < frame_rate_max))
    {
        printf("FrameRate Set to: \t\t%d\n", camera_params.frame_rate);
        EVT_CameraSetUInt32Param(camera, "FrameRate", camera_params.frame_rate);
    }
    else
    {
        printf("Invalid frame rate.");
        return 0;
    }

    return ReturnVal;
}


void close_camera(Emergent::CEmergentCamera* camera)
{    
    EVT_CameraClose(camera);
    printf("\nClose Camera: \t\tCamera Closed\n");
}


//Find all cameras in system.
int get_number_cameras(int max_cameras, GigEVisionDeviceInfo* deviceInfo)
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