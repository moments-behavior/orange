#include "camera.h"

// TODO: move this function to a seperate source file, and finish this function.
string evtGetErrorString(EVT_ERROR error)
{
    string error_string; 
    if(error == EVT_ERROR_SRCH)
    {
        error_string = "No such process.";
    }
    else if(error == EVT_ERROR_INTR)
    {
        error_string = "Parameter not found.";

    }
    else
    {
        error_string = "General error.";

    }
    return error_string;
}

// important camera tuning parameters
CameraParams create_camera_params(unsigned int width, unsigned int height, unsigned int frame_rate, unsigned int gain, unsigned int exposure, string pixel_format, string color_temp)
{
    CameraParams camera_params = {};
    camera_params.width = width;
    camera_params.height = height;
    camera_params.frame_rate = frame_rate;
    camera_params.gain = gain;
    camera_params.exposure = exposure;
    camera_params.pixel_format = pixel_format;
    camera_params.color_temp = color_temp;
    return camera_params;
}


//A function to reset to factory defaults for running eSDK examples 
// TODO: many thing doesn't work with this emergent native code 
void configure_factory_defaults(Emergent::CEmergentCamera* camera)
{
    unsigned int width_max, height_max, param_val_max;
    //const unsigned long enumBufferSize = 1000;
    //unsigned long enumBufferSizeReturn = 0;
    //char enumBuffer[enumBufferSize];
    //char* next_token;
    //char* enumMember = strtok_s(enumBuffer, ",", &next_token);

    //Order is important as param max/mins get updated.
    //checkCameraErrors(Emergent::EVT_CameraGetEnumParamRange(camera, "PixelFormat", enumBuffer, enumBufferSize, &enumBufferSizeReturn));
    //checkCameraErrors(Emergent::EVT_CameraSetEnumParam(camera, "PixelFormat", enumMember));

    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "FrameRate", 30));

    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "OffsetX", 0));
    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "OffsetY", 0));

    checkCameraErrors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max));
    //checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera,    "Width", width_max));

    checkCameraErrors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max));
    //checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera,    "Height", height_max));

    checkCameraErrors(Emergent::EVT_CameraSetEnumParam(camera, "AcquisitionMode", "Continuous"));
    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1));
    checkCameraErrors(Emergent::EVT_CameraSetEnumParam(camera, "TriggerSelector", "AcquisitionStart"));
    checkCameraErrors(Emergent::EVT_CameraSetEnumParam(camera, "TriggerMode", "Off"));
    checkCameraErrors(Emergent::EVT_CameraSetEnumParam(camera, "TriggerSource", "Software"));
    //checkCameraErrors(Emergent::EVT_CameraSetEnumParam(camera, "BufferMode", "Off"));
    //checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "BufferNum", 0));

    checkCameraErrors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "GevSCPSPacketSize", &param_val_max));
    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera,    "GevSCPSPacketSize", param_val_max));

    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "Gain", 256));
    checkCameraErrors(Emergent::EVT_CameraSetUInt32Param(camera, "Offset", 0));

    checkCameraErrors(Emergent::EVT_CameraSetBoolParam(camera, "LUTEnable", false));
    checkCameraErrors(Emergent::EVT_CameraSetBoolParam(camera, "AutoGain", false));
}

void open_camera_with_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params)
{
    //TODO: open camera using xml file after explored on camera settings
    //EVT_CameraOpen(&camera, &deviceInfo[camera_index], XML_FILE);
    
    checkCameraErrors(EVT_CameraOpen(camera, device_info));      

    configure_factory_defaults(camera);

    unsigned int width_max, height_max;
    checkCameraErrors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max));
    checkCameraErrors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width" , &width_max));
    printf("Resolution: \t\t%d x %d\n", width_max, height_max); 


    const char* pixel_format = camera_params.pixel_format.c_str();
    checkCameraErrors(EVT_CameraSetEnumParam(camera, "PixelFormat", pixel_format));
    printf("PixelFormat: \t\t%s\n", pixel_format);

    const char* color_temp = camera_params.color_temp.c_str();
    checkCameraErrors(EVT_CameraSetUInt32Param(camera, "Gain", camera_params.gain));
    checkCameraErrors(EVT_CameraSetUInt32Param(camera, "Exposure", camera_params.exposure));
    checkCameraErrors(EVT_CameraSetEnumParam(camera, "ColorTemp", color_temp));

    unsigned int frame_rate_max;
    checkCameraErrors(EVT_CameraGetUInt32ParamMax(camera, "FrameRate", &frame_rate_max));
    printf("FrameRate Max: \t\t%d\n", frame_rate_max);

    checkCameraErrors(EVT_CameraSetUInt32Param(camera, "FrameRate", camera_params.frame_rate));
    printf("FrameRate Set to: \t%d\n", camera_params.frame_rate);
}


void close_camera(Emergent::CEmergentCamera* camera)
{    
    checkCameraErrors(EVT_CameraClose(camera));
    printf("\nClose Camera: \t\tCamera Closed\n");
}


//Find all cameras in system.
int get_number_cameras(int max_cameras, GigEVisionDeviceInfo* device_info)
{
    int cameras_found = 0;
    unsigned int listcam_buf_size = max_cameras;
    unsigned int count;
    
    Emergent::EVT_ListDevices(device_info, &listcam_buf_size, &count);
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

void allocate_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, CameraParams camera_params, int buffer_size)
{

    // open stream is important to acclocate frame buffer successfully
    checkCameraErrors(EVT_CameraOpenStream(camera));

    for(int frame_count=0;frame_count<buffer_size;frame_count++)
    {
        //Three params used for memory allocation. Worst case covers all models so no recompilation required.   
        evt_frame[frame_count].size_x = camera_params.width;
        evt_frame[frame_count].size_y = camera_params.height;

        string pixel_format  = camera_params.pixel_format; 
        if(pixel_format == "BayerRG8")
        {
            evt_frame[frame_count].pixel_type = GVSP_PIX_BAYRG8;
        }
        else if(pixel_format == "RGB8Packed")
        {
            evt_frame[frame_count].pixel_type = GVSP_PIX_RGB8;
        }
        else if(pixel_format == "BGR8Packed")
        {
            evt_frame[frame_count].pixel_type = GVSP_PIX_BGR8;
        }
        else if(pixel_format == "YUV411Packed")
        {
            evt_frame[frame_count].pixel_type = GVSP_PIX_YUV411_PACKED;
        }
        else if(pixel_format == "YUV422Packed")
        {
            evt_frame[frame_count].pixel_type = GVSP_PIX_YUV422_PACKED;
        }
        else if(pixel_format == "YUV444Packed")
        {
            evt_frame[frame_count].pixel_type = GVSP_PIX_YUV444_PACKED;
        }
    
        else //Good for default case which covers color and mono as same size bytes/pixel.
        {    //Note that these settings are used for memory alloc only.
            evt_frame[frame_count].pixel_type = GVSP_PIX_MONO8;
        }
        checkCameraErrors(EVT_AllocateFrameBuffer(camera, &evt_frame[frame_count], EVT_FRAME_BUFFER_ZERO_COPY));
        checkCameraErrors(EVT_CameraQueueFrame(camera, &evt_frame[frame_count]));
    }
}


void destroy_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, int buffer_size)
{
        //TODO: functionalize this Release frame buffers
	for(int frame_count=0;frame_count<buffer_size;frame_count++)
	{
		checkCameraErrors(EVT_ReleaseFrameBuffer(camera, &evt_frame[frame_count]));
	}

	//Host side tear down for stream.
	checkCameraErrors(EVT_CameraCloseStream(camera));

}
