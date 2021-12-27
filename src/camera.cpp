#include "camera.h"
#include "camera_driver_helper.h"
#include "thread.h"

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
    //check_camera_errors(Emergent::EVT_CameraGetEnumParamRange(camera, "PixelFormat", enumBuffer, enumBufferSize, &enumBufferSizeReturn));
    //check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera, "PixelFormat", enumMember));
    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "FrameRate", 30));

    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "OffsetX", 0));
    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "OffsetY", 0));

    check_camera_errors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width", &width_max));
    //check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera,    "Width", width_max));

    check_camera_errors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max));
    //check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera,    "Height", height_max));

    check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera, "AcquisitionMode", "Continuous"));
    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "AcquisitionFrameCount", 1));
    check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera, "TriggerSelector", "AcquisitionStart"));
    check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera, "TriggerMode", "Off"));
    check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera, "TriggerSource", "Software"));
    //check_camera_errors(Emergent::EVT_CameraSetEnumParam(camera, "BufferMode", "Off"));
    //check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "BufferNum", 0));

    check_camera_errors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "GevSCPSPacketSize", &param_val_max));
    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera,    "GevSCPSPacketSize", param_val_max));

    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "Gain", 256));
    check_camera_errors(Emergent::EVT_CameraSetUInt32Param(camera, "Offset", 0));

    check_camera_errors(Emergent::EVT_CameraSetBoolParam(camera, "LUTEnable", false));
    check_camera_errors(Emergent::EVT_CameraSetBoolParam(camera, "AutoGain", false));
}

void open_camera_with_params(Emergent::CEmergentCamera* camera, GigEVisionDeviceInfo* device_info, CameraParams camera_params)
{
    //TODO: open camera using xml file after explored on camera settings
    //EVT_CameraOpen(&camera, &deviceInfo[camera_index], XML_FILE);
    
    check_camera_errors(EVT_CameraOpen(camera, device_info));      

    configure_factory_defaults(camera);

    unsigned int width_max, height_max;
    check_camera_errors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Height", &height_max));
    check_camera_errors(Emergent::EVT_CameraGetUInt32ParamMax(camera, "Width" , &width_max));
    printf("Resolution: \t\t%d x %d\n", width_max, height_max); 


    const char* pixel_format = camera_params.pixel_format.c_str();
    check_camera_errors(EVT_CameraSetEnumParam(camera, "PixelFormat", pixel_format));
    printf("PixelFormat: \t\t%s\n", pixel_format);

    const char* color_temp = camera_params.color_temp.c_str();
    check_camera_errors(EVT_CameraSetUInt32Param(camera, "Gain", camera_params.gain));
    check_camera_errors(EVT_CameraSetUInt32Param(camera, "Exposure", camera_params.exposure));
    check_camera_errors(EVT_CameraSetEnumParam(camera, "ColorTemp", color_temp));

    unsigned int frame_rate_max;
    check_camera_errors(EVT_CameraGetUInt32ParamMax(camera, "FrameRate", &frame_rate_max));
    printf("FrameRate Max: \t\t%d\n", frame_rate_max);

    check_camera_errors(EVT_CameraSetUInt32Param(camera, "FrameRate", camera_params.frame_rate));
    printf("FrameRate Set to: \t%d\n", camera_params.frame_rate);
}


void close_camera(Emergent::CEmergentCamera* camera)
{    
    check_camera_errors(EVT_CameraClose(camera));
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
    check_camera_errors(EVT_CameraOpenStream(camera));

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
        check_camera_errors(EVT_AllocateFrameBuffer(camera, &evt_frame[frame_count], EVT_FRAME_BUFFER_ZERO_COPY));
        check_camera_errors(EVT_CameraQueueFrame(camera, &evt_frame[frame_count]));
    }
}


void destroy_frame_buffer(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* evt_frame, int buffer_size)
{
        //TODO: functionalize this Release frame buffers
	for(int frame_count=0;frame_count<buffer_size;frame_count++)
	{
		check_camera_errors(EVT_ReleaseFrameBuffer(camera, &evt_frame[frame_count]));
	}

	//Host side tear down for stream.
	check_camera_errors(EVT_CameraCloseStream(camera));

}



void aquire_num_frames(Emergent::CEmergentCamera* camera, Emergent::CEmergentFrame* frame_recv, int num_frames)
{
    int camera_return {0};

    //aquisition
    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStart"));

    unsigned short id_prev = 0, dropped_frames = 0;
    unsigned int frames_recd = 0;   

    float start_time = tick();
    for(int frame_count=0;frame_count<num_frames;frame_count++)
    {
        camera_return = EVT_CameraGetFrame(camera, frame_recv, EVT_INFINITE);
        if(!camera_return)
        {
            //Counting dropped frames through frame_id as redundant check. 
            if(((frame_recv->frame_id) != id_prev+1) && (frame_count != 0)) dropped_frames++;
            else
            {
                frames_recd++;
                // TODO: program what to do with received frame 

            }
        }
        else{dropped_frames++; printf("\nEVT_CameraGetFrame Error = %8.8x!\n", camera_return);}

        //In GVSP there is no id 0 so when 16 bit id counter in camera is max then the next id is 1 so set prev id to 0 for math above.
        if(frame_recv->frame_id == 65535)
            id_prev = 0;
        else
            id_prev = frame_recv->frame_id;

        if(camera_return)
            break; //No requeue reqd

        camera_return = EVT_CameraQueueFrame(camera, frame_recv); //Re-queue.
        if(camera_return) printf("EVT_CameraQueueFrame Error!\n");

        if(frame_count % 100 == 99) {printf("."); fflush(stdout);}    
        if(frame_count % 10000 == 9999) printf("\n");

        if(dropped_frames >= 100) break;
    }
    float end_time = tick();
    float time_diff = end_time - start_time;

    check_camera_errors(EVT_CameraExecuteCommand(camera, "AcquisitionStop"));

    //Report stats
    printf("\n");
    printf("Images Captured: \t%d\n", frames_recd);
    printf("Dropped Frames: \t%d\n", dropped_frames);
    printf("Calculated Frame Rate: \t%f\n", frames_recd/time_diff);
}