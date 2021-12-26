#include <iostream>
#include "camera.h"

int main(int argc, char **args) {

    const short SUCCESS {0};
    int ReturnVal = SUCCESS;

    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    if (!get_number_cameras(max_cameras, device_info)) 
    {
        return 0;
    }
    
    // popular change to camera settings 
    unsigned int width {3208}; // TODO, make this parameters changeble
    unsigned int height {2200};
    unsigned int frame_rate {100};
    unsigned int gain {3000}; 
    unsigned int exposure {5000};
    string pixel_format = "YUV422Packed";
    string color_temp = "CT_3000K";

    // initialize number of cameras based on count, struct vector later
    Emergent::CEmergentCamera camera;
    CameraParams camera_params = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp);
    open_camera_with_params(&camera, &device_info[0], camera_params);
    
    int buffer_size = 30;
    Emergent::CEmergentFrame evt_frame[buffer_size]; 
    Emergent::CEmergentFrame frame_recv;
    allocate_frame_buffer(&camera, evt_frame, camera_params, buffer_size);

    
    
    //TODO: functionalize this Release frame buffers
	for(int frame_count=0;frame_count<buffer_size;frame_count++)
	{
		EVT_ReleaseFrameBuffer(&camera, &evt_frame[frame_count]);
	}

	//Host side tear down for stream.
	EVT_CameraCloseStream(&camera);


 
    if (ReturnVal!= SUCCESS)
    {
        close_camera(&camera);
        return 0;
    }


    close_camera(&camera);
    return 0;
}