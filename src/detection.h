#include "SyncDisplay.h"
#include "image_processing.h"
#include <cuda_runtime_api.h>

void detection_proc(SyncDisplay* sync_manager, CameraParams* camera_params, CameraControl* camera_control, unsigned char* display_buffer, int idx)
{

    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
	FrameGPU frame_original; // frame on gpu device 
    Debayer debayer;

    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);


    while(camera_control->subscribe) {        
        // wait for frame ready
        printf("wait for kick\n");
        sync_manager->WaitForKick();
        
        printf("detection\n");
        sync_manager->SignalMoveSent(idx);
        
        // start of per process operations  
        ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, sync_manager->m_frames[idx]->imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));
        if (camera_params->color){
            debayer_frame_gpu(camera_params, &frame_original, &debayer);
        } else {
            duplicate_channel_gpu(camera_params, &frame_original, &debayer);
        }
        ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        // end of per process operations

        printf("detection done \n");
		sync_manager->SignalDetectionDone(idx);
    }

    return;  
}