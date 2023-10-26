#include "SyncDisplay.h"
#include "image_processing.h"
#include <cuda_runtime_api.h>
#include "realtime_tool.h"
#include "kernel.cuh"

struct DetectionPerCam {
    CameraCalibResults camera_calib;
    bool have_calibration_results;
    unsigned char* cpu_frame;
    int frame_number;
};


struct DetectionResults {
    std::vector<cv::Point2f> marker_per_cam;
    int camera_id;
};

struct DetectionData {
    DetectionPerCam* detection_per_cam;
};

void allocate_detection_resources(DetectionData* detection_data, int num_cams, CameraParams* camera_params) {
        
    detection_data->detection_per_cam = (DetectionPerCam *)malloc(sizeof(DetectionPerCam) * num_cams);
    for (u32 j = 0; j < num_cams; j++) {
        u32 size_pic = camera_params[j].width * camera_params[j].height * 3 * sizeof(unsigned char);
        detection_data->detection_per_cam[j].have_calibration_results = false;
        detection_data->detection_per_cam[j].cpu_frame = (unsigned char *)malloc(size_pic);
    }
}

void detection3d_proc(SyncDisplay* sync_manager, CameraControl* camera_control, int num_sync_cameras)
{
    // threads for 3d triangulations
    DetectionResults detection_results[num_sync_cameras];

    while(camera_control->subscribe) {
        std::cout << "wait for start tri" << std::endl; 

        sync_manager->WaitForTriangulation();

        std::cout << "tri done" << std::endl; 
        sync_manager->SignalTriangulationDone();
    }
}

void detection_proc(SyncDisplay* sync_manager, CameraParams* camera_params, CameraControl* camera_control, unsigned char* display_buffer, DetectionPerCam* detection_per_cam, int idx)
{

    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
	FrameGPU frame_original; // frame on gpu device 
    Debayer debayer;

    // for opencv processing 
    unsigned char *d_rgb;
    cudaMalloc((void **)&d_rgb, camera_params->width * camera_params->height * 3);

    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);

    aruconano::MarkerDetector MDetector;
    // DetectionResults detection_results;
    // detection_results.camera_id = idx;

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
        // detection, aruco marker and yolo per thread goes here
        // need cuda kernel to convert to bgr, maybe can use another stream, since the default stream is used for gpu buffer
        rgba2bgr_convert(d_rgb, debayer.d_debayer, camera_params->width, camera_params->height, (cudaStream_t) 0);
        // copy back to cpu 
        ck(cudaMemcpy2DAsync(detection_per_cam->cpu_frame, camera_params->width*3, d_rgb, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost));
        // aruco marker detection 
        cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, detection_per_cam->cpu_frame).reshape(3, camera_params->height);
        std::vector<aruconano::Marker> markers = MDetector.detect(view);
        // for (size_t i = 0; i < markers.size(); i++) {
        //     std::cout << markers[i].id << std::endl;
        // }

        // display
        ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));
       
        
        
        std::this_thread::sleep_for(std::chrono::milliseconds(16));
        // end of per process operations

        printf("detection done \n");
		sync_manager->SignalDetectionDone(idx);
    }

    return;  
}