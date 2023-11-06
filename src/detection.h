#ifndef ORANGE_DETECTION
#define ORANGE_DETECTION

#include "SyncDisplay.h"
#include "image_processing.h"
#include <cuda_runtime_api.h>
#include "realtime_tool.h"
#include "kernel.cuh"

struct DetectionPerCam {
    bool have_calibration_results;
    unsigned char* cpu_frame;
    int frame_number;
    bool find_marker;
    cv::Point2f* marker_corners;
};

struct DetectionData {
    DetectionPerCam* detection_per_cam;
    CameraCalibResults* camera_calib;
    ArucoMarker3d* marker3d;
    bool draw_marker;
};

void allocate_detection_resources(DetectionData* detection_data, int num_cams, CameraParams* camera_params) {
        
    detection_data->detection_per_cam = (DetectionPerCam *)malloc(sizeof(DetectionPerCam) * num_cams);
    detection_data->camera_calib = (CameraCalibResults *)malloc(sizeof(CameraCalibResults) * num_cams);
    detection_data->marker3d = (ArucoMarker3d *) malloc(sizeof(ArucoMarker3d));

    for (u32 j = 0; j < num_cams; j++) {
        u32 size_pic = camera_params[j].width * camera_params[j].height * 3 * sizeof(unsigned char);
        detection_data->detection_per_cam[j].have_calibration_results = false;
        detection_data->detection_per_cam[j].cpu_frame = (unsigned char *)malloc(size_pic);
        detection_data->detection_per_cam[j].marker_corners = (cv::Point2f*)malloc(sizeof(cv::Point2f) * 4);
    }

    // proj corners
    detection_data->marker3d->proj_corners = (cv::Point2f**) malloc(sizeof(cv::Point2f*) * num_cams);
    for (u32 j = 0; j < num_cams; j++) {
        detection_data->marker3d->proj_corners[j] = (cv::Point2f*) malloc(sizeof(cv::Point2f) * 4);
    }

    detection_data->marker3d->corners = (cv::Point3f*) malloc(sizeof(cv::Point3f) * 4);
    detection_data->marker3d->new_detection = false;
    detection_data->draw_marker = true;
}

void detection3d_proc(SyncDisplay* sync_manager, DetectionData* detection_data, CameraControl* camera_control, int num_sync_cameras)
{
    // threads for 3d triangulations
    ArucoMarker2d marker2d_all_cams;

    while(camera_control->subscribe) {
        // std::cout << "wait for start tri" << std::endl; 
        sync_manager->WaitForTriangulation();

        sync_manager->SignalTriangulationInProc();
    
        if (!marker2d_all_cams.detected_cameras.empty()) {
            marker2d_all_cams.detected_cameras.clear();
            marker2d_all_cams.detected_points.clear();
        }

        // triangulation calculation
        for (int j =0; j < num_sync_cameras; j++) {
            // only detect 0 marker for now, need to make this general
            if (detection_data->detection_per_cam[j].find_marker) {
                std::vector<cv::Point2f> corners;
                for (size_t i = 0; i < 4; i++) {
                    corners.push_back(detection_data->detection_per_cam[j].marker_corners[i]);
                }
                marker2d_all_cams.detected_points.push_back(corners);
                marker2d_all_cams.detected_cameras.push_back(j);
            }
        }

        detection_data->marker3d->new_detection = find_marker3d(&marker2d_all_cams, detection_data->marker3d, detection_data->camera_calib, num_sync_cameras);
        // std::cout << detection_data->marker3d->new_detection << std::endl;
        // std::this_thread::sleep_for(std::chrono::milliseconds(16));

        // std::cout << "tri done" << std::endl; 
        sync_manager->SignalTriangulationDone();
    }
}

void detection_proc(SyncDisplay* sync_manager, CameraParams* camera_params, CameraControl* camera_control, CameraEachSelect* camera_select, unsigned char* display_buffer, DetectionData* detection_data, int idx)
{

    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
	FrameGPU frame_original; // frame on gpu device 
    Debayer debayer;

    // for opencv processing 
    unsigned char *d_bgr;
    cudaMalloc((void **)&d_bgr, camera_params->width * camera_params->height * 3);

    float *d_points;
    cudaMalloc((void **)&d_points, sizeof(float) * 10);
    float points[10];

    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);

    aruconano::MarkerDetector MDetector;
    // DetectionResults detection_results;
    // detection_results.camera_id = idx;

    while(camera_control->subscribe) {
        // wait for frame ready
        // printf("wait for kick\n");
        sync_manager->WaitForKick();
        
        // printf("detection\n");
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
        rgba2bgr_convert(d_bgr, debayer.d_debayer, camera_params->width, camera_params->height, 0);
        // copy back to cpu 
        ck(cudaMemcpy2D(detection_data->detection_per_cam[idx].cpu_frame, camera_params->width*3, d_bgr, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost));
        // aruco marker detection 
        cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, detection_data->detection_per_cam[idx].cpu_frame).reshape(3, camera_params->height);
        
        std::vector<aruconano::Marker> markers = MDetector.detect(view);
        detection_data->detection_per_cam[idx].find_marker = false;
        for (size_t i = 0; i < markers.size(); i++) {
            // std::cout << markers[i].id << std::endl;
            if (markers[i].id == 0) {
                detection_data->detection_per_cam[idx].find_marker = true;
                for (size_t j = 0; j < 4; j++) {
                    detection_data->detection_per_cam[idx].marker_corners[j] = markers[i][j];
                }
            }
        }

        // printf("detection done \n");
		sync_manager->SignalDetectionDone(idx);

        // printf("waif for triangulation done \n");
        sync_manager->WaitForTriangulationDone();
        // display
        // draw on gpu, need access to the detection_data 
        
        points[0] = detection_data->marker3d->proj_corners[idx][0].x;
        points[1] = detection_data->marker3d->proj_corners[idx][0].y;

        points[2] = detection_data->marker3d->proj_corners[idx][1].x;
        points[3] = detection_data->marker3d->proj_corners[idx][1].y;

        points[4] = detection_data->marker3d->proj_corners[idx][2].x;
        points[5] = detection_data->marker3d->proj_corners[idx][2].y;

        points[6] = detection_data->marker3d->proj_corners[idx][3].x;
        points[7] = detection_data->marker3d->proj_corners[idx][3].y;

        points[8] = detection_data->marker3d->proj_corners[idx][0].x;
        points[9] = detection_data->marker3d->proj_corners[idx][0].y;

        if (camera_select->stream_on) {
            ck(cudaMemcpy(d_points, points, sizeof(float) * 10, cudaMemcpyHostToDevice));
            gpu_draw_box(debayer.d_debayer, camera_params->width, camera_params->height, d_points, 0);
            ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));            
        }
        // printf("display done \n");
        sync_manager->SignalDisplayDone(idx);
    }

    return;  
}

#endif