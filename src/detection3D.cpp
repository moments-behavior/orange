#include "detection3D.h"

void detection3d_proc(SyncDetection* sync_detection, CameraControl* camera_control)
{
    // threads for 3d triangulations
    // ArucoMarker2d marker2d_all_cams;
    while(camera_control->subscribe) {
        // grab new data
        if(sync_detection->frame_unread) {
             
            // signal detection proc to start 
            for (int i =0; i < sync_detection->frame_ready.size(); i++) {
                sync_detection->frame_ready[i] = true;
            }
            std::cout << sync_detection->frame_ready.size() << "wait for detection" << std::endl;


            // wait for detection done from all the threads 
            {
                std::cout << sync_detection->detection_ready << std::endl;
                std::unique_lock<std::mutex> lock(sync_detection->m_mutex);
                while (!sync_detection->detection_ready && camera_control->subscribe) {
                    sync_detection->m_cond.wait(lock);
                    std::cout << "wait for detection ready" << std::endl;
                }
                sync_detection->detection_ready = false;
                std::cout << "triangulation start" << std::endl;

            }


            std::this_thread::sleep_for(std::chrono::seconds(2));

            // if (!marker2d_all_cams.detected_cameras.empty()) {
            //     marker2d_all_cams.detected_cameras.clear();
            //     marker2d_all_cams.detected_points.clear();
            // }

            // // triangulation calculation
            // for (int j =0; j < num_sync_cameras; j++) {
            //     // only detect 0 marker for now, need to make this general
            //     if (detection_data->detection_per_cam[j].find_marker) {
            //         std::vector<cv::Point2f> corners;
            //         for (size_t i = 0; i < 4; i++) {
            //             corners.push_back(detection_data->detection_per_cam[j].marker_corners[i]);
            //         }
            //         marker2d_all_cams.detected_points.push_back(corners);
            //         marker2d_all_cams.detected_cameras.push_back(j);
            //     }
            // }

            // detection_data->marker3d->new_detection = find_marker3d(&marker2d_all_cams, detection_data->marker3d, detection_data->camera_calib, num_sync_cameras);
            // std::cout << detection_data->marker3d->new_detection << std::endl;

            sync_detection->frame_unread = false;
        }
    }
}


void detection_proc(SyncDetection* sync_detection, CameraControl* camera_control, int idx)
{

    // ck(cudaSetDevice(camera_params->gpu_id));
    // // innitialization
	// FrameGPU frame_original; // frame on gpu device 
    // Debayer debayer;

    // // for opencv processing 
    // unsigned char *d_bgr;
    // cudaMalloc((void **)&d_bgr, camera_params->width * camera_params->height * 3);

    // unsigned char *d_gray;
    // cudaMalloc((void **)&d_gray, camera_params->width * camera_params->height);

    // NppiSize d_gray_size;
    // d_gray_size.width = camera_params->width;
    // d_gray_size.height = camera_params->height;

    // initalize_gpu_frame(&frame_original, camera_params);
    // initialize_gpu_debayer(&debayer, camera_params);

    // aruconano::MarkerDetector MDetector;
    // DetectionResults detection_results;
    // detection_results.camera_id = idx;

    while(camera_control->subscribe) {
        // wait for frame ready
        while (!sync_detection->frame_ready[idx]) {
            usleep(10);
        }
        
        // need to know all of them has started, then reset frame ready 
        std::cout << "detection per thread" << std::endl;
        std::this_thread::sleep_for(std::chrono::seconds(2));
        // // start of per process operations  
        // ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, sync_manager->m_frames[idx]->imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));
        // if (camera_params->color){
        //     debayer_frame_gpu(camera_params, &frame_original, &debayer);
        // } else {
        //     duplicate_channel_gpu(camera_params, &frame_original, &debayer);
        // }
        // // detection, aruco marker and yolo per thread goes here
        // rgba2bgr_convert(d_bgr, debayer.d_debayer, camera_params->width, camera_params->height, 0);
        // ck(cudaMemcpy2D(aruco_detection->marker2d[idx].cpu_frame, camera_params->width*3, d_bgr, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost));
        // cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, aruco_detection->marker2d[idx].cpu_frame).reshape(3, camera_params->height);

        // // convert rgba to grayscale, need to compare opencv marker detection and the package used, which is better, https://sourceforge.net/projects/aruco/ 
        // // const NppStatus npp_result = nppiRGBToGray_8u_AC4C1R(debayer.d_debayer, camera_params->width*4, d_gray, camera_params->width, d_gray_size);
        // // if (npp_result != 0)
        // // {
        // // std::cout << "\nNPP error %d \n"
        // //           << npp_result << std::endl;
        // // }
        // // ck(cudaMemcpy2D(detection_data->detection_per_cam[idx].cpu_frame_gray, camera_params->width, d_gray, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyDeviceToHost));
        // // cv::Mat view = cv::Mat(camera_params->width * camera_params->height, 1, CV_8U, detection_data->detection_per_cam[idx].cpu_frame).reshape(1, camera_params->height);

        // std::vector<aruconano::Marker> markers = MDetector.detect(view);
        // aruco_detection->marker2d[idx].find_marker = false;
        // for (size_t i = 0; i < markers.size(); i++) {
        //     // std::cout << markers[i].id << std::endl;
        //     if (markers[i].id == 21) {
        //         aruco_detection->marker2d[idx].find_marker = true;
        //         for (size_t j = 0; j < 4; j++) {
        //             aruco_detection->marker2d[idx].marker_corners[j] = markers[i][j];
        //         }
        //     }
        // }

        // signal done 
        {
            std::unique_lock<std::mutex> lock(sync_detection->m_mutex);
            // check if all the threads are ready 
            sync_detection->frame_ready[idx] = false;
            bool all_done = true;
            for (int i = 0; i < sync_detection->frame_ready.size(); i++) {
                std::cout << sync_detection->frame_ready[i] << std::endl;
                all_done = all_done && (!sync_detection->frame_ready[i]);
            }

            if (all_done){
                sync_detection->detection_ready = true;
                std::cout << "all detection done" << std::endl;
                sync_detection->m_cond.notify_all();
            }
        }
    }
}