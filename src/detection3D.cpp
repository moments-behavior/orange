#include "detection3D.h"
#include <algorithm>

cv::Mat triangulate_points(std::vector<cv::Point2f> image_points, std::vector<CameraCalibResults*> calib_results)
{
    std::vector<cv::Mat> sfm_points2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output3d;
    for (int i=0; i<calib_results.size(); i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << image_points[i].x, image_points[i].y);
        cv::Mat pointUndistort;
        cv::undistortPoints(point, pointUndistort, calib_results[i]->k, calib_results[i]->dist_coeffs, cv::noArray(), calib_results[i]->k);
        sfm_points2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(calib_results[i]->projection_mat);
        
    }
    cv::sfm::triangulatePoints(sfm_points2d, projection_matrices, output3d);
    output3d.convertTo(output3d, CV_32F);
    return output3d;
}


void marker3d_to_pose(Aruco3d* aruco_maker_3d)
{
    aruco_maker_3d->t_vec = aruco_maker_3d->corners[0] + aruco_maker_3d->corners[1] + aruco_maker_3d->corners[2] + aruco_maker_3d->corners[3];
    aruco_maker_3d->t_vec = aruco_maker_3d->t_vec / 4.0;
    std::cout << aruco_maker_3d->t_vec << std::endl;
    cv::Point3f corner1to4 = aruco_maker_3d->corners[3] - aruco_maker_3d->corners[0];
    cv::Point3f corner1to2 = aruco_maker_3d->corners[1] - aruco_maker_3d->corners[0];
    aruco_maker_3d->normal = corner1to4.cross(corner1to2);
    aruco_maker_3d->normal =  aruco_maker_3d->normal / cv::norm(aruco_maker_3d->normal);

    aruco_maker_3d->angle_x_axis = atan2(corner1to4.y, corner1to4.x);
    f32 result = aruco_maker_3d->angle_x_axis * 180 / PI;
    // printf("The marker is %f degrees from world x-axis. \n",  result);
}


bool find_marker3d(TriangulatePoints* aruco_marker_2d, std::vector<CameraCalibResults*>& calib_results, Aruco3d* marker3d)
{
    int num_detected_cams = aruco_marker_2d->detected_cameras.size();
    if (num_detected_cams >= 2) {
        // triangulate
        std::vector<CameraCalibResults*> calib_results_all; 
        for (size_t i = 0; i < num_detected_cams; i++) {
            calib_results_all.push_back(calib_results[aruco_marker_2d->detected_cameras[i]]);
        }
 
        for (size_t i = 0; i < 4; i++) {
            std::vector<cv::Point2f> image_points_all;
            for (size_t j = 0; j < num_detected_cams; j++) {
                image_points_all.push_back(aruco_marker_2d->detected_points[j][i]);
            }
            cv::Mat output3d = triangulate_points(image_points_all, calib_results_all); 
            cv::Point3f pts3d = cv::Point3f(output3d.at<float>(0), output3d.at<float>(1), output3d.at<float>(2));
            marker3d->corners[i] = pts3d;
        }
        
    } else {
        return false;
    }

    // // print marker corners
    // for (size_t i = 0; i < 4; i++) {
    //     std::cout << marker3d->corners[i] << ", " << std::endl;
    // }

    marker3d_to_pose(marker3d);
    return true;
}


void detection3d_proc(SyncDetection* sync_detection, CameraControl* camera_control, DetectionData* detection_data)
{
    // threads for 3d triangulations
    std::vector<CameraCalibResults*> calib_results;
    for (int i =0; i < sync_detection->cam_ids.size(); i++) {
        calib_results.push_back(&detection_data->detect_per_cam[sync_detection->cam_ids[i]].camera_calib);
    }

    TriangulatePoints marker2d_all_cams;
    while(camera_control->subscribe) {
        bool frame_unread = std::all_of(sync_detection->frame_unread.begin(), sync_detection->frame_unread.end(), [](bool v) { return v;});
        // std::cout << frame_unread << "frame_unread" << std::endl;
                
        if (frame_unread) {
            // signal detection proc to start 
            for (int i =0; i < sync_detection->frame_ready.size(); i++) {
                sync_detection->frame_ready[i] = true;
            }
            // std::cout << sync_detection->frame_ready.size() << "wait for detection" << std::endl;


            // wait for detection done from all the threads 
            {
                // std::cout << sync_detection->detection_ready << std::endl;
                std::unique_lock<std::mutex> lock(sync_detection->m_mutex);
                while (!sync_detection->detection_ready && camera_control->subscribe) {
                    sync_detection->m_cond.wait(lock);
                    // std::cout << "wait for detection ready" << std::endl;
                }
                sync_detection->detection_ready = false;
                // std::cout << "triangulation start" << std::endl;

            }

            if (!marker2d_all_cams.detected_cameras.empty()) {
                marker2d_all_cams.detected_cameras.clear();
                marker2d_all_cams.detected_points.clear();
            }

            // triangulation calculation
            for (int j=0; j<sync_detection->m_frames.size(); j++) {
                // only detect 0 marker for now, need to make this general
                if (detection_data->detect_per_cam[j].marker2d.find_marker) {
                    std::vector<cv::Point2f> corners;
                    for (size_t i = 0; i < 4; i++) {
                        corners.push_back(detection_data->detect_per_cam[j].marker2d.marker_corners[i]);
                    }
                    marker2d_all_cams.detected_points.push_back(corners);
                    marker2d_all_cams.detected_cameras.push_back(j);
                }
            }

            detection_data->marker3d.new_detection = find_marker3d(&marker2d_all_cams, calib_results, &detection_data->marker3d);
            // std::cout << detection_data->marker3d->new_detection << std::endl;

            for (int i =0; i < sync_detection->frame_unread.size(); i++) {
                sync_detection->frame_unread[i] = false;
            }
        }
    }
}


void detection_proc(SyncDetection* sync_detection, CameraControl* camera_control, CameraParams* cameras_params, DetectionData* detection_data, int idx)
{

    CameraParams* camera_params = &cameras_params[idx];

    ck(cudaSetDevice(camera_params->gpu_id));
    
    // innitialization
	FrameGPU frame_original; // frame on gpu device 
    Debayer debayer;

    // for opencv processing 
    unsigned char *d_bgr;
    int size_of_picture = camera_params->width * camera_params->height * 3;
    cudaMalloc((void **)&d_bgr, size_of_picture);
    unsigned char* cpu_frame;
    cpu_frame = (unsigned char *)malloc(size_of_picture);

    // unsigned char *d_gray;
    // cudaMalloc((void **)&d_gray, camera_params->width * camera_params->height);
    // NppiSize d_gray_size;
    // d_gray_size.width = camera_params->width;
    // d_gray_size.height = camera_params->height;

    if (!camera_params->gpu_direct) {
        initalize_gpu_frame(&frame_original, camera_params);
    }
    initialize_gpu_debayer(&debayer, camera_params);

    aruconano::MarkerDetector MDetector;

    while(camera_control->subscribe) {
        // wait for frame ready
        while (!sync_detection->frame_ready[idx]) {
            usleep(10);
        }
        
        // need to know all of them has started, then reset frame ready 
        // std::cout << "detection per thread" << std::endl;
        
        // start of per process operations
        if (!camera_params->gpu_direct) {
            ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, sync_detection->m_frames[idx]->imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));
        } else {
            frame_original.d_orig = (unsigned char*) sync_detection->m_frames[idx]->imagePtr;
        }

        if (camera_params->color){
            debayer_frame_gpu(camera_params, &frame_original, &debayer);
        } else {
            duplicate_channel_gpu(camera_params, &frame_original, &debayer);
        }
        // detection, aruco marker and yolo per thread goes here
        rgba2bgr_convert(d_bgr, debayer.d_debayer, camera_params->width, camera_params->height, 0);
        ck(cudaMemcpy2D(cpu_frame, camera_params->width*3, d_bgr, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost));
        cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, cpu_frame).reshape(3, camera_params->height);

        // // convert rgba to grayscale, need to compare opencv marker detection and the package used, which is better, https://sourceforge.net/projects/aruco/ 
        // // const NppStatus npp_result = nppiRGBToGray_8u_AC4C1R(debayer.d_debayer, camera_params->width*4, d_gray, camera_params->width, d_gray_size);
        // // if (npp_result != 0)
        // // {
        // // std::cout << "\nNPP error %d \n"
        // //           << npp_result << std::endl;
        // // }
        // // ck(cudaMemcpy2D(detection_data->detection_per_cam[idx].cpu_frame_gray, camera_params->width, d_gray, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyDeviceToHost));
        // // cv::Mat view = cv::Mat(camera_params->width * camera_params->height, 1, CV_8U, detection_data->detection_per_cam[idx].cpu_frame).reshape(1, camera_params->height);

        std::vector<aruconano::Marker> markers = MDetector.detect(view);
        detection_data->detect_per_cam[idx].marker2d.find_marker = false;
        for (size_t i = 0; i < markers.size(); i++) {
            // std::cout << "marder id: " << markers[i].id << std::endl;
            if (markers[i].id == 20) {
                detection_data->detect_per_cam[idx].marker2d.find_marker = true;
                for (size_t j = 0; j < 4; j++) {
                    detection_data->detect_per_cam[idx].marker2d.marker_corners[j] = markers[i][j];
                }
            }
        }

        // signal done 
        {
            std::unique_lock<std::mutex> lock(sync_detection->m_mutex);
            // check if all the threads are ready 
            sync_detection->frame_ready[idx] = false;
            bool all_done = true;
            for (int i = 0; i < sync_detection->frame_ready.size(); i++) {
                // std::cout << sync_detection->frame_ready[i] << std::endl;
                all_done = all_done && (!sync_detection->frame_ready[i]);
            }

            if (all_done){
                sync_detection->detection_ready = true;
                // std::cout << "all detection done" << std::endl;
                sync_detection->m_cond.notify_all();
            }
        }
    }
}