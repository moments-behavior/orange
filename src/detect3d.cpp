#include "detect3d.h"
#include "global.h"
#include "realtime_tool.h"
#include "video_capture.h"
#include "jarvis_pose_det.h"

bool all_ready(CameraEachSelect *cameras_select, std::vector<int> &cam3d_idx) {
    for (int idx : cam3d_idx) {
        if (cameras_select[idx].frame_detect_state.load() !=
            State_Frame_Detection_Ready) {
            return false;
        }
    }
    return true;
}

void detection3d_proc(CameraControl *camera_control,
                      CameraEachSelect *cameras_select, int num_cameras) {

    // threads for 3d triangulations
    std::vector<int> cam3d_idx;
    for (int i = 0; i < num_cameras; i++) {
        if (cameras_select[i].detect_mode == Detect3D_Standoff) {
            cam3d_idx.push_back(i);
        }
    }

    TriangulatePoints ball2d_all_cams;
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::time_point();
    int count = 0;
    while (camera_control->subscribe) {
        // start timing after 10 frames
        if (count == 10) {
            start = std::chrono::high_resolution_clock::now();
        }

        // 1. wait for all the detection ready; otherwise sleep
        std::unique_lock<std::mutex> lock(mtx3d);
        cv3d.wait(lock, [&] {
            return !camera_control->subscribe ||
                   all_ready(cameras_select, cam3d_idx);
        });

        if (!camera_control->subscribe) {
            break; // exit cleanly if subscription is turned off
        }

        if (!ball2d_all_cams.detected_cameras.empty()) {
            ball2d_all_cams.detected_cameras.clear();
            ball2d_all_cams.detected_points.clear();
            ball2d_all_cams.calib_results.clear();
        }

        // triangulation calculation
        for (int idx : cam3d_idx) {
            if (detection2d[idx].ball2d.find_ball.load()) {
                // make a copy
                std::vector<cv::Point2f> corners;
                corners.push_back(detection2d[idx].ball2d.center[0]);
                ball2d_all_cams.detected_points.push_back(corners);
                ball2d_all_cams.detected_cameras.push_back(idx);
                ball2d_all_cams.calib_results.push_back(
                    &detection2d[idx].camera_calib);
            }
        }

        // reset the 3d camera states
        for (int idx : cam3d_idx) {
            cameras_select[idx].frame_detect_state.store(State_Copy_New_Frame);
        }

        detection3d.ball3d.new_detection.store(
            find_ball3d(&ball2d_all_cams, &detection3d.ball3d));

        // project to all the streaming cameras
        if (detection3d.ball3d.new_detection.load()) {
            for (int i = 0; i < num_cameras; i++) {

                if (cameras_select[i].stream_on &&
                    detection2d[i].has_calibration_results) {

                    cv::Mat image_pts;
                    CameraCalibResults *cam_calib =
                        &detection2d[i].camera_calib;

                    std::vector<cv::Point3f> points3d;
                    points3d.push_back(detection3d.ball3d.center);

                    cv::projectPoints(points3d, cam_calib->rvec,
                                      cam_calib->tvec, cam_calib->k,
                                      cam_calib->dist_coeffs, image_pts);

                    // std::cout << image_pts.at<float>(0, 0) << ", "
                    //           << image_pts.at<float>(0, 1) << std::endl;
                    detection2d[i].ball2d.proj_center[0].x =
                        image_pts.at<float>(0, 0);
                    detection2d[i].ball2d.proj_center[0].y =
                        image_pts.at<float>(0, 1);
                }
            }
        }
        count++;
    }

    if (start == std::chrono::high_resolution_clock::time_point()) {
        // start is zero (uninitialized)
        std::cout << "Run it longer for meaning report of detection fps.\n";
    } else {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float calc_frame_rate = (count - 10) / elapsed.count();
        std::cout << "Triangule Frame Rate : " + std::to_string(calc_frame_rate)
                  << std::endl;
    }
}

// NEW: Jarvis 3D pose processing function
void jarvis_3d_pose_proc(CameraControl *camera_control,
                         CameraEachSelect *cameras_select, int num_cameras) {
    
    // Find cameras with Jarvis 3D pose detection enabled
    std::vector<int> jarvis_cam_idx;
    for (int i = 0; i < num_cameras; i++) {
        if (cameras_select[i].detect_mode == Detect3D_Pose) {
            jarvis_cam_idx.push_back(i);
        }
    }
    
    if (jarvis_cam_idx.empty()) {
        std::cout << "No cameras with Jarvis 3D pose detection enabled" << std::endl;
        return;
    }
    
    // Initialize Jarvis detector for multi-camera processing
    // Note: In a real implementation, we would need to coordinate between cameras
    // For now, this is a placeholder that shows the structure
    
    std::chrono::high_resolution_clock::time_point start =
        std::chrono::high_resolution_clock::time_point();
    int count = 0;
    
    while (camera_control->subscribe) {
        // Start timing after 10 frames
        if (count == 10) {
            start = std::chrono::high_resolution_clock::now();
        }
        
        // Wait for all Jarvis cameras to be ready
        std::unique_lock<std::mutex> lock(mtx3d);
        cv3d.wait(lock, [&] {
            return !camera_control->subscribe ||
                   all_ready(cameras_select, jarvis_cam_idx);
        });
        
        if (!camera_control->subscribe) {
            break;
        }
        
        // Collect 2D center detections from all cameras
        std::vector<cv::Point2f> centers_2d;
        std::vector<float> confidences;
        std::vector<int> detected_cameras;
        std::vector<CameraCalibResults*> calib_results;
        
        for (int idx : jarvis_cam_idx) {
            if (detection2d[idx].jarvis_center.find_center.load()) {
                centers_2d.push_back(detection2d[idx].jarvis_center.center);
                confidences.push_back(detection2d[idx].jarvis_center.confidence);
                detected_cameras.push_back(idx);
                calib_results.push_back(&detection2d[idx].camera_calib);
            }
        }
        
        // Reset camera states
        for (int idx : jarvis_cam_idx) {
            cameras_select[idx].frame_detect_state.store(State_Copy_New_Frame);
        }
        
        // Perform 3D center triangulation if we have enough cameras
        if (detected_cameras.size() >= 2) {
            // Collect 2D center points and calibration data for triangulation
            std::vector<cv::Point2f> center_points_2d;
            std::vector<CameraCalibResults*> calib_results;
            
            for (size_t i = 0; i < detected_cameras.size(); ++i) {
                int cam_idx = detected_cameras[i];
                if (detection2d[cam_idx].jarvis_center.find_center.load() &&
                    detection2d[cam_idx].has_calibration_results) {
                    center_points_2d.push_back(centers_2d[i]);
                    calib_results.push_back(&detection2d[cam_idx].camera_calib);
                }
            }
            
            if (center_points_2d.size() >= 2) {
                // Use existing triangulation function
                cv::Mat triangulated_3d = triangulate_points(center_points_2d, calib_results);
                cv::Point3f center_3d(triangulated_3d.at<float>(0), 
                                     triangulated_3d.at<float>(1), 
                                     triangulated_3d.at<float>(2));
                
                // Calculate average confidence
                float center_confidence = 0.0f;
                for (size_t i = 0; i < confidences.size(); ++i) {
                    center_confidence += confidences[i];
                }
                center_confidence /= confidences.size();
                
                // Update 3D center result
                detection3d.jarvis_center.center = center_3d;
                detection3d.jarvis_center.confidence = center_confidence;
                detection3d.jarvis_center.new_detection.store(true);
                
                std::cout << "Jarvis 3D center: (" << center_3d.x << ", " << center_3d.y 
                          << ", " << center_3d.z << ") confidence: " << center_confidence << std::endl;
                
                // Project 3D center back to all streaming cameras for visualization
                for (int i = 0; i < num_cameras; i++) {
                    if (cameras_select[i].stream_on && 
                        detection2d[i].has_calibration_results) {
                        
                        // Use existing 3D to 2D projection function
                        std::vector<cv::Point3d> points_3d = {center_3d};
                        std::vector<cv::Point2d> proj_points = project3d_to_2d(points_3d, &detection2d[i].camera_calib);
                        
                        if (proj_points.size() > 0) {
                            detection2d[i].jarvis_center.proj_center = proj_points[0];
                        }
                    }
                }
            }
        }
        
        // 3D keypoint processing
        if (detected_cameras.size() >= 2) {
            // Collect 2D keypoints from all cameras
            std::vector<std::vector<cv::Point2f>> keypoints_2d_all_cams(JARVIS_NUM_KEYPOINTS);
            std::vector<std::vector<CameraCalibResults*>> calib_results_all_keypoints(JARVIS_NUM_KEYPOINTS);
            
            // For each keypoint, collect 2D detections from all cameras
            for (int k = 0; k < JARVIS_NUM_KEYPOINTS; k++) {
                for (size_t i = 0; i < detected_cameras.size(); ++i) {
                    int cam_idx = detected_cameras[i];
                    if (detection2d[cam_idx].jarvis_keypoints.find_keypoints.load() &&
                        detection2d[cam_idx].jarvis_keypoints.confidence[k] > 0.5f &&
                        detection2d[cam_idx].has_calibration_results) {
                        keypoints_2d_all_cams[k].push_back(detection2d[cam_idx].jarvis_keypoints.keypoints[k]);
                        calib_results_all_keypoints[k].push_back(&detection2d[cam_idx].camera_calib);
                    }
                }
            }
            
            // Triangulate each keypoint if we have enough detections
            bool any_keypoint_detected = false;
            for (int k = 0; k < JARVIS_NUM_KEYPOINTS; k++) {
                if (keypoints_2d_all_cams[k].size() >= 2) {
                    // Use existing triangulation function
                    cv::Mat triangulated_3d = triangulate_points(keypoints_2d_all_cams[k], calib_results_all_keypoints[k]);
                    cv::Point3f keypoint_3d(triangulated_3d.at<float>(0), 
                                           triangulated_3d.at<float>(1), 
                                           triangulated_3d.at<float>(2));
                    
                    // Calculate average confidence for this keypoint
                    float keypoint_confidence = 0.0f;
                    int valid_detections = 0;
                    for (size_t i = 0; i < detected_cameras.size(); ++i) {
                        int cam_idx = detected_cameras[i];
                        if (detection2d[cam_idx].jarvis_keypoints.find_keypoints.load() &&
                            detection2d[cam_idx].jarvis_keypoints.confidence[k] > 0.5f) {
                            keypoint_confidence += detection2d[cam_idx].jarvis_keypoints.confidence[k];
                            valid_detections++;
                        }
                    }
                    keypoint_confidence /= valid_detections;
                    
                    // Store 3D keypoint result
                    detection3d.jarvis_pose.keypoints_3d[k] = keypoint_3d;
                    detection3d.jarvis_pose.confidence[k] = keypoint_confidence;
                    any_keypoint_detected = true;
                    
                    std::cout << "Jarvis keypoint " << k << " 3D: (" << keypoint_3d.x << ", " << keypoint_3d.y 
                              << ", " << keypoint_3d.z << ") confidence: " << keypoint_confidence << std::endl;
                } else {
                    detection3d.jarvis_pose.confidence[k] = 0.0f;
                }
            }
            
            if (any_keypoint_detected) {
                detection3d.jarvis_pose.new_detection.store(true);
                
                // Project 3D keypoints back to all streaming cameras for visualization
                for (int i = 0; i < num_cameras; i++) {
                    if (cameras_select[i].stream_on && 
                        detection2d[i].has_calibration_results) {
                        
                        for (int k = 0; k < JARVIS_NUM_KEYPOINTS; k++) {
                            if (detection3d.jarvis_pose.confidence[k] > 0.5f) {
                                // Use existing 3D to 2D projection function
                                std::vector<cv::Point3d> points_3d = {detection3d.jarvis_pose.keypoints_3d[k]};
                                std::vector<cv::Point2d> proj_points = project3d_to_2d(points_3d, &detection2d[i].camera_calib);
                                
                                if (proj_points.size() > 0) {
                                    detection2d[i].jarvis_keypoints.proj_keypoints[k] = proj_points[0];
                                }
                            }
                        }
                    }
                }
            }
        }
        
        count++;
    }
    
    if (start != std::chrono::high_resolution_clock::time_point()) {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float calc_frame_rate = (count - 10) / elapsed.count();
        std::cout << "Jarvis 3D Pose Frame Rate: " + std::to_string(calc_frame_rate)
                  << std::endl;
    }
}
