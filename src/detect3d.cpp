#include "detect3d.h"
#include "global.h"
#include "realtime_tool.h"
#include "video_capture.h"

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
        if (cameras_select[i].detect_mode == Detect3d_Standoff) {
            cam3d_idx.push_back(i);
        }
    }

    TriangulatePoints ball2d_all_cams;
    auto start = std::chrono::high_resolution_clock::now();
    int count = 0;
    while (camera_control->subscribe) {
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
    auto end = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> elapsed = end - start;
    float calc_frame_rate = count / elapsed.count();
    std::cout << "Triangule frame Rate : " + std::to_string(calc_frame_rate)
              << std::endl;
}
