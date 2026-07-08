#include "detect3d.h"
#include "galvo_sender.h"
#include "global.h"
#include "realtime_tool.h"
#include "video_capture.h"
#include <iostream>

bool all_ready(CameraEachSelect *cameras_select, std::vector<int> &cam3d_idx) {
    for (int idx : cam3d_idx) {
        if (cameras_select[idx].sigs->frame_detect_state.load() !=
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

        // triangulation calculation; latch the oldest contributing capture
        // timestamp before the states are reset (it stays valid until then)
        uint64_t capture_mono_ns = 0;
        for (int idx : cam3d_idx) {
            if (detection2d[idx].ball2d.find_ball.load()) {
                // make a copy
                std::vector<cv::Point2f> corners;
                corners.push_back(detection2d[idx].ball2d.center[0]);
                ball2d_all_cams.detected_points.push_back(corners);
                ball2d_all_cams.detected_cameras.push_back(idx);
                ball2d_all_cams.calib_results.push_back(
                    &detection2d[idx].camera_calib);
                uint64_t ts =
                    cameras_select[idx].sigs->frame_capture_mono_ns.load();
                if (ts != 0 && (capture_mono_ns == 0 || ts < capture_mono_ns)) {
                    capture_mono_ns = ts;
                }
            }
        }

        // reset the 3d camera states
        for (int idx : cam3d_idx) {
            cameras_select[idx].sigs->frame_detect_state.store(
                State_Copy_New_Frame);
        }

        bool found_ball3d = find_ball3d(&ball2d_all_cams, &detection3d.ball3d);
        detection3d.ball3d.new_detection.store(found_ball3d);

        // stream the triangulated target to the galvo motor-control app;
        // valid=0 tells it to hold instead of chasing a stale point. Skipped
        // while dummy_mode owns the sender. v2 carries the capture-time
        // position + velocity + measured pipeline age; the receiver
        // extrapolates to its own aim time.
        if (galvo_sender_params.enabled && !galvo_sender_params.dummy_mode) {
            if (found_ball3d) {
                galvo_sender_send_target_tracked(detection3d.ball3d.center.x,
                                                 detection3d.ball3d.center.y,
                                                 detection3d.ball3d.center.z,
                                                 true, capture_mono_ns);
            } else {
                galvo_sender_send_target_tracked(0, 0, 0, false, 0);
            }
        }

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
        detection3d.fps_estimator.update();
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
