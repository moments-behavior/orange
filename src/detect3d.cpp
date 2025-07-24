#include "detect3d.h"
#include "global.h"
#include "realtime_tool.h"
#include "video_capture.h"
#include <nvToolsExt.h>

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

    std::vector<TriangulatePoint> kps_all_cams;
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

        nvtxRangePush("copy detection");
        kps_all_cams.clear();
        for (int obj_id = 0; obj_id < 2; obj_id++) {
            for (int kp_idx = 0; kp_idx < 4; kp_idx++) {
                TriangulatePoint kp3d;
                for (int idx : cam3d_idx) {
                    if (detection2d[idx].dets.find_new.load()) {
                        for (int det_idx = 0;
                             det_idx < detection2d[idx].dets.obj2d.size();
                             det_idx++) {
                            Object detected_obj =
                                detection2d[idx].dets.obj2d[det_idx];
                            if (detected_obj.label == obj_id &&
                                !detected_obj.kps.empty()) {
                                if (detected_obj.kps[kp_idx * 3 + 2] > 0.5f) {
                                    cv::Point2f kp = {
                                        detected_obj.kps[kp_idx * 3],
                                        detected_obj.kps[kp_idx * 3 + 1]};
                                    kp3d.detected_points.push_back(kp);
                                    kp3d.detected_cameras.push_back(idx);
                                    kp3d.calib_results.push_back(
                                        &detection2d[idx].camera_calib);
                                    kp3d.obj_id = obj_id;
                                    kp3d.kp_id = kp_idx;
                                }
                            }
                        }
                    }
                }
                kps_all_cams.push_back(kp3d);
            }
        }
        nvtxRangePop();

        // reset the 3d camera states
        for (int idx : cam3d_idx) {
            cameras_select[idx].frame_detect_state.store(State_Copy_New_Frame);
        }

        nvtxRangePush("triangulation");
        detection3d.find_new = false;
        detection3d.kps.clear();
        detection3d.cam_object_kps.clear();
        bool any_true = false;
        for (size_t i = 0; i < kps_all_cams.size(); i++) {
            Keypoints3d kp3d;
            if (find_kp3d(&kps_all_cams[i], &kp3d)) {
                any_true = true;
                detection3d.kps.push_back(kp3d);
            }
        }
        nvtxRangePop();

        nvtxRangePush("reprojection");
        if (any_true) {
            // project to all the streaming cameras
            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on &&
                    detection2d[i].has_calibration_results) {
                    std::vector<cv::Point3f> points3d;
                    cv::Mat image_pts;
                    for (int j = 0; j < detection3d.kps.size(); j++) {
                        points3d.push_back(detection3d.kps[j].pt);
                    }
                    CameraCalibResults *cam_calib =
                        &detection2d[i].camera_calib;
                    cv::projectPoints(points3d, cam_calib->rvec,
                                      cam_calib->tvec, cam_calib->k,
                                      cam_calib->dist_coeffs, image_pts);

                    for (int j = 0; j < image_pts.rows; ++j) {
                        detection3d.cam_object_kps[i][detection3d.kps[j].obj_id]
                            .emplace_back(image_pts.at<cv::Point2f>(j).x);
                        detection3d.cam_object_kps[i][detection3d.kps[j].obj_id]
                            .emplace_back(image_pts.at<cv::Point2f>(j).y);
                    }
                }
            }
            detection3d.find_new.store(true);
        }
        nvtxRangePop();
        count++;
    }

    if (start == std::chrono::high_resolution_clock::time_point()) {
        // start is zero (uninitialized)
        std::cout << "Run it longer for meaningful report of detection fps.\n";
    } else {
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> elapsed = end - start;
        float calc_frame_rate = (count - 10) / elapsed.count();
        std::cout << "Triangule Frame Rate : " + std::to_string(calc_frame_rate)
                  << std::endl;
    }
}
