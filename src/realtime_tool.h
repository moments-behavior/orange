#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL

#include "camera.h"
#include "opencv2/core/core.hpp"
#include "types.h"
#include "yolov8.h"
#include <atomic>
#include <csignal>
#include <opencv2/calib3d.hpp>
#include <opencv2/sfm.hpp>
#include <unordered_map>
#include <vector>

#define PI 3.14159265

struct CameraCalibResults {
    cv::Mat k;
    cv::Mat dist_coeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projection_mat;
};

struct TriangulatePoint {
    int obj_id;
    int kp_id;
    std::vector<int> detected_cameras;
    std::vector<cv::Point2f> detected_points;
    std::vector<CameraCalibResults *> calib_results;
};

struct TriangulateMultiplePoints {
    int obj_id;
    int kp_id;
    std::vector<int> detected_cameras;
    std::vector<std::vector<cv::Point2f>> detected_points;
    std::vector<CameraCalibResults *> calib_results;
};

struct Aruco2d {
    int frame_number;
    bool find_marker;
    cv::Point2f marker_corners[4];
    cv::Point2f proj_corners[4];
};

struct DetectedObjects {
    std::atomic<bool> find_new;
    std::vector<Object> obj2d;
    DetectedObjects() : find_new(false) {}
};

struct DetectionDataPerCam {
    bool has_calibration_results;
    std::string calibration_file;
    CameraCalibResults camera_calib;
    DetectedObjects dets;
};

struct Aruco3d {
    int id;
    cv::Point3f corners[4];
    cv::Point3f t_vec;
    cv::Point3f normal;
    f32 angle_x_axis;
    std::atomic_bool new_detection;
};

struct Keypoints3d {
    cv::Point3f pt;
    int kp_id;
    int obj_id;
};

struct Detection3d {
    std::atomic<bool> find_new;
    std::vector<Keypoints3d> kps;
    std::unordered_map<int, std::unordered_map<int, std::vector<float>>>
        cam_object_kps;
};

void print_calibration_results(CameraCalibResults *calib_results);
bool load_camera_calibration_results(std::string calibration_file,
                                     CameraCalibResults *calib_results);
std::vector<cv::Point3d>
unproject2d_to_3d(const std::vector<cv::Point2d> &points,
                  const std::vector<double> &Z,
                  CameraCalibResults *camera_calib);
std::vector<cv::Point2d> project3d_to_2d(const std::vector<cv::Point3d> &points,
                                         CameraCalibResults *camera_calib);
void world_coordinates_projection_points(CameraCalibResults *cvp,
                                         double *axis_x_values,
                                         double *axis_y_values, float scale,
                                         CameraParams *camera_params);
cv::Mat triangulate_points(std::vector<cv::Point2f> image_points,
                           std::vector<CameraCalibResults *> calib_results);
void marker3d_to_pose(Aruco3d *aruco_maker_3d);
bool find_marker3d(TriangulateMultiplePoints *aruco_marker_2d,
                   std::vector<CameraCalibResults *> &calib_results,
                   Aruco3d *marker3d);
bool find_kp3d(TriangulatePoint *kp2d, Keypoints3d *kp3d);
#endif
