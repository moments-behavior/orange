#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL

#include "camera.h"
#include "opencv2/core/core.hpp"
#include "types.h"
#include "utils.h"
#include <atomic>
#include <csignal>
#include <opencv2/calib3d.hpp>
#include <opencv2/sfm.hpp>
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

struct TriangulatePoints {
    int id;
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

struct Ball2d {
    std::atomic<bool> find_ball;
    cv::Point2f center[1];
    std::vector<cv::Rect_<float>> rects;
    cv::Point2f proj_center[1];
    Ball2d() : find_ball(false) {}
};

struct DetectionDataPerCam {
    bool has_calibration_results;
    std::string calibration_file;
    CameraCalibResults camera_calib;
    Aruco2d marker2d;
    Ball2d ball2d;
    FPSEstimator fps_estimator;
};

struct Aruco3d {
    int id;
    cv::Point3f corners[4];
    cv::Point3f t_vec;
    cv::Point3f normal;
    f32 angle_x_axis;
    std::atomic_bool new_detection;
};

struct Ball3d {
    cv::Point3f center;
    std::atomic_bool new_detection;
};

struct Detection3d {
    Aruco3d marker3d;
    Ball3d ball3d;
    FPSEstimator fps_estimator;
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
bool find_marker3d(TriangulatePoints *aruco_marker_2d,
                   std::vector<CameraCalibResults *> &calib_results,
                   Aruco3d *marker3d);
bool find_ball3d(TriangulatePoints *ball_2d, Ball3d *ball3d);
#endif
