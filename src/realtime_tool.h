#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL

#include "imgui.h"
#include <opencv2/calib3d.hpp>
#include <opencv2/sfm.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "types.h"
#include "camera.h"
#include <map>
#include <iostream>

#define PI 3.14159265

struct CameraCalibResults
{
    cv::Mat k;
    cv::Mat dist_coeffs;
    cv::Mat r;
    cv::Mat rvec;
    cv::Mat tvec;
    cv::Mat projection_mat;
};

struct DetectionDataPerCam {
    bool have_calibration_results;
    std::string yolo_model;
    std::string calibration_file;
    CameraCalibResults camera_calib;
};

struct DetectionData {
    std::string yolo_model_folder; 
    std::string yolo_model; // TODO: remove this for the future? 
    std::string calibration_folder;
    std::vector<cv::Point3d> points3d;
    DetectionDataPerCam* detect_per_cam;
};

void print_calibration_results(CameraCalibResults* calib_results);
bool load_camera_calibration_results(std::string calibration_file, CameraCalibResults* calib_results); 
std::vector<cv::Point3d> unproject2d_to_3d(const std::vector<cv::Point2d> &points, const std::vector<double> &Z, CameraCalibResults *camera_calib);
std::vector<cv::Point2d> project3d_to_2d(const std::vector<cv::Point3d> &points, CameraCalibResults *camera_calib);
void world_coordinates_projection_points(CameraCalibResults* cvp, double* axis_x_values, double* axis_y_values, float scale, CameraParams* camera_params);
#endif