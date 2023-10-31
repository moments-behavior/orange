#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL
#include "imgui.h"
#include <opencv2/sfm.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "aruco_nano.h"
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


struct ArucoMarker2d 
{
    int id; 
    std::vector<int> detected_cameras;
    std::vector<std::vector<cv::Point2f>> detected_points;
};

struct ArucoMarker3d
{
    int id;
    cv::Point3f* corners;
    cv::Point2f** proj_corners;
    cv::Point3f t_vec;
    cv::Point3f normal; 
    f32 angle_x_axis;
    bool new_detection;
};

void print_calibration_results(CameraCalibResults* calib_results);
cv::Mat triangulate_points(std::vector<cv::Point2f> image_points, std::vector<CameraCalibResults*> calib_results);
void aruco_detection(unsigned char* display_buffer, CameraParams *cameras_params, ArucoMarker2d* aruco_marker_2d);
void marker3d_to_pose(ArucoMarker3d* aruco_maker_3d);
bool find_marker3d(ArucoMarker2d* aruco_marker_2d, ArucoMarker3d* aruco_maker_3d, CameraCalibResults* calib_results, int num_cameras);
std::map<unsigned int, cv::Point3f> get_3d_coordinates(std::vector<std::vector<cv::Rect>> bounding_boxes, std::vector<std::vector<int>> obj_ids, CameraCalibResults* CamParam);
bool load_camera_calibration_results(CameraCalibResults* calib_results, CameraParams *cameras_params);
void world_coordinates_projection_points(CameraCalibResults* cvp, double* axis_x_values, double* axis_y_values, float scale);
#endif