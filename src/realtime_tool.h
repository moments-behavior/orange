#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL

#include "camera.h"
#include "opencv2/core/core.hpp"
#include "types.h"
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

// Jarvis-Hybrid Net specific constants
#define JARVIS_NUM_KEYPOINTS 4  // Snout, EarL, EarR, Tail
#define JARVIS_NUM_CAMERAS 4    // Default number of cameras
#define JARVIS_CENTER_IMG_SIZE 320
#define JARVIS_KEYPOINT_IMG_SIZE 192

// Jarvis keypoint names (from config)
enum JarvisKeypoint {
    SNOUT = 0,
    EAR_L = 1, 
    EAR_R = 2,
    TAIL = 3
};

struct JarvisCenter2d {
    std::atomic<bool> find_center;
    cv::Point2f center;  // 2D center coordinates
    float confidence;    // Detection confidence
    cv::Point2f proj_center;  // Projected from 3D
    
    JarvisCenter2d() : find_center(false), confidence(0.0f) {}
};

struct JarvisKeypoints2d {
    std::atomic<bool> find_keypoints;
    cv::Point2f keypoints[JARVIS_NUM_KEYPOINTS];  // 2D keypoint coordinates
    float confidence[JARVIS_NUM_KEYPOINTS];       // Confidence per keypoint
    cv::Point2f proj_keypoints[JARVIS_NUM_KEYPOINTS]; // Projected from 3D
    
    JarvisKeypoints2d() : find_keypoints(false) {
        for (int i = 0; i < JARVIS_NUM_KEYPOINTS; ++i) {
            confidence[i] = 0.0f;
        }
    }
};

struct DetectionDataPerCam {
    bool has_calibration_results;
    std::string calibration_file;
    CameraCalibResults camera_calib;
    Aruco2d marker2d;
    Ball2d ball2d;
    JarvisCenter2d jarvis_center;      // NEW: Jarvis center detection
    JarvisKeypoints2d jarvis_keypoints; // NEW: Jarvis keypoint detection
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


struct JarvisCenter3d {
    cv::Point3f center;           // 3D center coordinates (x, y, z)
    float confidence;             // Detection confidence
    std::atomic_bool new_detection;
    
    JarvisCenter3d() : confidence(0.0f), new_detection(false) {}
};

struct JarvisPose3d {
    cv::Point3f keypoints[JARVIS_NUM_KEYPOINTS];  // 3D keypoint coordinates
    float confidence[JARVIS_NUM_KEYPOINTS];       // Confidence per keypoint
    std::atomic_bool new_detection;
    
    JarvisPose3d() : new_detection(false) {
        for (int i = 0; i < JARVIS_NUM_KEYPOINTS; ++i) {
            confidence[i] = 0.0f;
        }
    }
};

struct Detection3d {
    Aruco3d marker3d;
    Ball3d ball3d;
    JarvisCenter3d jarvis_center;  // NEW: Jarvis 3D center results
    JarvisPose3d jarvis_pose;      // NEW: Jarvis 3D pose results
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
