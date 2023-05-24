#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <vector>
#include <iostream>
#include "camera_calibration.h"
#include "video_capture_gpu.h"
#include "types.h"
#include "gui.h"
#include "aruco.h"
#include <opencv2/sfm.hpp>

struct CPURender
{
    GLuint image_texture;
    PictureBuffer display_buffer;
};

void allocate_cpu_render_resources(CPURender *cpu_buffers, u32 image_width, u32 image_height)
{
    u32 size_pic = image_height * image_width * 3 * sizeof(unsigned char);

    cpu_buffers->display_buffer.frame = (unsigned char *)malloc(size_pic);
    clear_buffer_with_constant_image(cpu_buffers->display_buffer.frame, image_width, image_height);
    cpu_buffers->display_buffer.frame_number = 0;
    cpu_buffers->display_buffer.available_to_write = true;

    glGenTextures(1, &cpu_buffers->image_texture);
    glBindTexture(GL_TEXTURE_2D, cpu_buffers->image_texture);
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, image_width, image_height, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
    // Setup filtering parameters for display
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
}


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
    std::vector<cv::Point3f> corners;
    cv::Point3f t_vec;
    cv::Point3f r_vec;  
};


void load_camera_calibration_results(CameraCalibResults* calib_results, CameraParams *cameras_params) 
{
    std::string calibration_file = "/home/user/Calibration/world/calibration_aruco/Cam" + std::to_string(cameras_params->camera_id) + ".yaml";
    cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Could not open the calibration file: \"" << calibration_file << "\"" << std::endl;
    }
    fs["camera_matrix"] >> calib_results->k;
    fs["distortion_coefficients"] >> calib_results->dist_coeffs;
    fs["tc_ext"] >> calib_results->tvec;
    fs["rc_ext"] >> calib_results->r;
    fs.release();
    cv::Rodrigues(calib_results->r, calib_results->rvec);
    cv::sfm::projectionFromKRt(calib_results->k, calib_results->r, calib_results->tvec, calib_results->projection_mat);
}

void print_calibration_results(CameraCalibResults* calib_results) {
    std::cout << "k = " << std::endl << cv::format(calib_results->k, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "dist_coeffs  = " << std::endl << cv::format(calib_results->dist_coeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "r = " << std::endl << cv::format(calib_results->r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "tvec = " << std::endl << cv::format(calib_results->tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "rvec = " << std::endl << cv::format(calib_results->rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "projection_mat = " << std::endl << cv::format(calib_results->projection_mat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
}

cv::Point3f triangulate_points(std::vector<cv::Point2f> image_points, vector<CameraCalibResults*> calib_results)
{
    std::vector<cv::Mat> sfm_points2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output3d;
    for (int i=0; i<calib_results.size(); i++)
    {
        cv::Mat point = (cv::Mat_<float>(2, 1) << image_points[i].x, image_points[i].y);
        cv::Mat pointUndistort;
        std::cout << calib_results[i]->k << std::endl;
        cv::undistortPoints(point, pointUndistort, calib_results[i]->k, calib_results[i]->dist_coeffs, cv::noArray(), calib_results[i]->k);
        sfm_points2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(calib_results[i]->projection_mat);
        
    }

    cv::sfm::triangulatePoints(sfm_points2d, projection_matrices, output3d);
    output3d.convertTo(output3d, CV_32F);
    cv::Point3f pts3d = cv::Point3f(output3d.at<float>(0), output3d.at<float>(1), output3d.at<float>(2));
    return pts3d;
}


void aruco_detection(PictureBuffer* display_buffer, CameraParams *cameras_params, ArucoMarker2d* aruco_marker_2d) 
{    
    cv::Mat view = cv::Mat(cameras_params->width * cameras_params->height * 3, 1, CV_8U, display_buffer->frame).reshape(3, cameras_params->height);
    aruco::MarkerDetector MDetector;
    // detect 
    std::vector<aruco::Marker> markers = MDetector.detect(view);
    for (size_t i = 0; i < markers.size(); i++) {
        std::cout << markers[i] << std::endl;
        markers[i].draw(view);

        if (markers[i].id == 0) {
            // id 0 is ramp
            std::vector<cv::Point2f> corners;
            for (size_t j = 0; j < 4; j++) {
                corners.push_back(markers[i][j]);
            }
            aruco_marker_2d->detected_points.push_back(corners);
            aruco_marker_2d->detected_cameras.push_back(cameras_params->camera_id);
        }
    } 
}


void marker3d_to_pose(ArucoMarker3d* aruco_maker_3d)
{
    aruco_maker_3d->t_vec = aruco_maker_3d->corners[0] + aruco_maker_3d->corners[1] + aruco_maker_3d->corners[2] + aruco_maker_3d->corners[3];
    aruco_maker_3d->t_vec = aruco_maker_3d->t_vec / 4.0;
        
    cv::Point3f corner1to4 = aruco_maker_3d->corners[3] - aruco_maker_3d->corners[0];
    cv::Point3f corner1to2 = aruco_maker_3d->corners[1] - aruco_maker_3d->corners[0];
    aruco_maker_3d->r_vec = corner1to4.cross(corner1to2);
    aruco_maker_3d->r_vec =  aruco_maker_3d->r_vec / cv::norm(aruco_maker_3d->r_vec);
    
}

void find_marker3d(ArucoMarker2d* aruco_marker_2d, ArucoMarker3d* aruco_maker_3d, CameraCalibResults* calib_results)
{
    int num_detected_cams = aruco_marker_2d->detected_cameras.size();
    if (num_detected_cams > 2) {
        // triangulate
        vector<CameraCalibResults*> calib_results_all; 
        for (size_t i = 0; i < num_detected_cams; i++) {
            calib_results_all.push_back(calib_results + i);
        }
 
        for (size_t i = 0; i < 4; i++) {
            std::vector<cv::Point2f> image_points_all;
            for (size_t j = 0; j < num_detected_cams; j++) {
                image_points_all.push_back(aruco_marker_2d->detected_points[j][i]);
            }
            cv::Point3f output3d = triangulate_points(image_points_all, calib_results_all);
            aruco_maker_3d->corners.push_back(output3d);
        }
    }

    // // print marker corners
    // for (size_t i = 0; i < 4; i++) {
    //     std::cout << aruco_maker_3d->corners[i] << ", " << std::endl;
    // }

    marker3d_to_pose(aruco_maker_3d);
}


void load_calibration_config_file(std::string inputSettingsFile, Settings *calib_setting)
{
    FileStorage fs(inputSettingsFile, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
    }
    fs["Settings"] >> *calib_setting;
    fs.release();

    if (!calib_setting->goodInput)
    {
        cout << "Invalid input detected. Application stopping. " << endl;
    }
    else
    {
        std::cout << "Calibration configuration file loaded: \"" << inputSettingsFile << "\"" << std::endl;
    }
}

#endif