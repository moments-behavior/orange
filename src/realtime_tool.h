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
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include <math.h>

#define PI 3.14159265

static void draw_cv_contours(std::vector<cv::Rect> boxes, std::vector<std::string> labels, std::vector<int> class_ids)
{
    for (int i=0; i< boxes.size(); i++)
    {
        double x[5] = {(double)boxes[i].x, (double)boxes[i].x, (double)boxes[i].x + boxes[i].width, (double)boxes[i].x + boxes[i].width, (double)boxes[i].x};
        double y[5] = {(double)2200 - boxes[i].y, (double)2200 - boxes[i].y - boxes[i].height, (double)2200 - boxes[i].y - boxes[i].height, (double)2200 - boxes[i].y, (double)2200 - boxes[i].y};
        
        if(class_ids[i] == 0){
            ImPlot::SetNextLineStyle(ImVec4(1.0, 0.0, 1.0,1.0), 3.0);
        } else{
            ImPlot::SetNextLineStyle(ImVec4(0.5, 1.0, 1.0,1.0), 3.0);}

        ImPlot::PlotLine(labels[i].c_str(), &x[0], &y[0], 5); 
    }
}


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
    cv::Point3f normal; 
    f32 angle_x_axis;
};


void load_camera_calibration_results(CameraCalibResults* calib_results, CameraParams *cameras_params) 
{
    std::string calibration_file = "/home/user/Calibration/4_edge_cams/Cam" + std::to_string(cameras_params->camera_id) + ".yaml";
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
        // std::cout << markers[i] << std::endl;
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
    aruco_maker_3d->normal = corner1to4.cross(corner1to2);
    aruco_maker_3d->normal =  aruco_maker_3d->normal / cv::norm(aruco_maker_3d->normal);

    aruco_maker_3d->angle_x_axis = atan2(corner1to4.y, corner1to4.x);
    f32 result = aruco_maker_3d->angle_x_axis * 180 / PI;
    printf("The marker is %f degrees from world x-axis. \n",  result);    

}


bool find_marker3d(ArucoMarker2d* aruco_marker_2d, ArucoMarker3d* aruco_maker_3d, CameraCalibResults* calib_results)
{
    int num_detected_cams = aruco_marker_2d->detected_cameras.size();
    if (num_detected_cams >= 2) {
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
    } else {
        return false;
    }

    // print marker corners
    for (size_t i = 0; i < 4; i++) {
        std::cout << aruco_maker_3d->corners[i] << ", " << std::endl;
    }

    marker3d_to_pose(aruco_maker_3d);
    return true;
}

std::map<unsigned int, cv::Point3f> get_3d_coordinates(vector<vector<cv::Rect>> bounding_boxes, vector<vector<int>> obj_ids, CameraCalibResults* CamParam)
{
    // points 
    std::map<unsigned int, vector<cv::Point2f>> mapOfObjects;
    std::map<unsigned int, vector<CameraCalibResults*>> mapOfCameras;
    
    // reformat detection data as dictionary 
    for (int cam_idx = 0; cam_idx < bounding_boxes.size(); cam_idx++){
        for (int box_id = 0; box_id < bounding_boxes[cam_idx].size(); box_id++) {
        // for (auto &i : bounding_boxes[cam_idx]) {
            if(mapOfObjects.count(obj_ids[cam_idx][box_id]) > 0){
                // calcualte center of mass from bounding box 
                float c_x =  float(bounding_boxes[cam_idx][box_id].x) + float(bounding_boxes[cam_idx][box_id].width)/2.0;
                float c_y =  float(bounding_boxes[cam_idx][box_id].y) + float(bounding_boxes[cam_idx][box_id].height)/2.0;
                mapOfObjects[obj_ids[cam_idx][box_id]].push_back(cv::Point2f(c_x, c_y));
                mapOfCameras[obj_ids[cam_idx][box_id]].push_back(CamParam + cam_idx);
            }
            else{
                float c_x =  float(bounding_boxes[cam_idx][box_id].x) + float(bounding_boxes[cam_idx][box_id].width)/2.0;
                float c_y =  float(bounding_boxes[cam_idx][box_id].y) + float(bounding_boxes[cam_idx][box_id].height)/2.0;
                vector<cv::Point2f> points_per_obj; 
                vector<CameraCalibResults*> camera_per_obj;
                points_per_obj.push_back(cv::Point2f(c_x, c_y));
                camera_per_obj.push_back(CamParam + cam_idx);
                mapOfObjects.insert({obj_ids[cam_idx][box_id], points_per_obj});
                mapOfCameras.insert({obj_ids[cam_idx][box_id], camera_per_obj});
            }
        }
    }

    // triangulation
    std::map<unsigned int, cv::Point3f> mapOfPoints3D;
    for ( auto it = mapOfObjects.begin(); it != mapOfObjects.end(); ++it)
    {
        if(it->second.size() >= 2){
            // triangulate if there are 2 camera detection
            cv::Point3f point3d = triangulate_points(it->second, mapOfCameras[it->first]);  
            mapOfPoints3D.insert({it->first, point3d});
        }
    }
    return mapOfPoints3D;
}



// https://stackoverflow.com/questions/21206870/opencv-rigid-transformation-between-two-3d-point-clouds
// eigen has a better implementation: http://eigen.tuxfamily.org/dox/group__Geometry__Module.html#gab3f5a82a24490b936f8694cf8fef8e60
cv::Vec3d CalculateMean(const cv::Mat_<cv::Vec3d> &points)
{
    cv::Mat_<cv::Vec3d> result;
    cv::reduce(points, result, 0, CV_REDUCE_AVG);
    return result(0, 0);
}

cv::Mat_<double> FindRigidTransform(const cv::Mat_<cv::Vec3d> &points1, const cv::Mat_<cv::Vec3d> points2)
{
    /* Calculate centroids. */
    cv::Vec3d t1 = -CalculateMean(points1);
    cv::Vec3d t2 = -CalculateMean(points2);

    cv::Mat_<double> T1 = cv::Mat_<double>::eye(4, 4);
    T1(0, 3) = t1[0];
    T1(1, 3) = t1[1];
    T1(2, 3) = t1[2];

    cv::Mat_<double> T2 = cv::Mat_<double>::eye(4, 4);
    T2(0, 3) = -t2[0];
    T2(1, 3) = -t2[1];
    T2(2, 3) = -t2[2];

    /* Calculate covariance matrix for input points. Also calculate RMS deviation from centroid
     * which is used for scale calculation.
     */
    cv::Mat_<double> C(3, 3, 0.0);
    double p1Rms = 0, p2Rms = 0;
    for (int ptIdx = 0; ptIdx < points1.rows; ptIdx++) {
        cv::Vec3d p1 = points1(ptIdx, 0) + t1;
        cv::Vec3d p2 = points2(ptIdx, 0) + t2;
        p1Rms += p1.dot(p1);
        p2Rms += p2.dot(p2);
        for (int i = 0; i < 3; i++) {
            for (int j = 0; j < 3; j++) {
                C(i, j) += p2[i] * p1[j];
            }
        }
    }

    cv::Mat_<double> u, s, vh;
    cv::SVD::compute(C, s, u, vh);

    cv::Mat_<double> R = u * vh;

    if (cv::determinant(R) < 0) {
        R -= u.col(2) * (vh.row(2) * 2.0);
    }

    double scale = sqrt(p2Rms / p1Rms);
    R *= scale;

    cv::Mat_<double> M = cv::Mat_<double>::eye(4, 4);
    R.copyTo(M.colRange(0, 3).rowRange(0, 3));

    cv::Mat_<double> result = T2 * M * T1;
    result /= result(3, 3);

    return result.rowRange(0, 3);
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



void world_coordinates_projection_points(CameraCalibResults* cvp, double* axis_x_values, double* axis_y_values, float scale)
{
    std::vector<cv::Point3f> world_coordinates;
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(scale * 1.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, scale * 1.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, scale * 1.0f));

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(world_coordinates, cvp->rvec, cvp->tvec, cvp->k, cvp->dist_coeffs, img_pts);
    
    for (int i = 0; i < 4; i++){
        axis_x_values[i] = img_pts.at(i).x;
        axis_y_values[i] = 2200 - img_pts.at(i).y;
    }
}

static void gui_plot_world_coordinates(CameraCalibResults* cvp, int cam_id)
{
    double axis_x_values[4]; double axis_y_values[4]; 
    world_coordinates_projection_points(cvp, axis_x_values, axis_y_values, 50);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, ImVec4(1.0, 1.0, 1.0,1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0,1.0), 3.0);
    std::string name = "World Origin";
    
    float one_axis_x[2];
    float one_axis_y[2];

    std::vector<triple_f> node_colors = {
        {1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}};
                
    for (u32 edge=0; edge < 3; edge++)
    {
        double xs[2] {axis_x_values[0], axis_x_values[edge+1]};
        double ys[2] {axis_y_values[0], axis_y_values[edge+1]};
        
        ImVec4 my_color; 
        my_color.w = 1.0f; 
        my_color.x = node_colors[edge+1].x;
        my_color.y = node_colors[edge+1].y;
        my_color.z = node_colors[edge+1].z;

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, my_color);
        ImPlot::SetNextLineStyle(my_color, 3.0);
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);
    }
    
}

#endif