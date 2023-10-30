#ifndef ORANGE_REALTIME_TOOL
#define ORANGE_REALTIME_TOOL
#include "imgui.h"
#include <opencv2/sfm.hpp>
#include "opencv2/core/core_c.h"
#include "opencv2/core/core.hpp"
#include "aruco_nano.h"
#include "types.h"
#include "camera.h"

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
    tuple_f** proj_corners;
    cv::Point3f t_vec;
    cv::Point3f normal; 
    f32 angle_x_axis;
    bool new_detection;
};

void print_calibration_results(CameraCalibResults* calib_results) {
    std::cout << "k = " << std::endl << cv::format(calib_results->k, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "dist_coeffs  = " << std::endl << cv::format(calib_results->dist_coeffs, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "r = " << std::endl << cv::format(calib_results->r, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "tvec = " << std::endl << cv::format(calib_results->tvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "rvec = " << std::endl << cv::format(calib_results->rvec, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
    std::cout << "projection_mat = " << std::endl << cv::format(calib_results->projection_mat, cv::Formatter::FMT_PYTHON) << std::endl << std::endl;
}

cv::Mat triangulate_points(std::vector<cv::Point2f> image_points, std::vector<CameraCalibResults*> calib_results)
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
    return output3d;
}


void aruco_detection(unsigned char* display_buffer, CameraParams *cameras_params, ArucoMarker2d* aruco_marker_2d) 
{    
    cv::Mat view = cv::Mat(cameras_params->width * cameras_params->height * 3, 1, CV_8U, display_buffer).reshape(3, cameras_params->height);
    aruconano::MarkerDetector MDetector;
    // detect 
    std::vector<aruconano::Marker> markers = MDetector.detect(view);
    for (size_t i = 0; i < markers.size(); i++) {
        // std::cout << markers[i] << std::endl;
        // markers[i].draw(view);

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
    std::cout << aruco_maker_3d->t_vec << std::endl;
    cv::Point3f corner1to4 = aruco_maker_3d->corners[3] - aruco_maker_3d->corners[0];
    cv::Point3f corner1to2 = aruco_maker_3d->corners[1] - aruco_maker_3d->corners[0];
    aruco_maker_3d->normal = corner1to4.cross(corner1to2);
    aruco_maker_3d->normal =  aruco_maker_3d->normal / cv::norm(aruco_maker_3d->normal);

    aruco_maker_3d->angle_x_axis = atan2(corner1to4.y, corner1to4.x);
    f32 result = aruco_maker_3d->angle_x_axis * 180 / PI;
    // printf("The marker is %f degrees from world x-axis. \n",  result);    

}


bool find_marker3d(ArucoMarker2d* aruco_marker_2d, ArucoMarker3d* aruco_maker_3d, CameraCalibResults* calib_results, int num_cameras)
{
    int num_detected_cams = aruco_marker_2d->detected_cameras.size();
    if (num_detected_cams >= 2) {
        // triangulate
        std::vector<CameraCalibResults*> calib_results_all; 
        for (size_t i = 0; i < num_detected_cams; i++) {
            calib_results_all.push_back(calib_results + i);
        }
 
        for (size_t i = 0; i < 4; i++) {
            std::vector<cv::Point2f> image_points_all;
            for (size_t j = 0; j < num_detected_cams; j++) {
                image_points_all.push_back(aruco_marker_2d->detected_points[j][i]);
            }
            cv::Mat output3d = triangulate_points(image_points_all, calib_results_all); 
            cv::Point3f pts3d = cv::Point3f(output3d.at<float>(0), output3d.at<float>(1), output3d.at<float>(2));
            aruco_maker_3d->corners[i] = pts3d;

            // reprojection
            for (size_t j = 0; j < num_cameras; j++) {
                cv::Mat image_pts;
                cv::projectPoints(output3d, calib_results[j].rvec, calib_results[j].tvec, calib_results[j].k, calib_results[j].dist_coeffs, image_pts);
                aruco_maker_3d->proj_corners[j][i].x = image_pts.at<float>(0, 0);
                aruco_maker_3d->proj_corners[j][i].y = image_pts.at<float>(0, 1);
            }
        }
        
    } else {
        return false;
    }

    // print marker corners
    // for (size_t i = 0; i < 4; i++) {
    //     std::cout << aruco_maker_3d->corners[i] << ", " << std::endl;
    // }

    marker3d_to_pose(aruco_maker_3d);
    return true;
}


std::map<unsigned int, cv::Point3f> get_3d_coordinates(std::vector<std::vector<cv::Rect>> bounding_boxes, std::vector<std::vector<int>> obj_ids, CameraCalibResults* CamParam)
{
    // points 
    std::map<unsigned int, std::vector<cv::Point2f>> mapOfObjects;
    std::map<unsigned int, std::vector<CameraCalibResults*>> mapOfCameras;
    
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
                std::vector<cv::Point2f> points_per_obj; 
                std::vector<CameraCalibResults*> camera_per_obj;
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
            cv::Mat output3d = triangulate_points(it->second, mapOfCameras[it->first]);  
            cv::Point3f point3d = cv::Point3f(output3d.at<float>(0), output3d.at<float>(1), output3d.at<float>(2));
            mapOfPoints3D.insert({it->first, point3d});
        }
    }
    return mapOfPoints3D;
}



bool load_camera_calibration_results(CameraCalibResults* calib_results, CameraParams *cameras_params) 
{
    std::string calibration_file = "/home/user/Calibration/4_edge_cams_serial/Calib_" + cameras_params->camera_serial + ".yaml";
    cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Could not open the calibration file: \"" << calibration_file << "\"" << std::endl;
        return false;
    }
    fs["camera_matrix"] >> calib_results->k;
    fs["distortion_coefficients"] >> calib_results->dist_coeffs;
    fs["tc_ext"] >> calib_results->tvec;
    fs["rc_ext"] >> calib_results->r;
    fs.release();
    cv::Rodrigues(calib_results->r, calib_results->rvec);
    cv::sfm::projectionFromKRt(calib_results->k, calib_results->r, calib_results->tvec, calib_results->projection_mat);
    return true;
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

static void draw_aruco_markers(ArucoMarker3d* aruco_marker, int camera_id)
{
    double x[5] = {(double)aruco_marker->proj_corners[camera_id][0].x, 
        (double)aruco_marker->proj_corners[camera_id][1].x, 
        (double)aruco_marker->proj_corners[camera_id][2].x, 
        (double)aruco_marker->proj_corners[camera_id][3].x, 
        (double)aruco_marker->proj_corners[camera_id][0].x};
    
    double y[5] = {(double)2200 - (double)aruco_marker->proj_corners[camera_id][0].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][1].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][2].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][3].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][0].y};
    
    ImPlot::SetNextLineStyle(ImVec4(1.0, 0.0, 1.0, 1.0), 3.0);
    ImPlot::PlotLine("##", &x[0], &y[0], 5); 
}


#endif