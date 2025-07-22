#include "realtime_tool.h"
#include <filesystem>
#include <iostream>

std::string cvmat_type2str(int type) {
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth) {
    case CV_8U:
        r = "8U";
        break;
    case CV_8S:
        r = "8S";
        break;
    case CV_16U:
        r = "16U";
        break;
    case CV_16S:
        r = "16S";
        break;
    case CV_32S:
        r = "32S";
        break;
    case CV_32F:
        r = "32F";
        break;
    case CV_64F:
        r = "64F";
        break;
    default:
        r = "User";
        break;
    }

    r += "C";
    r += (chans + '0');

    return r;
}

void print_calibration_results(CameraCalibResults *calib_results) {
    std::cout << "k = " << std::endl
              << cv::format(calib_results->k, cv::Formatter::FMT_PYTHON)
              << std::endl
              << cvmat_type2str(calib_results->k.type()) << std::endl
              << std::endl;
    std::cout << "dist_coeffs  = " << std::endl
              << cv::format(calib_results->dist_coeffs,
                            cv::Formatter::FMT_PYTHON)
              << std::endl
              << cvmat_type2str(calib_results->dist_coeffs.type()) << std::endl
              << std::endl;
    std::cout << "r = " << std::endl
              << cv::format(calib_results->r, cv::Formatter::FMT_PYTHON)
              << std::endl
              << cvmat_type2str(calib_results->r.type()) << std::endl
              << std::endl;
    std::cout << "tvec = " << std::endl
              << cv::format(calib_results->tvec, cv::Formatter::FMT_PYTHON)
              << std::endl
              << cvmat_type2str(calib_results->tvec.type()) << std::endl
              << std::endl;
    std::cout << "rvec = " << std::endl
              << cv::format(calib_results->rvec, cv::Formatter::FMT_PYTHON)
              << std::endl
              << cvmat_type2str(calib_results->rvec.type()) << std::endl
              << std::endl;
    std::cout << "projection_mat = " << std::endl
              << cv::format(calib_results->projection_mat,
                            cv::Formatter::FMT_PYTHON)
              << std::endl
              << cvmat_type2str(calib_results->projection_mat.type())
              << std::endl
              << std::endl;
}

bool load_camera_calibration_results(std::string calibration_file,
                                     CameraCalibResults *calib_results) {

    if (!std::filesystem::exists(calibration_file)) {
        std::cout << "Calibration file does not exist: " << calibration_file
                  << std::endl;
        return false;
    }
    cv::FileStorage fs(calibration_file, cv::FileStorage::READ);
    if (!fs.isOpened()) {
        std::cout << "Could not open the calibration file: \""
                  << calibration_file << "\"" << std::endl;
        return false;
    }
    fs["camera_matrix"] >> calib_results->k;
    fs["distortion_coefficients"] >> calib_results->dist_coeffs;
    fs["tc_ext"] >> calib_results->tvec;
    fs["rc_ext"] >> calib_results->r;
    fs.release();
    cv::Rodrigues(calib_results->r, calib_results->rvec);
    cv::sfm::projectionFromKRt(calib_results->k, calib_results->r,
                               calib_results->tvec,
                               calib_results->projection_mat);
    return true;
}

std::vector<cv::Point2d> project3d_to_2d(const std::vector<cv::Point3d> &points,
                                         CameraCalibResults *camera_calib) {
    std::vector<cv::Point2d> img_pts;
    if (points.size() > 0) {
        cv::projectPoints(points, camera_calib->rvec, camera_calib->tvec,
                          camera_calib->k, camera_calib->dist_coeffs, img_pts);
    }
    return img_pts;
}

std::vector<cv::Point3d>
unproject2d_to_3d(const std::vector<cv::Point2d> &points,
                  const std::vector<double> &Z,
                  CameraCalibResults *camera_calib) {
    double f_x = camera_calib->k.at<double>(0, 0);
    double f_y = camera_calib->k.at<double>(1, 1);
    double c_x = camera_calib->k.at<double>(0, 2);
    double c_y = camera_calib->k.at<double>(1, 2);

    std::vector<cv::Point2d> points_undistorted;
    assert(Z.size() == 1 || Z.size() == points.size());
    if (!points.empty()) {
        cv::undistortPoints(points, points_undistorted, camera_calib->k,
                            camera_calib->dist_coeffs, cv::noArray(),
                            camera_calib->k);
    }

    std::vector<cv::Point3d> temp_pts;
    temp_pts.reserve(points.size());
    for (size_t idx = 0; idx < points_undistorted.size(); ++idx) {
        temp_pts.push_back(cv::Point3d((points_undistorted[idx].x - c_x) / f_x,
                                       (points_undistorted[idx].y - c_y) / f_y,
                                       1.0));
    }

    cv::Mat temp_pts_mat(3, temp_pts.size(), CV_64FC1, temp_pts.data());
    cv::Mat r_transpose = camera_calib->r;

    cv::Mat left_eqn = r_transpose * temp_pts_mat;
    cv::Mat right_eqn0 = r_transpose * camera_calib->tvec;

    std::vector<cv::Point3d> z_camera;
    for (size_t idx = 0; idx < points_undistorted.size(); ++idx) {
        const double z = Z.size() == 1 ? Z[0] : Z[idx];
        double temp_z =
            (z + right_eqn0.at<double>(2, 0)) / left_eqn.at<double>(2, idx);
        z_camera.push_back(temp_pts[idx] * temp_z);
    }

    cv::Mat z_camera_mat(3, temp_pts.size(), CV_64FC1, z_camera.data());
    cv::Mat pts_world = r_transpose * (z_camera_mat - camera_calib->tvec);

    std::vector<cv::Point3d> results;
    // Convert cv::Mat to std::vector<cv::Point3d>
    for (int i = 0; i < pts_world.cols; i++) {
        double x = pts_world.at<double>(0, i);
        double y = pts_world.at<double>(1, i);
        double z = pts_world.at<double>(2, i);
        results.emplace_back(x, y, z);
    }
    return results;
}

void world_coordinates_projection_points(CameraCalibResults *cvp,
                                         double *axis_x_values,
                                         double *axis_y_values, float scale,
                                         CameraParams *camera_params) {
    std::vector<cv::Point3f> world_coordinates;
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(scale * 1.0f, 0.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, scale * 1.0f, 0.0f));
    world_coordinates.push_back(cv::Point3f(0.0f, 0.0f, scale * 1.0f));

    std::vector<cv::Point2f> img_pts;
    cv::projectPoints(world_coordinates, cvp->rvec, cvp->tvec, cvp->k,
                      cvp->dist_coeffs, img_pts);

    for (int i = 0; i < 4; i++) {
        axis_x_values[i] = img_pts.at(i).x;
        axis_y_values[i] = camera_params->height - img_pts.at(i).y;
    }
}

cv::Mat triangulate_points(std::vector<cv::Point2f> image_points,
                           std::vector<CameraCalibResults *> calib_results) {
    std::vector<cv::Mat> sfm_points2d;
    std::vector<cv::Mat> projection_matrices;
    cv::Mat output3d;
    for (int i = 0; i < calib_results.size(); i++) {
        cv::Mat point =
            (cv::Mat_<float>(2, 1) << image_points[i].x, image_points[i].y);
        cv::Mat pointUndistort;
    }

    for (int i = 0; i < calib_results.size(); i++) {
        cv::Mat point =
            (cv::Mat_<float>(2, 1) << image_points[i].x, image_points[i].y);
        cv::Mat pointUndistort;

        cv::undistortPoints(point, pointUndistort, calib_results[i]->k,
                            calib_results[i]->dist_coeffs, cv::noArray(),
                            calib_results[i]->k);
        sfm_points2d.push_back(pointUndistort.reshape(1, 2));
        projection_matrices.push_back(calib_results[i]->projection_mat);
    }
    cv::sfm::triangulatePoints(sfm_points2d, projection_matrices, output3d);
    output3d.convertTo(output3d, CV_32F);
    return output3d;
}

void marker3d_to_pose(Aruco3d *aruco_maker_3d) {
    aruco_maker_3d->t_vec =
        aruco_maker_3d->corners[0] + aruco_maker_3d->corners[1] +
        aruco_maker_3d->corners[2] + aruco_maker_3d->corners[3];
    aruco_maker_3d->t_vec = aruco_maker_3d->t_vec / 4.0;
    // std::cout << aruco_maker_3d->t_vec << std::endl;
    cv::Point3f corner1to4 =
        aruco_maker_3d->corners[3] - aruco_maker_3d->corners[0];
    cv::Point3f corner1to2 =
        aruco_maker_3d->corners[1] - aruco_maker_3d->corners[0];
    aruco_maker_3d->normal = corner1to4.cross(corner1to2);
    aruco_maker_3d->normal =
        aruco_maker_3d->normal / cv::norm(aruco_maker_3d->normal);

    aruco_maker_3d->angle_x_axis = atan2(corner1to4.y, corner1to4.x);
    f32 result = aruco_maker_3d->angle_x_axis * 180 / PI;
    // printf("The marker is %f degrees from world x-axis. \n",  result);
}

bool find_marker3d(TriangulateMultiplePoints *aruco_marker_2d,
                   std::vector<CameraCalibResults *> &calib_results,
                   Aruco3d *marker3d) {
    int num_detected_cams = aruco_marker_2d->detected_cameras.size();
    if (num_detected_cams >= 2) {
        // triangulate
        std::vector<CameraCalibResults *> calib_results_all;
        for (size_t i = 0; i < num_detected_cams; i++) {
            calib_results_all.push_back(
                calib_results[aruco_marker_2d->detected_cameras[i]]);
        }

        for (size_t i = 0; i < 4; i++) {
            std::vector<cv::Point2f> image_points_all;
            for (size_t j = 0; j < num_detected_cams; j++) {
                image_points_all.push_back(
                    aruco_marker_2d->detected_points[j][i]);
            }
            cv::Mat output3d =
                triangulate_points(image_points_all, calib_results_all);
            cv::Point3f pts3d = cv::Point3d(output3d);
            // cv::Point3f(output3d.at<float>(0), output3d.at<float>(1),
            // output3d.at<float>(2));
            marker3d->corners[i] = pts3d;
        }

    } else {
        return false;
    }

    // // print marker corners
    // for (size_t i = 0; i < 4; i++) {
    //     std::cout << marker3d->corners[i] << ", " << std::endl;
    // }

    marker3d_to_pose(marker3d);
    return true;
}

bool find_kp3d(TriangulatePoint *kp2d, Keypoints3d *kp3d) {
    int num_detected_cams = kp2d->detected_cameras.size();
    if (num_detected_cams >= 2) {
        // triangulate
        std::vector<cv::Point2f> image_points_all;
        for (size_t j = 0; j < num_detected_cams; j++) {
            image_points_all.push_back(kp2d->detected_points[j]);
            // print_calibration_results(ball_2d->calib_results[j]);
        }
        cv::Mat output3d =
            triangulate_points(image_points_all, kp2d->calib_results);
        cv::Point3f pts3d = cv::Point3d(output3d);
        // cv::Point3f(output3d.at<float>(0), output3d.at<float>(1),
        // output3d.at<float>(2));
        kp3d->pt = pts3d;
        kp3d->kp_id = kp2d->kp_id;
        kp3d->obj_id = kp2d->obj_id;
        // std::cout << "Ball: " << ball3d->center << std::endl;
    } else {
        return false;
    }
    return true;
}
