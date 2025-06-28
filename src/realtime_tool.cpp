#include "realtime_tool.h"

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
