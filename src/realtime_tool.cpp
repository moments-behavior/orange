#include "realtime_tool.h"

std::string cvmat_type2str(int type)
{
    std::string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch (depth)
    {
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


void print_calibration_results(CameraCalibResults *calib_results)
{
    std::cout << "k = " << std::endl
              << cv::format(calib_results->k, cv::Formatter::FMT_PYTHON) << std::endl
              << cvmat_type2str(calib_results->k.type()) << std::endl
              << std::endl;
    std::cout << "dist_coeffs  = " << std::endl
              << cv::format(calib_results->dist_coeffs, cv::Formatter::FMT_PYTHON) << std::endl
              << cvmat_type2str(calib_results->dist_coeffs.type()) << std::endl
              << std::endl;
    std::cout << "r = " << std::endl
              << cv::format(calib_results->r, cv::Formatter::FMT_PYTHON) << std::endl
              << cvmat_type2str(calib_results->r.type()) << std::endl
              << std::endl;
    std::cout << "tvec = " << std::endl
              << cv::format(calib_results->tvec, cv::Formatter::FMT_PYTHON) << std::endl
              << cvmat_type2str(calib_results->tvec.type()) << std::endl              
              << std::endl;
    std::cout << "rvec = " << std::endl
              << cv::format(calib_results->rvec, cv::Formatter::FMT_PYTHON) << std::endl
              << cvmat_type2str(calib_results->rvec.type()) << std::endl                            
              << std::endl;
    std::cout << "projection_mat = " << std::endl
              << cv::format(calib_results->projection_mat, cv::Formatter::FMT_PYTHON) << std::endl
              << cvmat_type2str(calib_results->projection_mat.type()) << std::endl                                          
              << std::endl;
}


bool load_camera_calibration_results(std::string calibration_file, CameraCalibResults *calib_results)
{
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

// std::vector<cv::Point3f> unproject(std::vector<cv::Point2f> points2d, std::vector<cv::Point2f> z_coord, CameraCalibResults *camera_calib)
// {
//     f64 f_x = camera_calib->k.at<double>(0, 0);
//     f64 f_y = camera_calib->k.at<double>(1, 1);
//     f64 c_x = camera_calib->k.at<double>(0, 2);
//     f64 c_y = camera_calib->k.at<double>(1, 2);

//     // cv::Mat point_undistort;
//     // cv::undistortPoints(points2d, point_undistort, camera_calib->k, camera_calib->dist_coeffs, cv::noArray(), camera_calib->k);
// }