#ifndef ORANGE_REALTIME_GUI
#define ORANGE_REALTIME_GUI
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
    Mat camera_matrix;
    Mat dist_coeff;
    Mat tc_ext;
    Mat rc_ext;
};


struct CalibData
{
    vector<vector<vector<Point2f>>> imagePoints;
};

void aruco_detection(PictureBuffer* display_buffer) 
{    
    cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, display_buffer->frame).reshape(3, 2200);
    aruco::MarkerDetector MDetector;
    // detect 
    std::vector<aruco::Marker> markers = MDetector.detect(view);
    for (size_t i = 0; i < markers.size(); i++) {
        std::cout << markers[i] << std::endl;
        markers[i].draw(view);
    } 
}

void load_camera_calibration_results(CameraCalibResults* calib_results, CameraParams *cameras_params) {
    std::string calibration_file = "/home/user/Calibration/world/calibration_aruco/Cam" + std::to_string(cameras_params->camera_id) + ".yaml";
    FileStorage fs(calibration_file, FileStorage::READ);
    if (!fs.isOpened())
    {
        std::cout << "Could not open the calibration file: \"" << calibration_file << "\"" << std::endl;
    }
    fs["camera_matrix"] >> calib_results->camera_matrix;
    fs["distortion_coefficients"] >> calib_results->dist_coeff;
    fs["tc_ext"] >> calib_results->tc_ext;
    fs["rc_ext"] >> calib_results->rc_ext;
    fs.release();
}

void print_camera_calibration(CameraCalibResults* calib_results, CameraParams *cameras_params) {
    std::cout << "Cam" << std::to_string(cameras_params->camera_id) << std::endl;
    std::cout << "camera_matrix:  " << std::endl << " "  << calib_results->camera_matrix << std::endl << std::endl;
    std::cout << "dist_coeff:  " << std::endl << " "  << calib_results->dist_coeff << std::endl << std::endl;
    std::cout << "tc_ext:  " << std::endl << " "  << calib_results->tc_ext << std::endl << std::endl;
    std::cout << "rc_ext:  " << std::endl << " "  << calib_results->rc_ext << std::endl << std::endl;
}


// static void reprojection(std::vector<CameraParams> camera_params, int num_cams)
// {
   

//     if (num_views_labeled >= 2){
        
//         std::vector<cv::Mat> sfmPoints2d;
//         std::vector<cv::Mat> projection_matrices;
//         cv::Mat output;

//         for (u32 view_idx = 0; view_idx < num_cams; view_idx++)
//         {
//             if(keypoints->keypoints2d[view_idx][node].is_labeled)
//             {
//                 cv::Mat point = (cv::Mat_<float>(2, 1) << keypoints->keypoints2d[view_idx][node].position.x, (float)2200 - keypoints->keypoints2d[view_idx][node].position.y);
//                 cv::Mat pointUndistort;
//                 cv::undistortPoints(point, pointUndistort, camera_params[view_idx].k, camera_params[view_idx].dist_coeffs, cv::noArray(), camera_params[view_idx].k);
                
//                 sfmPoints2d.push_back(pointUndistort.reshape(1, 2));
//                 projection_matrices.push_back(camera_params[view_idx].projection_mat);
//             }
//         }

//         cv::sfm::triangulatePoints(sfmPoints2d, projection_matrices, output);
//         output.convertTo(output, CV_32F);

//         keypoints->keypoints3d[node].x = output.at<float>(0);
//         keypoints->keypoints3d[node].y = output.at<float>(1);
//         keypoints->keypoints3d[node].z = output.at<float>(2);

//         for (u32 view_idx = 0; view_idx < num_cams; view_idx++)
//         {
//             cv::Mat imagePts;
//             cv::projectPoints(output, camera_params[view_idx].rvec, camera_params[view_idx].tvec, camera_params[view_idx].k, camera_params[view_idx].dist_coeffs, imagePts);
//             double x = imagePts.at<float>(0, 0);
//             double y = float(2200) - imagePts.at<float>(0, 1);
//             keypoints->keypoints2d[view_idx][node].position.x = x;
//             keypoints->keypoints2d[view_idx][node].position.y = y;
//             keypoints->keypoints2d[view_idx][node].is_labeled = true;
//         }
//     }

// }



// void calibration_window(CPURender *cpu_buffers[], Settings *calib_setting, CameraControl *camera_control, CameraParams *cameras_params, u32 num_cameras, CameraCalibResults *calib_results, CalibData *calib_data)
// {
//     if (ImGui::Begin("Calibration"))
//     {
//         if (ImGui::Button("Load config file"))
//         {

//             const std::string inputSettingsFile = "/home/user/src/orange/circle.xml";
//             FileStorage fs(inputSettingsFile, FileStorage::READ);
//             if (!fs.isOpened())
//             {
//                 std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
//             }
//             fs["Settings"] >> *calib_setting;
//             fs.release();

//             if (!calib_setting->goodInput)
//             {
//                 cout << "Invalid input detected. Application stopping. " << endl;
//             }
//             else
//             {
//                 std::cout << "Calibration configuration file loaded: \"" << inputSettingsFile << "\"" << std::endl;
//             }

//             camera_control->calibration = true;

//             for (int i = 0; i < num_cameras; i++)
//             {
//                 vector<vector<Point2f>> image_points_per_cam;
//                 calib_data->imagePoints.push_back(image_points_per_cam);
//             }

//             for (int i = 0; i < num_cameras; i++)
//             {

//                 Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
//                 // initialization
//                 if (!calib_setting->intrinsicGuess)
//                 {
//                     if (!calib_setting->useFisheye && calib_setting->flag & CALIB_FIX_ASPECT_RATIO)
//                         cameraMatrix.at<double>(0, 0) = calib_setting->aspectRatio;
//                 }
//                 else
//                 {
//                     cameraMatrix = (Mat_<double>(3, 3) << 2800, 0, 1100, 0, 2800, 1600, 0, 0, 1);
//                 }
//                 calib_results->camera_matrices.push_back(cameraMatrix);

//                 if (calib_setting->useFisheye)
//                 {
//                     Mat distCoeffs = Mat::zeros(4, 1, CV_64F);
//                     calib_results->dist_coeffs.push_back(distCoeffs);
//                 }
//                 else
//                 {
//                     Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
//                     calib_results->dist_coeffs.push_back(distCoeffs);
//                 }
//             }
//         }

//         if (camera_control->calibration)
//         {
//             if (ImGui::Button("Detect"))
//             {
//                 for (int i = 0; i < num_cameras; i++)
//                 {
//                     cpu_buffers[i]->display_buffer.available_to_write = false;
//                 }

//                 for (int i = 0; i < num_cameras; i++)
//                 {

//                     int winSize = 11; // Half of search window for cornerSubPix
//                     // local the frame and process frame
//                     cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i]->display_buffer.frame).reshape(3, 2200);

//                     //! [find_pattern]
//                     vector<Point2f> pointBuf;
//                     bool found;
//                     int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

//                     switch (calib_setting->calibrationPattern) // Find feature points on the input format
//                     {
//                     case Settings::CHESSBOARD:
//                         found = findChessboardCorners(view, calib_setting->boardSize, pointBuf);
//                         break;
//                     case Settings::CIRCLES_GRID:
//                         found = findCirclesGrid(view, calib_setting->boardSize, pointBuf);
//                         break;
//                     case Settings::ASYMMETRIC_CIRCLES_GRID:
//                         found = findCirclesGrid(view, calib_setting->boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
//                         std::cout << "here?" << std::endl;
//                         break;
//                     default:
//                         found = false;
//                         break;
//                     }

//                     std::cout << "\n after finding corner?:" << found << std::endl;
//                     //! [find_pattern]
//                     //! [pattern_found]
//                     if (found) // If done with success,
//                     {
//                         // improve the found corners' coordinate accuracy for chessboard
//                         if (calib_setting->calibrationPattern == Settings::CHESSBOARD)
//                         {
//                             Mat viewGray;
//                             cvtColor(view, viewGray, COLOR_BGR2GRAY);
//                             cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
//                                          Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
//                         }

//                         calib_data->imagePoints[i].push_back(pointBuf);
//                         std::cout << pointBuf << std::endl;
//                         // Draw the corners.
//                         drawChessboardCorners(view, calib_setting->boardSize, Mat(pointBuf), found);
//                         bitwise_not(view, view);
//                     }
//                 }
//             }

//             if (ImGui::Button("Get new frame"))
//             {
//                 for (int i = 0; i < num_cameras; i++)
//                 {
//                     cpu_buffers[i]->display_buffer.available_to_write = true;
//                 }
//             }

//             for (int i = 0; i < num_cameras; i++)
//             {
//                 int no_frames = calib_data->imagePoints[i].size();
//                 std::string no_frames_str = "Number of Frames: " + std::to_string(no_frames);
//                 if (no_frames < 25)
//                 {
//                     ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), no_frames_str.c_str());
//                 }
//                 else
//                 {
//                     ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), no_frames_str.c_str());
//                 }
//             }

//             if (ImGui::Button("Run calibration"))
//             {
//                 float grid_width = calib_setting->squareSize * (calib_setting->boardSize.width - 1);
//                 Size imageSize = cv::Size(2200, 3200);
//                 cout << "imageSize" << imageSize << endl;

//                 for (int i = 0; i < num_cameras; i++)
//                 {
//                     string cam_calib_out = "Cam" + std::to_string(cameras_params[i].camera_id) + ".xml";
//                     if (runCalibrationAndSave(cam_calib_out, *calib_setting, imageSize, calib_results->camera_matrices[i], calib_results->dist_coeffs[i], calib_data->imagePoints[i], grid_width, false))
//                     {
//                         printf("Calibrated");
//                     }
//                 }
//             }

//             if (ImGui::Button("Load intrinsics"))
//             {
//                 for (int i = 0; i < num_cameras; i++)
//                 {
//                     string input_intrinsic_files = "Cam" + std::to_string(cameras_params[i].camera_id) + ".xml";
//                     loadIntrinsics(input_intrinsic_files, calib_results->camera_matrices[i], calib_results->dist_coeffs[i]);
//                 }
//             }

//             if (ImGui::Button("Estimate camera pose"))
//             {

//                 // detect
//                 for (int i = 0; i < num_cameras; i++)
//                 {
//                     cpu_buffers[i]->display_buffer.available_to_write = false;
//                 }

//                 for (int i = 0; i < num_cameras; i++)
//                 {
//                     int winSize = 11; // Half of search window for cornerSubPix
//                     // local the frame and process frame
//                     cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i]->display_buffer.frame).reshape(3, 2200);

//                     //! [find_pattern]
//                     vector<Point2f> pointBuf;
//                     bool found;
//                     int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

//                     switch (calib_setting->calibrationPattern) // Find feature points on the input format
//                     {
//                     case Settings::CHESSBOARD:
//                         found = findChessboardCorners(view, calib_setting->boardSize, pointBuf);
//                         break;
//                     case Settings::CIRCLES_GRID:
//                         found = findCirclesGrid(view, calib_setting->boardSize, pointBuf);
//                         break;
//                     case Settings::ASYMMETRIC_CIRCLES_GRID:
//                         found = findCirclesGrid(view, calib_setting->boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
//                         std::cout << "here?" << std::endl;
//                         break;
//                     default:
//                         found = false;
//                         break;
//                     }

//                     std::cout << "\n after finding corner?:" << found << std::endl;
//                     //! [find_pattern]
//                     //! [pattern_found]
//                     if (found) // If done with success,
//                     {
//                         // improve the found corners' coordinate accuracy for chessboard
//                         if (calib_setting->calibrationPattern == Settings::CHESSBOARD)
//                         {
//                             Mat viewGray;
//                             cvtColor(view, viewGray, COLOR_BGR2GRAY);
//                             cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
//                                          Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
//                         }
//                         calib_data->imagePoints[i].push_back(pointBuf);
//                         // Draw the corners.
//                         drawChessboardCorners(view, calib_setting->boardSize, Mat(pointBuf), found);
//                         bitwise_not(view, view);

//                         string cam_calib_estrinsics = "Cam" + std::to_string(cameras_params[i].camera_id) + "_extrinsics.xml";
//                         // estimate extrinsics
//                         if (estimatePose(cam_calib_estrinsics, *calib_setting, pointBuf, calib_results->camera_matrices[i], calib_results->dist_coeffs[i], SOLVEPNP_ITERATIVE))
//                         {
//                             std::cout << "Extrinsics estimated successfully." << std::endl;
//                         }
//                     }
//                 }
//             }
//         }
//     }
//     ImGui::End();
// }

#endif