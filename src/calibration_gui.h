#ifndef ORANGE_CALIBRATION_GUI
#define ORANGE_CALIBRATION_GUI
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
    vector<Mat> camera_matrices;
    vector<Mat> dist_coeffs;
};

struct CalibData
{
    bool *selected_images_to_save;
    vector<vector<vector<Point2f>>> imagePoints;
    vector<int> image_save_index;
};

void calibration_window(CPURender *cpu_buffers[], Settings *calib_setting, CameraControl *camera_control, CameraParams *cameras_params, u32 num_cameras, CameraCalibResults *calib_results, CalibData *calib_data)
{
    if (ImGui::Begin("Calibration"))
    {
        if (ImGui::Button("Load config file"))
        {

            const std::string inputSettingsFile = "/home/user/src/orange/circle.xml";
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

            camera_control->calibration = true;

            for (int i = 0; i < num_cameras; i++)
            {
                vector<vector<Point2f>> image_points_per_cam;
                calib_data->imagePoints.push_back(image_points_per_cam);
            }

            for (int i = 0; i < num_cameras; i++)
            {

                Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
                // initialization
                if (!calib_setting->intrinsicGuess)
                {
                    if (!calib_setting->useFisheye && calib_setting->flag & CALIB_FIX_ASPECT_RATIO)
                        cameraMatrix.at<double>(0, 0) = calib_setting->aspectRatio;
                }
                else
                {
                    cameraMatrix = (Mat_<double>(3, 3) << 2800, 0, 1100, 0, 2800, 1600, 0, 0, 1);
                }
                calib_results->camera_matrices.push_back(cameraMatrix);

                if (calib_setting->useFisheye)
                {
                    Mat distCoeffs = Mat::zeros(4, 1, CV_64F);
                    calib_results->dist_coeffs.push_back(distCoeffs);
                }
                else
                {
                    Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
                    calib_results->dist_coeffs.push_back(distCoeffs);
                }
            }

            for (int i = 0; i < num_cameras; i++)
            {
                calib_data->image_save_index.push_back(0);
            }
            calib_data->selected_images_to_save = new bool[num_cameras];
            for (int i = 0; i < num_cameras; i++)
            {
                calib_data->selected_images_to_save[i] = false;
            }
        }

        if (camera_control->calibration)
        {

            if (ImGui::Button("Save images all"))
            {

                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = false;
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i]->display_buffer.frame).reshape(3, 2200);
                    string image_name = "/home/user/Calibration/world/Cam" + std::to_string(cameras_params[i].camera_id) + "/image_" + std::to_string(calib_data->image_save_index[i]) + ".tiff";
                    cv::imwrite(image_name, view);
                    calib_data->image_save_index[i]++;
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = true;
                }
            }

            for (int i = 0; i < num_cameras; i++)
            {
                ImGui::InputInt("Saving image index: ", &calib_data->image_save_index[i]);
            }

            for (int i = 0; i < num_cameras; i++)
            {
                char label[32];
                sprintf(label, "Cam%d", i);
                ImGui::Checkbox(label, &calib_data->selected_images_to_save[i]);
                ImGui::SameLine();
            }

            if (ImGui::Button("Save selected"))
            {
                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = false;
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    if (calib_data->selected_images_to_save[i])
                    {
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i]->display_buffer.frame).reshape(3, 2200);
                        string image_name = "/home/user/Calibration/world/Cam" + std::to_string(cameras_params[i].camera_id) + "/image_" + std::to_string(calib_data->image_save_index[i]) + ".tiff";
                        cv::imwrite(image_name, view);
                        calib_data->image_save_index[i]++;
                    }
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = true;
                }
            }

            if (ImGui::Button("Detect"))
            {
                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = false;
                }

                for (int i = 0; i < num_cameras; i++)
                {

                    int winSize = 11; // Half of search window for cornerSubPix
                    // local the frame and process frame
                    cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i]->display_buffer.frame).reshape(3, 2200);

                    //! [find_pattern]
                    vector<Point2f> pointBuf;
                    bool found;
                    int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

                    switch (calib_setting->calibrationPattern) // Find feature points on the input format
                    {
                    case Settings::CHESSBOARD:
                        found = findChessboardCorners(view, calib_setting->boardSize, pointBuf);
                        break;
                    case Settings::CIRCLES_GRID:
                        found = findCirclesGrid(view, calib_setting->boardSize, pointBuf);
                        break;
                    case Settings::ASYMMETRIC_CIRCLES_GRID:
                        found = findCirclesGrid(view, calib_setting->boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
                        std::cout << "here?" << std::endl;
                        break;
                    default:
                        found = false;
                        break;
                    }

                    std::cout << "\n after finding corner?:" << found << std::endl;
                    //! [find_pattern]
                    //! [pattern_found]
                    if (found) // If done with success,
                    {
                        // improve the found corners' coordinate accuracy for chessboard
                        if (calib_setting->calibrationPattern == Settings::CHESSBOARD)
                        {
                            Mat viewGray;
                            cvtColor(view, viewGray, COLOR_BGR2GRAY);
                            cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
                                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
                        }

                        calib_data->imagePoints[i].push_back(pointBuf);
                        std::cout << pointBuf << std::endl;
                        // Draw the corners.
                        drawChessboardCorners(view, calib_setting->boardSize, Mat(pointBuf), found);
                        bitwise_not(view, view);
                    }
                }
            }

            if (ImGui::Button("Get new frame"))
            {
                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = true;
                }
            }

            for (int i = 0; i < num_cameras; i++)
            {
                int no_frames = calib_data->imagePoints[i].size();
                std::string no_frames_str = "Number of Frames: " + std::to_string(no_frames);
                if (no_frames < 25)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), no_frames_str.c_str());
                }
                else
                {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), no_frames_str.c_str());
                }
            }

            if (ImGui::Button("Run calibration"))
            {
                float grid_width = calib_setting->squareSize * (calib_setting->boardSize.width - 1);
                Size imageSize = cv::Size(2200, 3200);
                cout << "imageSize" << imageSize << endl;

                for (int i = 0; i < num_cameras; i++)
                {
                    string cam_calib_out = "Cam" + std::to_string(cameras_params[i].camera_id) + ".xml";
                    if (runCalibrationAndSave(cam_calib_out, *calib_setting, imageSize, calib_results->camera_matrices[i], calib_results->dist_coeffs[i], calib_data->imagePoints[i], grid_width, false))
                    {
                        printf("Calibrated");
                    }
                }
            }

            if (ImGui::Button("Load intrinsics"))
            {
                for (int i = 0; i < num_cameras; i++)
                {
                    string input_intrinsic_files = "Cam" + std::to_string(cameras_params[i].camera_id) + ".xml";
                    loadIntrinsics(input_intrinsic_files, calib_results->camera_matrices[i], calib_results->dist_coeffs[i]);
                }
            }

            if (ImGui::Button("Estimate camera pose"))
            {

                // detect
                for (int i = 0; i < num_cameras; i++)
                {
                    cpu_buffers[i]->display_buffer.available_to_write = false;
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    int winSize = 11; // Half of search window for cornerSubPix
                    // local the frame and process frame
                    cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i]->display_buffer.frame).reshape(3, 2200);

                    //! [find_pattern]
                    vector<Point2f> pointBuf;
                    bool found;
                    int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

                    switch (calib_setting->calibrationPattern) // Find feature points on the input format
                    {
                    case Settings::CHESSBOARD:
                        found = findChessboardCorners(view, calib_setting->boardSize, pointBuf);
                        break;
                    case Settings::CIRCLES_GRID:
                        found = findCirclesGrid(view, calib_setting->boardSize, pointBuf);
                        break;
                    case Settings::ASYMMETRIC_CIRCLES_GRID:
                        found = findCirclesGrid(view, calib_setting->boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
                        std::cout << "here?" << std::endl;
                        break;
                    default:
                        found = false;
                        break;
                    }

                    std::cout << "\n after finding corner?:" << found << std::endl;
                    //! [find_pattern]
                    //! [pattern_found]
                    if (found) // If done with success,
                    {
                        // improve the found corners' coordinate accuracy for chessboard
                        if (calib_setting->calibrationPattern == Settings::CHESSBOARD)
                        {
                            Mat viewGray;
                            cvtColor(view, viewGray, COLOR_BGR2GRAY);
                            cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
                                         Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
                        }
                        calib_data->imagePoints[i].push_back(pointBuf);
                        // Draw the corners.
                        drawChessboardCorners(view, calib_setting->boardSize, Mat(pointBuf), found);
                        bitwise_not(view, view);

                        string cam_calib_estrinsics = "Cam" + std::to_string(cameras_params[i].camera_id) + "_extrinsics.xml";
                        // estimate extrinsics
                        if (estimatePose(cam_calib_estrinsics, *calib_setting, pointBuf, calib_results->camera_matrices[i], calib_results->dist_coeffs[i], SOLVEPNP_ITERATIVE))
                        {
                            std::cout << "Extrinsics estimated successfully." << std::endl;
                        }
                    }
                }
            }
        }
    }
    ImGui::End();
}

#endif