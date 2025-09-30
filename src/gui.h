#ifndef ORANGE_GUI
#define ORANGE_GUI
#include "camera.h"
#include "detect3d.h"
#include "global.h"
#include "gx_helper.h"
#include "imgui.h"
#include "implot.h"
#include "realtime_tool.h"
#include "video_capture.h"
#include <iostream>
#include <math.h>
#include <thread>

struct GL_Texture {
    GLuint texture;
    GLuint pbo;
    cudaGraphicsResource_t cuda_resource;
    unsigned char *cuda_buffer;
    size_t cuda_pbo_storage_buffer_size;
    int num_channels;
};

void setup_texture(GL_Texture &tex, int width, int height);

void upload_texture_from_pbo(GL_Texture &tex, int width, int height);

void clear_upload_and_cleanup(GL_Texture &tex, int width, int height);

void start_camera_streaming(
    std::vector<std::thread> &camera_threads, CameraControl *camera_control,
    CameraEmergent *ecams, CameraParams *cameras_params,
    CameraEachSelect *cameras_select, GL_Texture *tex, int num_cameras,
    int evt_buffer_size, bool ptp_stream_sync, const std::string &encoder_setup,
    const std::string &folder_name, PTPParams *ptp_params,
    std::string calib_yaml_folder, std::thread &detection3d_thread);

void stop_camera_streaming(std::vector<std::thread> &camera_threads,
                           CameraControl *camera_control, CameraEmergent *ecams,
                           CameraParams *cameras_params,
                           CameraEachSelect *cameras_select,
                           const int num_cameras, const int evt_buffer_size,
                           PTPParams *ptp_params,
                           std::thread &detection3d_thread);

bool input_text(const char *label, std::string &str, ImGuiInputTextFlags flags);

void HelpMarker(const char *desc);

void set_camera_properties(CameraEmergent *ecams, CameraParams *cameras_params,
                           CameraEachSelect *cameras_select,
                           const int num_cameras,
                           std::vector<std::string> &color_temps);

void gui_plot_world_coordinates(CameraCalibResults *cvp,
                                CameraParams *camera_params);

void draw_aruco_markers(Aruco2d *aruco_marker, int frame_height);

void draw_ball_center(cv::Point2f ball_center, int frame_height, ImVec4 color,
                      std::string name, ImPlotMarker marker, float pt_size);

void draw_box(cv::Rect_<float> bbox, int frame_height, ImVec4 color,
              std::string name, ImPlotMarker marker, float pt_size);

void draw_boxes(std::vector<cv::Rect_<float>> bboxes, int frame_height,
                ImVec4 color, std::string name, ImPlotMarker marker,
                float pt_size);

#endif
