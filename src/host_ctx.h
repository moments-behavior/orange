#pragma once
#include "gui.h"
#include "plot_buffers.h"
#include "video_capture.h"
#include <string>
#include <vector>

struct HostOpenCtx {
    std::string *selected_network_folder = nullptr;
    GigEVisionDeviceInfo *device_info = nullptr; // array base (owned elsewhere)
    int *cam_count = nullptr;
    std::vector<bool> *check = nullptr;
    int *num_cameras = nullptr;
    CameraParams **cameras_params = nullptr;        // will point to new[]
    CameraEachSelect **cameras_select = nullptr;    // will point to new[]
    CameraEmergent **ecams = nullptr;               // will point to new[]
    ScrollingBuffer **realtime_plot_data = nullptr; // will point to new[]
    CameraControl *camera_control = nullptr;
};

struct HostStartThreadCtx {
    std::thread *detection3d_thread;
    std::string *calib_yaml_folder = nullptr;
    bool *ptp_stream_sync = nullptr;
    std::string *input_folder = nullptr;
    std::vector<std::thread> *camera_threads = nullptr;
    PTPParams *ptp_params = nullptr;
    CameraControl *camera_control = nullptr;
    std::string *encoder_codec = nullptr;
    std::string *encoder_preset = nullptr;
    int *num_cameras = nullptr;
    int *evt_buffer_size = nullptr;
    int *display_gpu_id = nullptr;
    GL_Texture **tex_gl = nullptr;
    CameraParams **cameras_params = nullptr;     // will point to new[]
    CameraEachSelect **cameras_select = nullptr; // will point to new[]
    CameraEmergent **ecams = nullptr;            // will point to new[]
};
