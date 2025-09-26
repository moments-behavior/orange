#pragma once
#include "plot_buffers.h"
#include "video_capture.h" // CameraParams, CameraEachSelect, CameraEmergent, ScrollingBuffer
#include <string>
#include <vector>
// GigEVisionDeviceInfo, CameraControl prototypes

// All fields are pointers so your existing storage/owners can live elsewhere.
// Pointers-to-pointers are used where your code allocates and assigns new
// arrays.
struct HostOpenCtx {
    std::string *selected_network_folder = nullptr;

    // camera inventory + selection
    GigEVisionDeviceInfo *device_info = nullptr; // array base (owned elsewhere)
    int *cam_count = nullptr;
    std::vector<bool> *check = nullptr;

    // outputs allocated by open_selected_cameras(...)
    int *num_cameras = nullptr;
    CameraParams **cameras_params = nullptr;        // will point to new[]
    CameraEachSelect **cameras_select = nullptr;    // will point to new[]
    CameraEmergent **ecams = nullptr;               // will point to new[]
    ScrollingBuffer **realtime_plot_data = nullptr; // will point to new[]

    // control block
    CameraControl *camera_control = nullptr;
};
