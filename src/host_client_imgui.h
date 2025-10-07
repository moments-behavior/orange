#pragma once
#include "enet_utils.h" // for AppContext
#include "gui.h"
#include "plot_buffers.h"
#include "video_capture.h"
#include <string>
#include <utility>
#include <vector>

struct HostClientCtx {
    int *network_config_select = nullptr;
    std::vector<std::string> *network_config_folders = nullptr;
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

    std::thread *detection3d_thread = nullptr;
    std::string *calib_yaml_folder = nullptr;
    std::string *input_folder = nullptr;
    std::vector<std::thread> *camera_threads = nullptr;
    PTPParams *ptp_params = nullptr;
    std::string *encoder_codec = nullptr;
    std::string *encoder_preset = nullptr;
    int *evt_buffer_size = nullptr;
    int *display_gpu_id = nullptr;
    GL_Texture **tex_gl = nullptr;
};

void host_client_start_net_thread(AppContext &ctx); // starts dispatcher thread
void host_client_stop_net_thread();                 // joins thread on shutdown

void host_client_set_step_in_tick(bool v); // optional mode switch
void host_client_init(
    AppContext &ctx, const std::vector<std::pair<std::string, int>> &endpoints);
void host_client_tick();
void host_client_draw_gui();
void set_host_client_ctx(HostClientCtx *ctx);
