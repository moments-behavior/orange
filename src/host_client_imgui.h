#pragma once
#include "enet_utils.h" // for AppContext
#include "gui.h"
#include "plot_buffers.h"
#include "video_capture.h"
#include <string>
#include <utility>
#include <vector>

struct HostClientCtx {
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
void HostClient_StopNetThread();                    // joins thread on shutdown

void HostClient_SetStepInTick(bool v); // optional mode switch
void HostClient_Init(AppContext &ctx,
                     const std::vector<std::pair<std::string, int>> &endpoints);
void HostClient_Tick();
void HostClient_DrawImGui();
void set_host_client_ctx(HostClientCtx *ctx);
