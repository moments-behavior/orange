#pragma once
#include "camera.h"
#include "imgui.h" // ImVec2, ImVector
#include "video_capture.h"
#include <cmath> // fmodf

struct ScrollingBuffer {
    int MaxSize;
    int Offset;
    ImVector<ImVec2> Data;
    ScrollingBuffer(int max_size = 2000) {
        MaxSize = max_size;
        Offset = 0;
        Data.reserve(MaxSize);
    }
    void AddPoint(float x, float y) {
        if (Data.size() < MaxSize)
            Data.push_back(ImVec2(x, y));
        else {
            Data[Offset] = ImVec2(x, y);
            Offset = (Offset + 1) % MaxSize;
        }
    }
    void Erase() {
        if (Data.size() > 0) {
            Data.shrink(0);
            Offset = 0;
        }
    }
};

// utility structure for realtime plot
struct RollingBuffer {
    float Span;
    ImVector<ImVec2> Data;
    RollingBuffer() {
        Span = 10.0f;
        Data.reserve(2000);
    }
    void AddPoint(float x, float y) {
        float xmod = fmodf(x, Span);
        if (!Data.empty() && xmod < Data.back().x)
            Data.shrink(0);
        Data.push_back(ImVec2(xmod, y));
    }
};

inline void open_selected_cameras(const std::vector<bool> &check, int cam_count,
                                  GigEVisionDeviceInfo *device_info,
                                  std::vector<std::string> &camera_config_files,
                                  int &num_cameras,
                                  CameraParams *&cameras_params,
                                  CameraEachSelect *&cameras_select,
                                  CameraEmergent *&ecams,
                                  ScrollingBuffer *&realtime_plot_data) {

    num_cameras = 0;
    for (int i = 0; i < cam_count; i++) {
        if (check[i]) {
            num_cameras++;
        }
    }
    if (num_cameras > 0) {
        cameras_params = new CameraParams[num_cameras]();
        cameras_select = new CameraEachSelect[num_cameras]();

        std::vector<int> selected_cameras;
        for (int i = 0; i < cam_count; i++) {
            if (check[i]) {
                selected_cameras.push_back(i);
            }
        }
        for (int i = 0; i < num_cameras; i++) {
            set_camera_params(&cameras_params[i], &cameras_select[i],
                              &device_info[selected_cameras[i]],
                              camera_config_files, selected_cameras[i],
                              num_cameras);
        }

        for (int i = 0; i < num_cameras; i++) {
            cameras_select[i].stream_on = false;
            if (cameras_params[i].camera_name == "Cam16") {
                cameras_select[i].stream_on = true;
                cameras_select[i].detect_mode = Detect2D_GLThread;
            }
            if (cameras_params[i].camera_name == "shelter") {
                cameras_select[i].stream_on = true;
            }
        }

        ecams = new CameraEmergent[num_cameras];
        for (int i = 0; i < num_cameras; i++) {
            open_camera_with_params(&ecams[i].camera,
                                    &device_info[cameras_params[i].camera_id],
                                    &cameras_params[i]);
        }

        realtime_plot_data = new ScrollingBuffer[num_cameras];
    }
}
