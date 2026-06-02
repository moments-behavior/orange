#ifndef ORANGE_GUI
#define ORANGE_GUI
#include "camera.h"
#include "global.h"
#include "gx_helper.h"
#include "imgui.h"
#include "implot.h"
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

void setup_texture(GL_Texture &tex, int width, int height);

void upload_texture_from_pbo(GL_Texture &tex, int width, int height);

void clear_upload_and_cleanup(GL_Texture &tex, int width, int height);

void start_camera_streaming(
    std::vector<std::thread> &camera_threads, CameraControl *camera_control,
    CameraEmergent *ecams, CameraParams *cameras_params,
    CameraEachSelect *cameras_select, GL_Texture *tex, int num_cameras,
    int evt_buffer_size, bool ptp_stream_sync, const std::string &encoder_setup,
    const std::string &folder_name, PTPParams *ptp_params);

void stop_camera_streaming(std::vector<std::thread> &camera_threads,
                           CameraControl *camera_control, CameraEmergent *ecams,
                           CameraParams *cameras_params,
                           CameraEachSelect *cameras_select,
                           const int num_cameras, const int evt_buffer_size,
                           PTPParams *ptp_params);

bool input_text(const char *label, std::string &str, ImGuiInputTextFlags flags);

void HelpMarker(const char *desc);

void set_camera_properties(CameraEmergent *ecams, CameraParams *cameras_params,
                           CameraEachSelect *cameras_select,
                           const int num_cameras,
                           std::vector<std::string> &color_temps);
#endif
