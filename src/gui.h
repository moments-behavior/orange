#ifndef ORANGE_GUI
#define ORANGE_GUI
#include "camera.h"
#include "global.h"
#include "gx_helper.h"
#include "video_capture.h"
#include <math.h>
#include <thread>

struct EncoderConfig {
    std::string encoder_codec;
    std::string encoder_preset;
    std::string folder_name;
};

struct GL_Texture {
    GLuint texture;
    GLuint pbo;
    cudaGraphicsResource_t cuda_resource;
    unsigned char *cuda_buffer;
    size_t cuda_pbo_storage_buffer_size;
    cudaStream_t streams;
    int num_channels;
};

void setup_texture(GL_Texture &tex, int width, int height) {
    cudaStreamCreate(&tex.streams);
    create_pbo(&tex.pbo, width, height);
    register_pbo_to_cuda(&tex.pbo, &tex.cuda_resource);
    map_cuda_resource(&tex.cuda_resource, tex.streams);
    cuda_pointer_from_resource(&tex.cuda_buffer,
                               &tex.cuda_pbo_storage_buffer_size,
                               &tex.cuda_resource);
    create_texture(&tex.texture, width, height);
}

void upload_texture_from_pbo(GL_Texture &tex, int width, int height) {
    bind_pbo(&tex.pbo);
    bind_texture(&tex.texture);
    upload_image_pbo_to_texture(width,
                                height); // Uses currently bound PBO and texture
    unbind_pbo();
    unbind_texture();
}

void clear_upload_and_cleanup(GL_Texture &tex, int width, int height) {
    // Clear the CUDA buffer
    int size_pic = width * height * sizeof(unsigned char) * 4;
    cudaMemset(tex.cuda_buffer, 0, size_pic);

    // Upload from PBO to texture
    upload_texture_from_pbo(tex, width, height);

    // Cleanup resources
    gx_delete_buffer(&tex.pbo);
    unmap_cuda_resource(&tex.cuda_resource);
    cuda_unregister_pbo(tex.cuda_resource);
    cudaStreamDestroy(tex.streams);
}

inline void start_camera_streaming(
    std::vector<std::thread> &camera_threads, CameraControl *camera_control,
    CameraEmergent *ecams, CameraParams *cameras_params,
    CameraEachSelect *cameras_select, GL_Texture *tex, int num_cameras,
    int evt_buffer_size, bool ptp_stream_sync, const std::string &encoder_setup,
    const std::string &folder_name, PTPParams *ptp_params,
    INDIGOSignalBuilder *indigo_signal_builder, const std::string &yolo_model) {
    for (int i = 0; i < num_cameras; i++) {
        camera_open_stream(&ecams[i].camera, &cameras_params[i]);
        ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                              &cameras_params[i], evt_buffer_size);

        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct) {
            allocate_frame_reorder_buffer(
                &ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
        }
    }

    if (ptp_stream_sync) {
        for (int i = 0; i < num_cameras; i++) {
            ptp_camera_sync(&ecams[i].camera, &cameras_params[i]);
        }
        camera_control->sync_camera = true;
    }

    if (camera_control->trigger_mode) {
        for (int i = 0; i < num_cameras; i++) {
            camera_trigger_mode(&ecams[i].camera, &cameras_params[i]);
        }
    }

    for (int i = 0; i < num_cameras; i++) {
        cameras_select[i].yolo_model = yolo_model.c_str();
    }

    for (int i = 0; i < num_cameras; i++) {
        camera_threads.emplace_back(
            &acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i],
            camera_control, tex[i].cuda_buffer, encoder_setup, folder_name,
            ptp_params, indigo_signal_builder);
    }
}

inline void
stop_camera_streaming(std::vector<std::thread> &camera_threads,
                      CameraControl *camera_control, CameraEmergent *ecams,
                      CameraParams *cameras_params,
                      CameraEachSelect *cameras_select, const int num_cameras,
                      const int evt_buffer_size, PTPParams *ptp_params) {
    for (auto &t : camera_threads)
        t.join();

    for (int i = 0; i < num_cameras; i++) {
        camera_threads.pop_back();
    }

    for (int i = 0; i < num_cameras; i++) {
        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                             evt_buffer_size, cameras_params);
        delete[] ecams[i].evt_frame;
        check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera),
                            cameras_params[i].camera_serial.c_str());
    }

    if (num_cameras > 1) {
        for (int i = 0; i < num_cameras; i++) {
            ptp_sync_off(&ecams[i].camera, cameras_params);
        }
        ptp_params->ptp_counter = 0;
        ptp_params->ptp_global_time = 0;
        camera_control->sync_camera = false;
    }
}

static void set_camera_properties(CameraEmergent *ecams,
                                  CameraParams *cameras_params,
                                  const int num_cameras,
                                  std::vector<std::string> &color_temps) {
    if (ImGui::TreeNode("Camera Property")) {
        static int selected_camera = 0;
        static int slider_gain, slider_exposure, slider_frame_rate,
            slider_width, slider_height, OffsetX, OffsetY, slider_focus,
            slider_iris;

        for (int n = 0; n < num_cameras; n++) {
            if (ImGui::Selectable(cameras_params[n].camera_name.c_str(),
                                  selected_camera == n)) {
                selected_camera = n;
            }
            slider_gain = cameras_params[selected_camera].gain;
            slider_iris = cameras_params[selected_camera].iris;
            slider_focus = cameras_params[selected_camera].focus;
            slider_width = cameras_params[selected_camera].width;
            slider_height = cameras_params[selected_camera].height;
            slider_exposure = cameras_params[selected_camera].exposure;
            slider_frame_rate = cameras_params[selected_camera].frame_rate;
            OffsetX = cameras_params[selected_camera].offsetx;
            OffsetY = cameras_params[selected_camera].offsety;
        }

        ImGui::Checkbox("GPU Direct",
                        &cameras_params[selected_camera].gpu_direct);
        ImGui::Checkbox("Color", &cameras_params[selected_camera].color);

        if (cameras_params[selected_camera].color) {
            auto it = std::find(color_temps.begin(), color_temps.end(),
                                cameras_params->color_temp);
            int item_current_idx = (it != color_temps.end())
                                       ? std::distance(color_temps.begin(), it)
                                       : 0;
            std::vector<const char *> item_cstrs;
            for (const auto &item : color_temps) {
                item_cstrs.push_back(item.c_str());
            }
            if (ImGui::Combo("Color Temp", &item_current_idx, item_cstrs.data(),
                             color_temps.size())) {
                update_color_temperature(&ecams[selected_camera].camera,
                                         color_temps[item_current_idx],
                                         &cameras_params[selected_camera]);
            }
        }

        if (ImGui::SliderInt("Width", &slider_width,
                             cameras_params[selected_camera].width_min,
                             cameras_params[selected_camera].width_max, "%d")) {
            slider_width = (slider_width / 16) * 16; // round to even number
            update_width_value(&ecams[selected_camera].camera, slider_width,
                               &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("Height", &slider_height,
                             cameras_params[selected_camera].height_min,
                             cameras_params[selected_camera].height_max,
                             "%d")) {
            slider_height = (slider_height / 16) * 16; // round to even number
            update_height_value(&ecams[selected_camera].camera, slider_height,
                                &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("OffsetX", &OffsetX,
                             cameras_params[selected_camera].offsetx_min,
                             cameras_params[selected_camera].offsetx_max,
                             "%d")) {
            // round to 16
            OffsetX = (OffsetX / 16) * 16; // round to even number
            update_offsetX_value(&ecams[selected_camera].camera, OffsetX,
                                 &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("OffsetY", &OffsetY,
                             cameras_params[selected_camera].offsety_min,
                             cameras_params[selected_camera].offsety_max,
                             "%d")) {
            // round to 16
            OffsetY = (OffsetY / 16) * 16; // round to even number
            update_offsetY_value(&ecams[selected_camera].camera, OffsetY,
                                 &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("Gain", &slider_gain,
                             cameras_params[selected_camera].gain_min,
                             cameras_params[selected_camera].gain_max, "%d")) {
            update_gain_value(&ecams[selected_camera].camera, slider_gain,
                              &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("Focus", &slider_focus,
                             cameras_params[selected_camera].focus_min,
                             cameras_params[selected_camera].focus_max, "%d")) {
            update_focus_value(&ecams[selected_camera].camera, slider_focus,
                               &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("Iris", &slider_iris,
                             cameras_params[selected_camera].iris_min,
                             cameras_params[selected_camera].iris_max, "%d")) {
            update_iris_value(&ecams[selected_camera].camera, slider_iris,
                              &cameras_params[selected_camera]);
        }

        if (ImGui::SliderInt("Exposure", &slider_exposure,
                             cameras_params[selected_camera].exposure_min,
                             cameras_params[selected_camera].exposure_max,
                             "%d")) {
            update_exposure_framerate_value(&ecams[selected_camera].camera,
                                            slider_exposure, &slider_frame_rate,
                                            &cameras_params[selected_camera]);
        }

        char label[32];
        sprintf(label, "FrameRate (%d -> %d)",
                cameras_params[selected_camera].frame_rate_min,
                cameras_params[selected_camera].frame_rate_max);
        if (ImGui::SliderInt(label, &slider_frame_rate,
                             cameras_params[selected_camera].frame_rate_min,
                             cameras_params[selected_camera].frame_rate_max,
                             "%d")) {
            update_frame_rate_value(&ecams[selected_camera].camera,
                                    slider_frame_rate,
                                    &cameras_params[selected_camera]);
        }

        ImGui::TreePop();
    }
}

// utility structure for realtime plot
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

#endif
