#ifndef ORANGE_GUI
#define ORANGE_GUI
#include "IconsForkAwesome.h"
#include "camera.h"
#include "detect3d.h"
#include "global.h"
#include "gx_helper.h"
#include "imgui.h"
#include "realtime_tool.h"
#include "video_capture.h"
#include <math.h>
#include <thread>

struct EncoderConfig {
    std::string encoder_codec;
    int gop;
    std::string encoder_preset;
    std::string folder_name;
};

struct GL_Texture {
    GLuint texture;
    GLuint pbo;
    cudaGraphicsResource_t cuda_resource;
    unsigned char *cuda_buffer;
    size_t cuda_pbo_storage_buffer_size;
    int num_channels;
};

inline void setup_texture(GL_Texture &tex, int width, int height) {
    create_pbo(&tex.pbo, width, height);
    register_pbo_to_cuda(&tex.pbo, &tex.cuda_resource);
    map_cuda_resource(&tex.cuda_resource);
    cuda_pointer_from_resource(&tex.cuda_buffer,
                               &tex.cuda_pbo_storage_buffer_size,
                               &tex.cuda_resource);
    create_texture(&tex.texture, width, height);
}

inline void upload_texture_from_pbo(GL_Texture &tex, int width, int height) {
    bind_pbo(&tex.pbo);
    bind_texture(&tex.texture);
    upload_image_pbo_to_texture(width,
                                height); // Uses currently bound PBO and texture
    unbind_pbo();
    unbind_texture();
}

inline void clear_upload_and_cleanup(GL_Texture &tex, int width, int height) {
    // 1. Clear the buffer (only if valid)
    if (tex.cuda_buffer) {
        int size_pic =
            width * height * 4 * sizeof(unsigned char); // assuming uchar4
        cudaMemset(tex.cuda_buffer, 0, size_pic);
    }

    // 2. Upload from PBO to texture
    upload_texture_from_pbo(tex, width, height);

    // 3. Unmap if mapped
    if (tex.cuda_resource) {
        cudaError_t err = cudaGraphicsUnmapResources(1, &tex.cuda_resource, 0);
    }

    // 4. Now it's safe to unregister
    if (tex.cuda_resource) {
        cudaGraphicsUnregisterResource(tex.cuda_resource);
        tex.cuda_resource = nullptr;
    }

    // 5. Delete OpenGL PBO
    if (tex.pbo) {
        glDeleteBuffers(1, &tex.pbo);
        tex.pbo = 0;
    }

    // 6. Delete OpenGL texture
    if (tex.texture) {
        glDeleteTextures(1, &tex.texture);
        tex.texture = 0;
    }

    // 7. Null everything else
    tex.cuda_buffer = nullptr;
    tex.cuda_pbo_storage_buffer_size = 0;
}

inline void start_camera_streaming(
    std::vector<std::thread> &camera_threads, CameraControl *camera_control,
    CameraEmergent *ecams, CameraParams *cameras_params,
    CameraEachSelect *cameras_select, GL_Texture *tex, int num_cameras,
    int evt_buffer_size, bool ptp_stream_sync, const std::string &encoder_setup,
    const std::string &folder_name, PTPParams *ptp_params,
    INDIGOSignalBuilder *indigo_signal_builder, std::string calib_yaml_folder,
    std::thread &detection3d_thread) {

    detection2d = new DetectionDataPerCam[num_cameras];
    int idx3d = 0;
    int total_standoff_detector = 0;
    for (int i = 0; i < num_cameras; i++) {
        detection2d[i].calibration_file = calib_yaml_folder + "/Cam" +
                                          cameras_params[i].camera_serial +
                                          ".yaml";
        detection2d[i].has_calibration_results =
            load_camera_calibration_results(detection2d[i].calibration_file,
                                            &detection2d[i].camera_calib);
        if (detection2d[i].has_calibration_results) {
            std::cout << detection2d[i].calibration_file << std::endl;
        }
        cameras_select[i].idx2d = i;
        if (cameras_select[i].detect_mode == Detect3D_Standoff) {
            cameras_select[i].idx3d = idx3d;
            idx3d++;
        }
        if (cameras_select[i].detect_mode == Detect3D_Standoff ||
            cameras_select[i].detect_mode == Detect2D_Standoff) {
            total_standoff_detector++;
        }
    }

    for (int i = 0; i < num_cameras; i++) {
        cameras_select[i].total_standoff_detector = total_standoff_detector;
    }

    if (idx3d >= 2) {
        detection3d_thread = std::thread(&detection3d_proc, camera_control,
                                         cameras_select, num_cameras);
    }

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
                      const int evt_buffer_size, PTPParams *ptp_params,
                      std::thread &detection3d_thread) {
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

    cv3d.notify_all();

    if (detection3d_thread.joinable()) {
        detection3d_thread.join();
    }
    delete[] detection2d;
    detection2d = nullptr;

    for (int i = 0; i < num_cameras; i++) {
        cameras_select[i].frame_detect_state.store(State_Frame_Idle);
        cameras_select[i].total_standoff_detector = 0;
        cameras_select[i].idx2d = 0;
        cameras_select[i].idx3d = 0;
    }
    detector_counter.store(0);
}

inline bool input_text(const char *label, std::string &str,
                       ImGuiInputTextFlags flags = 0) {
    // Create a buffer big enough for current string + margin
    static const size_t buf_size = 256;
    char buf[buf_size];
    std::snprintf(buf, buf_size, "%s", str.c_str());

    if (ImGui::InputText(label, buf, buf_size, flags)) {
        str = std::string(buf);
        return true; // value changed
    }
    return false;
}

inline void HelpMarker(const char *desc) {
    ImGui::TextDisabled(ICON_FK_INFO_CIRCLE);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) &&
        ImGui::BeginTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

inline void set_camera_properties(CameraEmergent *ecams,
                                  CameraParams *cameras_params,
                                  CameraEachSelect *cameras_select,
                                  const int num_cameras,
                                  std::vector<std::string> &color_temps,
                                  EncoderConfig *encoder_config) {

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

        ImGui::SliderInt("GOP", &encoder_config->gop, 1, 10, "%d second");
        ImGui::SameLine();
        HelpMarker("Set keyframe interval as a multiple of the framerate. "
                   "Default is set to 1 second.");

        input_text("YOLO", cameras_select->yolo_model);
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

inline void gui_plot_world_coordinates(CameraCalibResults *cvp,
                                       CameraParams *camera_params) {
    double axis_x_values[4];
    double axis_y_values[4];
    world_coordinates_projection_points(cvp, axis_x_values, axis_y_values, 50,
                                        camera_params);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0,
                               ImVec4(1.0, 1.0, 1.0, 1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0, 1.0), 3.0);
    std::string name = "World Origin";

    std::vector<triple_f> node_colors = {{1.0f, 1.0f, 1.0f},
                                         {1.0f, 0.0f, 0.0f},
                                         {0.0f, 1.0f, 0.0f},
                                         {0.0f, 0.0f, 1.0f}};

    for (u32 edge = 0; edge < 3; edge++) {
        double xs[2]{axis_x_values[0], axis_x_values[edge + 1]};
        double ys[2]{axis_y_values[0], axis_y_values[edge + 1]};

        ImVec4 my_color;
        my_color.w = 1.0f;
        my_color.x = node_colors[edge + 1].x;
        my_color.y = node_colors[edge + 1].y;
        my_color.z = node_colors[edge + 1].z;

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, my_color);
        ImPlot::SetNextLineStyle(my_color, 3.0);
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);
    }
}

inline void draw_aruco_markers(Aruco2d *aruco_marker, int frame_height) {
    double x[5] = {(double)aruco_marker->proj_corners[0].x,
                   (double)aruco_marker->proj_corners[1].x,
                   (double)aruco_marker->proj_corners[2].x,
                   (double)aruco_marker->proj_corners[3].x,
                   (double)aruco_marker->proj_corners[0].x};

    double y[5] = {
        (double)frame_height - (double)aruco_marker->proj_corners[0].y,
        (double)frame_height - (double)aruco_marker->proj_corners[1].y,
        (double)frame_height - (double)aruco_marker->proj_corners[2].y,
        (double)frame_height - (double)aruco_marker->proj_corners[3].y,
        (double)frame_height - (double)aruco_marker->proj_corners[0].y};

    ImPlot::SetNextLineStyle(ImVec4(1.0, 0.0, 1.0, 1.0), 3.0);
    ImPlot::PlotLine("Aruco", x, y, 5);
}

inline void draw_keypoints(const std::vector<float> &keypoints,
                           int frame_height, ImVec4 color,
                           const std::string &name,
                           ImPlotMarker marker = ImPlotMarker_Circle,
                           float pt_size = 4.0f) {
    int num_points = keypoints.size() / 2;
    if (num_points == 0)
        return;

    std::vector<double> xs(num_points);
    std::vector<double> ys(num_points);

    for (int i = 0; i < num_points; ++i) {
        xs[i] = static_cast<double>(keypoints[2 * i]);
        ys[i] = static_cast<double>(frame_height - keypoints[2 * i + 1]);
    }

    ImPlot::PushStyleColor(ImPlotCol_Line, color);
    ImPlot::SetNextMarkerStyle(marker, pt_size, color, 2.5f);
    ImPlot::PlotScatter(name.c_str(), xs.data(), ys.data(), num_points);
    ImPlot::PopStyleColor();
}

inline void draw_box(cv::Rect_<float> bbox, int frame_height, ImVec4 color,
                     std::string name, ImPlotMarker marker, float pt_size) {
    double x[5] = {(double)bbox.x, (double)bbox.x + bbox.width,
                   (double)bbox.x + bbox.width, (double)bbox.x, (double)bbox.x};

    double y[5] = {(double)frame_height - ((double)bbox.y),
                   (double)frame_height - ((double)bbox.y),
                   (double)frame_height - ((double)bbox.y + bbox.height),
                   (double)frame_height - ((double)bbox.y + bbox.height),
                   (double)frame_height - ((double)bbox.y)};

    ImPlot::SetNextLineStyle(color, 2.0);
    ImPlot::PlotLine(name.c_str(), x, y, 5);
}

inline void draw_detection(std::vector<Object> objs, int num_classes,
                           int frame_height, ImPlotMarker marker,
                           float pt_size) {

    std::vector<ImVec4> colors;
    for (int i = 0; i < num_classes; i++) {
        ImVec4 color = (ImVec4)ImColor::HSV(i / (float)num_classes, 0.8f, 0.8f);
        colors.push_back(color);
    }

    for (size_t i = 0; i < objs.size(); i++) {
        std::string bbox_name = "##bbox" + std::to_string(i);
        draw_box(objs[i].rect, frame_height, colors[objs[i].label], bbox_name,
                 marker, pt_size);
        if (!objs[i].kps.empty()) {
            std::vector<float> filtered_xy;
            for (size_t j = 0; j + 2 < objs[i].kps.size(); j += 3) {
                float x = objs[i].kps[j];
                float y = objs[i].kps[j + 1];
                float s = objs[i].kps[j + 2];

                if (s > 0.5f) {
                    filtered_xy.push_back(x);
                    filtered_xy.push_back(y);
                }
            }

            std::string kp_name = "##kps" + std::to_string(i);
            draw_keypoints(filtered_xy, frame_height, colors[objs[i].label],
                           kp_name);
        }
    }
}

#endif
