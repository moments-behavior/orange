#include "gui.h"
#include "global.h"
#include "video_capture.h"

void setup_texture(GL_Texture &tex, int width, int height) {
    create_pbo(&tex.pbo, width, height);
    register_pbo_to_cuda(&tex.pbo, &tex.cuda_resource);
    map_cuda_resource(&tex.cuda_resource);
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

void start_camera_streaming(
    std::vector<std::thread> &camera_threads, CameraControl *camera_control,
    CameraEmergent *ecams, CameraParams *cameras_params,
    CameraEachSelect *cameras_select, GL_Texture *tex, int num_cameras,
    int evt_buffer_size, bool ptp_stream_sync, const std::string &encoder_setup,
    const std::string &folder_name, PTPParams *ptp_params) {
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
            ptp_params);
    }
}

void stop_camera_streaming(std::vector<std::thread> &camera_threads,
                           CameraControl *camera_control, CameraEmergent *ecams,
                           CameraParams *cameras_params,
                           CameraEachSelect *cameras_select,
                           const int num_cameras, const int evt_buffer_size,
                           PTPParams *ptp_params) {
    for (auto &t : camera_threads)
        t.join();

    camera_threads.clear();

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

    for (int i = 0; i < num_cameras; i++) {
        cameras_select[i].encoder_fps_estimator.reset();
        cameras_select[i].capture_fps_estimator.reset();
        cameras_select[i].dropped_frames = 0;
        cameras_select[i].pictures_counter = 0;
    }
}

bool input_text(const char *label, std::string &str,
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

void HelpMarker(const char *desc) {
    ImGui::TextDisabled(ICON_FK_INFO_CIRCLE);
    if (ImGui::IsItemHovered(ImGuiHoveredFlags_DelayShort) &&
        ImGui::BeginTooltip()) {
        ImGui::PushTextWrapPos(ImGui::GetFontSize() * 35.0f);
        ImGui::TextUnformatted(desc);
        ImGui::PopTextWrapPos();
        ImGui::EndTooltip();
    }
}

void set_camera_properties(CameraEmergent *ecams, CameraParams *cameras_params,
                           CameraEachSelect *cameras_select,
                           const int num_cameras,
                           std::vector<std::string> &color_temps) {

    if (ImGui::TreeNode("Camera Property")) {
        static int selected_camera = 0;
        static int slider_gain, slider_exposure, slider_frame_rate,
            slider_width, slider_height, OffsetX, OffsetY, slider_focus,
            slider_iris, slider_gop;

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

        if (ImGui::SliderInt("GOP", &slider_gop, 1, 10, "%d second")) {
            cameras_params[selected_camera].gop = slider_gop;
        }
        ImGui::SameLine();
        HelpMarker("Set keyframe interval as a multiple of the framerate. "
                   "Default is set to 1 second.");

        ImGui::Checkbox("GPU Direct",
                        &cameras_params[selected_camera].gpu_direct);
        if (cameras_params[selected_camera].gpu_direct) {
            ImGui::InputInt("GPU ID", &cameras_params[selected_camera].gpu_id);
        }

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
