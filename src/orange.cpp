#include "camera.h"
#include "enet_fb_helpers.h"
#include "enet_utils.h"
#include "global.h"
#include "gui.h"
#include "host_client_imgui_procedural.h"
#include "host_ctx.h"
#include "imgui.h"
#include "implot.h"
#include "plot_buffers.h"
#include "realtime_tool.h"
#include "utils.h"
#include "video_capture.h"
#include <ImGuiFileDialog.h>
#include <iostream>
#include <sys/stat.h>

int main(int argc, char **args) {
    int display_gpu_id = 0;
    CHECK(cudaSetDevice(display_gpu_id));

    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    *window =
        (gx_context){.swap_interval = 1, // use vsync
                     .width = 1920,
                     .height = 1080,
                     .render_target_title = (char *)malloc(100), // window title
                     .glsl_version = (char *)malloc(100)};

    render_initialize_target(window);

    const int max_cameras = 20;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    int cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);

    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split(cwd, delimiter);
    std::string orange_root_dir_str =
        "/home/" + tokenized_path[2] + "/orange_data";
    prepare_application_folders(orange_root_dir_str);
    std::string recording_root_dir_str = "/data0";
    std::string input_folder = recording_root_dir_str + "/exp/unsorted";
    std::string calib_yaml_folder = orange_root_dir_str + "/calib_yaml";

    std::vector<bool> check;
    for (int i = 0; i < cam_count; i++) {
        check.push_back(false);
    }
    CameraParams *cameras_params;
    CameraEachSelect *cameras_select;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    GL_Texture *tex_gl;
    int num_cameras = 0;
    CameraControl *camera_control =
        new CameraControl{false, false, false, false, false};

    int evt_buffer_size{100};
    PTPParams *ptp_params =
        new PTPParams{0, 0, 0, 0, false, false, false, false};

    EncoderConfig *encoder_config = new EncoderConfig{"h264", 1, "p1"};
    std::vector<std::string> camera_config_files;

    ScrollingBuffer *realtime_plot_data;
    bool show_realtime_plot = false;
    bool ptp_stream_sync = false;

    AppContext ctx; // ENetGuard constructed here (enet_initialize)
    std::vector<std::pair<std::string, int>> cams = {{"127.0.0.1", 34001},
                                                     {"127.0.0.1", 34002}};
    HostClient_StartNetThread(ctx); // start dispatcher thread
    HostClient_Init(ctx, cams);

    std::vector<std::string> network_config_folders;
    int network_config_select = -1;
    std::string network_start_folder_name =
        orange_root_dir_str + "/config/network";
    for (const auto &entry :
         std::filesystem::directory_iterator(network_start_folder_name)) {
        network_config_folders.push_back(entry.path().string());
    }

    std::vector<std::string> local_config_folders;
    std::string local_start_folder_name = orange_root_dir_str + "/config/local";
    for (const auto &entry :
         std::filesystem::directory_iterator(local_start_folder_name)) {
        local_config_folders.push_back(entry.path().string());
    }
    std::string picture_save_folder =
        orange_root_dir_str + "/pictures/" + get_current_date();
    std::string calib_save_folder;

    int local_config_select = 0;
    bool select_all_cameras = false;
    char *temp_string = (char *)malloc(64);
    *temp_string = '\0';
    bool save_image_all_ready{false};

    std::vector<std::string> color_temps = {"CT_Off",   "CT_2800K", "CT_3000K",
                                            "CT_4000K", "CT_5000K", "CT_6500K",
                                            "CT_Custom"};
    std::vector<std::string> server_names = {"waffle-0", "waffle-1"};
    std::thread detection3d_thread;
    bool show_error = false;
    std::string error_message;

    HostOpenCtx open_ctx{&camera_config_files,
                         &network_config_folders,
                         &network_config_select,
                         device_info,
                         &cam_count,
                         &check,
                         &num_cameras,
                         &cameras_params,
                         &cameras_select,
                         &ecams,
                         &realtime_plot_data,
                         camera_control};
    HostClient_SetOpenCtx(&open_ctx);

    while (!glfwWindowShouldClose(window->render_target)) {
        HostClient_Tick();
        create_new_frame();

        if (ImGui::Begin("Network")) {
            if (network_config_select < 0 ||
                network_config_select >= (int)network_config_folders.size()) {
                int idx = find_cfg_index(network_config_folders, "rig_new");
                network_config_select =
                    (idx >= 0 ? idx
                              : (network_config_folders.empty() ? -1 : 0));
            }

            ImGuiStyle &style = ImGui::GetStyle();
            const int n = (int)network_config_folders.size();
            for (int i = 0; i < n; ++i) {
                if (i > 0)
                    ImGui::SameLine(0.0f, style.ItemInnerSpacing.x);

                std::string label =
                    std::filesystem::path(network_config_folders[i])
                        .filename()
                        .string();

                const bool is_rig_new = (label == "rig_new");
                if (is_rig_new)
                    ImGui::PushStyleColor(ImGuiCol_Text,
                                          ImVec4(1.0f, 0.55f, 0.0f, 1.0f));

                ImGui::RadioButton(
                    (label + "##cfg" + std::to_string(i)).c_str(),
                    &network_config_select, i);

                if (is_rig_new)
                    ImGui::PopStyleColor();
            }
        }
        ImGui::End();

        HostClient_DrawImGui(); // shows the “Advance Phase” button & logs

        if (ImGui::Begin("Orange", nullptr)) {
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)",
                        1000.0f / ImGui::GetIO().Framerate,
                        ImGui::GetIO().Framerate);

            if (camera_control->open) {
                ImGui::BeginDisabled();
            }

            if (ImGui::BeginTable("Cameras", 3,
                                  ImGuiTableFlags_Resizable |
                                      ImGuiTableFlags_NoSavedSettings |
                                      ImGuiTableFlags_Borders)) {
                for (int i = 0; i < cam_count; i++) {
                    sprintf(temp_string, "%d", i);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Selectable(temp_string, check[i],
                                      ImGuiSelectableFlags_SpanAllColumns);
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", device_info[i].serialNumber);
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", device_info[i].currentIp);
                }
                ImGui::EndTable();
            }

            if (ImGui::Button(select_all_cameras ? "Clear all"
                                                 : "Select all")) {
                select_all_cameras = !select_all_cameras;
                if (select_all_cameras) {
                    for (int i = 0; i < cam_count; i++) {
                        check[i] = true;
                    }
                } else {
                    for (int i = 0; i < cam_count; i++) {
                        check[i] = false;
                    }
                }
            }

            if (camera_control->open) {
                ImGui::EndDisabled();
            }

            if (camera_control->subscribe) {
                ImGui::BeginDisabled();
            }

            ImGui::Separator();
            ImGui::Spacing();

            if (camera_control->subscribe) {
                ImGui::EndDisabled();
            }

            if (camera_control->record_video) {
                ImGui::BeginDisabled();
            }

            ImGui::PushStyleColor(ImGuiCol_Button,
                                  ImVec4(0.5f, 0.0f, 0.7f, 1.0f)); // normal
            ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                  ImVec4(0.7f, 0.2f, 0.9f, 1.0f)); // hover
            ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                  ImVec4(0.4f, 0.0f, 0.6f, 1.0f)); // active
            if (ImGui::Button("Save to")) {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = input_folder;
                config.flags = ImGuiFileDialogFlags_Modal;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseRecordingDir",
                                                        "Choose a Directory",
                                                        nullptr, config);
            }
            ImGui::PopStyleColor(3);
            ImGui::SameLine();
            ImGui::Text("%s", input_folder.c_str());

            {
                const char *codecs[] = {"h264", "hevc"};
                static int codec_current = -1;

                if (codec_current == -1) {
                    for (int i = 0; i < IM_ARRAYSIZE(codecs); ++i) {
                        if (encoder_config->encoder_codec == codecs[i]) {
                            codec_current = i;
                            break;
                        }
                    }
                }

                if (ImGui::Combo("Codec", &codec_current, codecs,
                                 IM_ARRAYSIZE(codecs))) {
                    encoder_config->encoder_codec = codecs[codec_current];
                }
            }

            {
                const char *presets[] = {"p1", "p3", "p5", "p7"};
                static int preset_current = -1;

                if (preset_current == -1) {
                    for (int i = 0; i < IM_ARRAYSIZE(presets); ++i) {
                        if (encoder_config->encoder_preset == presets[i]) {
                            preset_current = i;
                            break;
                        }
                    }
                }

                if (ImGui::Combo("Preset", &preset_current, presets,
                                 IM_ARRAYSIZE(presets))) {
                    encoder_config->encoder_preset = presets[preset_current];
                }
            }

            int fps_temp =
                streaming_target_fps.load(); // get the current atomic value

            if (ImGui::InputInt("streaming fps", &fps_temp)) {
                // Clamp if necessary
                if (fps_temp < 1)
                    fps_temp = 1;
                if (fps_temp > 240)
                    fps_temp = 240;
                streaming_target_fps.store(fps_temp);
            }

            if (camera_control->record_video) {
                ImGui::EndDisabled();
            }

            if (camera_control->open) {
                if (camera_control->record_video) {
                    ImGui::BeginDisabled();
                }

                ImGui::Checkbox("Show camera temperature", &show_realtime_plot);
                set_camera_properties(ecams, cameras_params, cameras_select,
                                      num_cameras, color_temps, encoder_config);

                if (camera_control->record_video) {
                    ImGui::EndDisabled();
                }

                if (camera_control->subscribe) {
                    ImGui::BeginDisabled();
                }

                bool stream_all_cameras = true;
                for (int i = 0; i < num_cameras; i++) {
                    if (!cameras_select[i].stream_on) {
                        stream_all_cameras = false;
                        break;
                    }
                }

                bool record_all_cameras = true;
                for (int i = 0; i < num_cameras; i++) {
                    if (!cameras_select[i].record) {
                        record_all_cameras = false;
                        break;
                    }
                }

                if (ImGui::BeginTable("Camera Control Setting", 5,
                                      ImGuiTableFlags_Resizable |
                                          ImGuiTableFlags_NoSavedSettings |
                                          ImGuiTableFlags_Borders)) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("name");
                    ImGui::TableNextColumn();
                    ImGui::Text("serial");
                    ImGui::TableNextColumn();
                    ImGui::Text("stream ");
                    ImGui::SameLine();
                    if (ImGui::Checkbox("all##stream", &stream_all_cameras)) {
                        if (stream_all_cameras) {
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].stream_on = true;
                            }
                        } else {
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].stream_on = false;
                            }
                        }
                    }

                    ImGui::TableNextColumn();
                    ImGui::Text("record ");
                    ImGui::SameLine();
                    if (ImGui::Checkbox("all##record", &record_all_cameras)) {
                        if (record_all_cameras) {
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].record = true;
                            }
                        } else {
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].record = false;
                            }
                        }
                    }
                    ImGui::TableNextColumn();
                    ImGui::Text("yolo");

                    for (int i = 0; i < num_cameras; i++) {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text("%s",
                                    cameras_params[i].camera_name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%s",
                                    cameras_params[i].camera_serial.c_str());
                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##checkbox_stream%d", i);
                        ImGui::Checkbox(temp_string,
                                        &cameras_select[i].stream_on);
                        ImGui::SameLine();
                        if (cameras_select[i].stream_on) {
                            {
                                static const char *downsample_labels[] = {
                                    "1", "2", "4", "8", "16", "32"};
                                static const int downsample_options[] = {
                                    1, 2, 4, 8, 16, 32};
                                const int num_options =
                                    IM_ARRAYSIZE(downsample_options);

                                // Find index from current value
                                int current_index = 0;
                                for (int j = 0; j < num_options; ++j) {
                                    if (cameras_select[i].downsample ==
                                        downsample_options[j]) {
                                        current_index = j;
                                        break;
                                    }
                                }

                                // Show Combo by index
                                ImGui::PushItemWidth(50);
                                std::string ds_label =
                                    "downsample##" + std::to_string(i);
                                if (ImGui::Combo(
                                        ds_label.c_str(), &current_index,
                                        downsample_labels, num_options)) {
                                    // Update value from selected index
                                    cameras_select[i].downsample =
                                        downsample_options[current_index];
                                }
                                ImGui::PopItemWidth();
                            }
                        }

                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##checkbox_record%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].record);
                        ImGui::TableNextColumn();

                        int current_index =
                            static_cast<int>(cameras_select[i].detect_mode);
                        sprintf(temp_string, "##detection_mode%d", i);
                        if (ImGui::Combo(temp_string, &current_index,
                                         DetectModeNames,
                                         IM_ARRAYSIZE(DetectModeNames))) {
                            if (current_index != 0 &&
                                cameras_select[i].yolo_model.empty()) {
                                current_index = 0;
                                error_message = "Specify YOLO model first in "
                                                "Camera Property.";
                                show_error = true;
                            }
                            cameras_select[i].detect_mode =
                                static_cast<DetectMode>(current_index);
                        }
                    }
                    ImGui::EndTable();
                }

                if (camera_control->subscribe) {
                    ImGui::EndDisabled();
                }

                if (camera_control->subscribe == true) {
                    ImGui::Separator();
                    ImGui::Spacing();
                    if (ImGui::Button("Picture save to")) {
                        make_folder(picture_save_folder);
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].pictures_counter = 0;
                        }
                        IGFD::FileDialogConfig config;
                        config.countSelectionMax = 1;
                        config.path = picture_save_folder;
                        config.flags = ImGuiFileDialogFlags_Modal;
                        ImGuiFileDialog::Instance()->OpenDialog(
                            "ChoosePictureDir", "Choose a Directory", nullptr,
                            config);
                    }
                    ImGui::SameLine();
                    ImGui::Text("%s", picture_save_folder.c_str());
                    static int current_picture_format = 0;
                    const char *picture_format_items[] = {"jpg", "tiff", "png"};
                    ImGui::Combo("Picture format", &current_picture_format,
                                 picture_format_items,
                                 IM_ARRAYSIZE(picture_format_items));
                    for (int i = 0; i < num_cameras; i++) {
                        cameras_select[i].frame_save_format = std::string(
                            picture_format_items[current_picture_format]);
                    }

                    if (save_pics_counter == num_cameras) {
                        save_image_all_ready = true;
                    }

                    ImGui::Text("Reset counter: ");
                    const int cols = 5;
                    for (int i = 0; i < num_cameras; ++i) {
                        std::string label =
                            cameras_params[i].camera_name + ": " +
                            std::to_string(cameras_select[i].pictures_counter) +
                            "##calibration_save";
                        if (ImGui::Selectable(label.c_str(), false,
                                              ImGuiSelectableFlags_None,
                                              ImVec2(150, 50))) {
                            cameras_select[i].pictures_counter = 0;
                        }

                        // Keep items on the same line until end of row
                        if ((i + 1) % cols != 0)
                            ImGui::SameLine();
                    }

                    if (!save_image_all_ready) {
                        ImGui::BeginDisabled();
                    }

                    ImGui::NewLine();
                    if (ImGui::Button("Reset counters all")) {
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].pictures_counter = 0;
                        }
                    }

                    // // order important
                    // if (save_image_all_ready &&
                    //     calib_state == CalibSavePictures) {
                    //     save_pics_counter = 0;
                    //     send_message_to_indigo(
                    //         ctx.sender, ctx.peers, "indigo",
                    //         FetchGame::SignalType_CalibrationNextPose);
                    //     calib_state = CalibNextPose;
                    // }

                    if (calib_state == CalibPoseReached) {
                        make_folder(calib_save_folder);
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].frame_save_name = std::to_string(
                                cameras_select[i].pictures_counter);
                            cameras_select[i].picture_save_folder =
                                calib_save_folder;
                            cameras_select[i].frame_save_state.store(
                                State_Copy_New_Frame);
                        }
                        calib_state = CalibSavePictures;
                    }

                    if (ImGui::Button("Calib save images with counter")) {
                        make_folder(calib_save_folder);
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].frame_save_name = std::to_string(
                                cameras_select[i].pictures_counter);
                            cameras_select[i].picture_save_folder =
                                calib_save_folder;
                            cameras_select[i].frame_save_state.store(
                                State_Copy_New_Frame);
                        }
                    }

                    if (!save_image_all_ready) {
                        ImGui::EndDisabled();
                    }
                }
            }
        }
        ImGui::End();

        // file explorer display
        if (ImGuiFileDialog::Instance()->Display("ChooseRecordingDir")) {
            // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {
                // action if OK
                auto selected_folder =
                    ImGuiFileDialog::Instance()->GetSelection();
                input_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChoosePictureDir")) {
            // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {
                // action if OK
                auto selected_folder =
                    ImGuiFileDialog::Instance()->GetSelection();
                picture_save_folder =
                    ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGui::Begin("Local")) {
            if (camera_control->open) {
                ImGui::BeginDisabled();
            }

            for (int i = 0; i < local_config_folders.size(); i++) {
                std::vector<std::string> folder_token =
                    string_split(local_config_folders[i], "/");
                sprintf(temp_string, "%s", folder_token.back().c_str());
                ImGui::RadioButton(temp_string, &local_config_select, i);
                ImGui::SameLine();
            }
            ImGui::RadioButton("Null", &local_config_select,
                               local_config_folders.size());

            if (camera_control->open) {
                ImGui::EndDisabled();
            }

            if (camera_control->subscribe) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button(camera_control->open ? "Close Camera"
                                                   : "Open camera")) {
                if (!camera_control->open) {
                    if (local_config_select < local_config_folders.size()) {
                        update_camera_configs(
                            camera_config_files,
                            local_config_folders[local_config_select]);
                        select_cameras_have_configs(
                            camera_config_files, device_info, check, cam_count);
                    }

                    num_cameras = 0;
                    for (int i = 0; i < cam_count; i++) {
                        if (check[i]) {
                            num_cameras++;
                        }
                    }
                    if (num_cameras > 0) {
                        camera_control->open = true;
                        cameras_params = new CameraParams[num_cameras];
                        cameras_select = new CameraEachSelect[num_cameras];

                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++) {
                            if (check[i]) {
                                selected_cameras.push_back(i);
                            }
                        }

                        std::vector<bool> skip_setting_params;
                        skip_setting_params.resize(num_cameras);
                        for (int i = 0; i < num_cameras; i++) {
                            if (!set_camera_params(
                                    &cameras_params[i], &cameras_select[i],
                                    &device_info[selected_cameras[i]],
                                    camera_config_files, selected_cameras[i],
                                    num_cameras)) {
                                skip_setting_params[i] = true;
                                cameras_params[i].camera_id =
                                    selected_cameras[i];
                                cameras_params[i].num_cameras = num_cameras;
                            } else {
                                skip_setting_params[i] = false;
                            }
                        }

                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].stream_on = false;
                            if (cameras_params[i].camera_name ==
                                "ceiling_center") {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].detect_mode =
                                    Detect2D_GLThread;
                            }

                            if (cameras_params[i].camera_name == "shelter") {
                                cameras_select[i].stream_on = true;
                            }
                        }

                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            if (!skip_setting_params[i]) {
                                open_camera_with_params(
                                    &ecams[i].camera,
                                    &device_info[cameras_params[i].camera_id],
                                    &cameras_params[i]);
                            } else {
                                update_camera_params(
                                    &ecams[i].camera,
                                    &device_info[cameras_params[i].camera_id],
                                    &cameras_params[i]);
                            }
                        }
                        realtime_plot_data = new ScrollingBuffer[num_cameras];
                    }
                } else {
                    camera_control->open = false;
                    for (int i = 0; i < num_cameras; i++) {
                        close_camera(&ecams[i].camera, &cameras_params[i]);
                    }
                    delete[] cameras_params;
                    delete[] cameras_select;
                    delete[] ecams;
                }
            }
            if (camera_control->subscribe) {
                ImGui::EndDisabled();
            }

            if (!camera_control->record_video && camera_control->open) {
                if (camera_control->subscribe) {
                    ImGui::BeginDisabled();
                }
                ImGui::Checkbox("PTP Stream Sync", &ptp_stream_sync);
                ImGui::SameLine();
                // ImGui::Checkbox("Trigger Mode",
                // &camera_control->trigger_mode);
                if (camera_control->subscribe) {
                    ImGui::EndDisabled();
                }
                if (ImGui::Button(camera_control->subscribe
                                      ? "Stop streaming"
                                      : "Start streaming")) {
                    (camera_control->subscribe) = !(camera_control->subscribe);
                    if (camera_control->subscribe) {
                        cudaSetDevice(display_gpu_id);
                        tex_gl = new GL_Texture[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                int camera_width =
                                    int(cameras_params[i].width /
                                        cameras_select[i].downsample);
                                int camera_height =
                                    int(cameras_params[i].height /
                                        cameras_select[i].downsample);
                                setup_texture(tex_gl[i], camera_width,
                                              camera_height);
                            }
                        }
                        start_camera_streaming(
                            camera_threads, camera_control, ecams,
                            cameras_params, cameras_select, tex_gl, num_cameras,
                            evt_buffer_size, ptp_stream_sync, "",
                            encoder_config->folder_name, ptp_params,
                            calib_yaml_folder, detection3d_thread, ctx);
                    } else {
                        stop_camera_streaming(
                            camera_threads, camera_control, ecams,
                            cameras_params, cameras_select, num_cameras,
                            evt_buffer_size, ptp_params, detection3d_thread);
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                int camera_width =
                                    int(cameras_params[i].width /
                                        cameras_select[i].downsample);
                                int camera_height =
                                    int(cameras_params[i].height /
                                        cameras_select[i].downsample);
                                clear_upload_and_cleanup(
                                    tex_gl[i], camera_width, camera_height);
                            }
                        }
                        delete[] tex_gl;
                        tex_gl = nullptr;
                    }
                }
            }

            if (camera_control->stop_record) {
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImVec4{0.4f, 0.0f, 0.0f, 1.0f});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                      ImVec4{0.7f, 0.1f, 0.1f, 1.0f});
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                      ImVec4{0.5f, 0.0f, 0.0f, 1.0f});
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button,
                                      ImVec4{0.0f, 0.5f, 0.0f, 1.0f});
                ImGui::PushStyleColor(ImGuiCol_ButtonHovered,
                                      ImVec4{0.2f, 0.8f, 0.2f, 1.0f});
                ImGui::PushStyleColor(ImGuiCol_ButtonActive,
                                      ImVec4{0.1f, 0.6f, 0.1f, 1.0f});
            }

            if (camera_control->open) {
                if (ImGui::Button(camera_control->stop_record ? ICON_FK_PAUSE
                                                              : ICON_FK_PLAY)) {
                    (camera_control->stop_record) =
                        !(camera_control->stop_record);
                    if (camera_control->stop_record) {
                        if (camera_control->subscribe) {
                            camera_control->subscribe = false;
                            stop_camera_streaming(
                                camera_threads, camera_control, ecams,
                                cameras_params, cameras_select, num_cameras,
                                evt_buffer_size, ptp_params,
                                detection3d_thread);
                            for (int i = 0; i < num_cameras; i++) {
                                if (cameras_select[i].stream_on) {
                                    int camera_width =
                                        int(cameras_params[i].width /
                                            cameras_select[i].downsample);
                                    int camera_height =
                                        int(cameras_params[i].height /
                                            cameras_select[i].downsample);
                                    clear_upload_and_cleanup(
                                        tex_gl[i], camera_width, camera_height);
                                }
                            }
                            delete[] tex_gl;
                            tex_gl = nullptr;
                        }

                        camera_control->subscribe = true;
                        std::string encoder_setup =
                            "-codec " + encoder_config->encoder_codec +
                            " -preset " + encoder_config->encoder_preset;
                        camera_control->record_video = true;
                        encoder_config->folder_name =
                            input_folder + "/" + get_current_date_time();
                        make_folder(encoder_config->folder_name);
                        if (num_cameras > 1) {
                            ptp_stream_sync = true;
                        } else {
                            ptp_stream_sync = false;
                        }

                        cudaSetDevice(display_gpu_id);
                        tex_gl = new GL_Texture[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                int camera_width =
                                    int(cameras_params[i].width /
                                        cameras_select[i].downsample);
                                int camera_height =
                                    int(cameras_params[i].height /
                                        cameras_select[i].downsample);
                                setup_texture(tex_gl[i], camera_width,
                                              camera_height);
                            }
                        }

                        start_camera_streaming(
                            camera_threads, camera_control, ecams,
                            cameras_params, cameras_select, tex_gl, num_cameras,
                            evt_buffer_size, ptp_stream_sync, encoder_setup,
                            encoder_config->folder_name, ptp_params,
                            calib_yaml_folder, detection3d_thread, ctx);
                    } else {
                        camera_control->subscribe = false;
                        stop_camera_streaming(
                            camera_threads, camera_control, ecams,
                            cameras_params, cameras_select, num_cameras,
                            evt_buffer_size, ptp_params, detection3d_thread);
                        ptp_stream_sync = false;
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                int camera_width =
                                    int(cameras_params[i].width /
                                        cameras_select[i].downsample);
                                int camera_height =
                                    int(cameras_params[i].height /
                                        cameras_select[i].downsample);
                                clear_upload_and_cleanup(
                                    tex_gl[i], camera_width, camera_height);
                            }
                        }
                        delete[] tex_gl;
                        tex_gl = nullptr;
                        camera_control->record_video = false;
                    }
                }
            }

            ImGui::PopStyleColor(3);
        }
        ImGui::End();

        if (camera_control->subscribe) {
            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on) {
                    int camera_width = int(cameras_params[i].width /
                                           cameras_select[i].downsample);
                    int camera_height = int(cameras_params[i].height /
                                            cameras_select[i].downsample);
                    upload_texture_from_pbo(tex_gl[i], camera_width,
                                            camera_height);
                }
            }

            std::string g_formatted_elapsed_time;
            int64_t start_ns;

            if (camera_control->record_video) {
                start_ns = record_start_time_ns.load();
                if (start_ns > 0) {
                    int64_t now_ns =
                        std::chrono::duration_cast<std::chrono::nanoseconds>(
                            std::chrono::steady_clock::now().time_since_epoch())
                            .count();

                    auto elapsed_sec = std::chrono::seconds(
                        (now_ns - start_ns) / 1'000'000'000);
                    g_formatted_elapsed_time = format_elapsed_time(elapsed_sec);
                }
            }

            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on) {
                    std::string window_name = cameras_params[i].camera_name;
                    ImGui::Begin(window_name.c_str());

                    if (start_ns > 0) {
                        ImGui::TextColored(ImVec4{0.0, 1.0f, 0, 1.0f},
                                           "Elapsed Time: %s",
                                           g_formatted_elapsed_time.c_str());
                    } else {
                        if (camera_control->record_video) {
                            ImGui::TextColored(ImVec4{1.0, 1.0f, 0, 1.0f},
                                               "Recording starting...");

                        } else {
                            ImGui::TextColored(ImVec4{1.0, 0.0f, 0, 1.0f},
                                               "NOT RECORDING, ");
                        }
                    }

                    ImGui::SameLine();

                    std::ostringstream oss;
                    oss << std::fixed << std::setprecision(1);

                    oss << "Streaming FPS: " << streaming_fps.load();
                    oss << "  |  "
                        << "Capture FPS: "
                        << cameras_select[i].capture_fps_estimator.get_fps();
                    oss << "  |  "
                        << "Dropped Frames: "
                        << cameras_select[i].dropped_frames;

                    if (cameras_select[i].record &&
                        camera_control->record_video) {
                        oss << "  |  "
                            << "Encoding FPS: "
                            << cameras_select[i]
                                   .encoder_fps_estimator.get_fps();
                    }
                    if (cameras_select[i].detect_mode == Detect2D_Standoff) {
                        oss << "  |  "
                            << "Detection2D FPS: "
                            << detection2d[i].fps_estimator.get_fps();
                    } else if (cameras_select[i].detect_mode ==
                               Detect3D_Standoff) {
                        oss << "  |  "
                            << "Detection2D FPS: "
                            << detection2d[i].fps_estimator.get_fps();
                        oss << "  |  "
                            << "Detection3D FPS: "
                            << detection3d.fps_estimator.get_fps();
                    }
                    std::string text = oss.str();
                    ImGui::Text("%s", text.c_str());

                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    // ImGui::Image((void*)(intptr_t)texture[i],
                    // avail_size);
                    ImPlotAxisFlags axisFlags = ImPlotAxisFlags_NoTickLabels |
                                                ImPlotAxisFlags_NoTickMarks |
                                                ImPlotAxisFlags_NoGridLines;
                    if (ImPlot::BeginPlot("##no_plot_name", avail_size,
                                          ImPlotFlags_Equal |
                                              ImPlotAxisFlags_AutoFit)) {
                        ImPlot::SetupAxesLimits(0, cameras_params[i].width, 0,
                                                cameras_params[i].height);
                        ImPlot::SetupAxis(ImAxis_X1, nullptr,
                                          axisFlags); // X-axis
                        ImPlot::SetupAxis(ImAxis_Y1, nullptr,
                                          axisFlags); // Y-axis
                        ImPlot::PlotImage("##no_image_name",
                                          (void *)(intptr_t)tex_gl[i].texture,
                                          ImVec2(0, 0),
                                          ImVec2(cameras_params[i].width,
                                                 cameras_params[i].height));

                        if (cameras_select[i].detect_mode ==
                                Detect3D_Standoff ||
                            cameras_select[i].detect_mode ==
                                Detect2D_Standoff) {
                            if (detection2d[i].ball2d.find_ball.load()) {
                                std::string ball2d_name =
                                    "ball##" + std::to_string(i);
                                draw_boxes(
                                    detection2d[i].ball2d.rects,
                                    cameras_params[i].height,
                                    (ImVec4)ImColor::HSV(0.0, 1.0f, 1.0f),
                                    ball2d_name, ImPlotMarker_Circle, 6.0);
                            }
                        }

                        if (detection2d[i].has_calibration_results) {
                            gui_plot_world_coordinates(
                                &detection2d[i].camera_calib,
                                &cameras_params[i]);
                            if (detection3d.ball3d.new_detection.load()) {
                                std::string ball_proj_name =
                                    "ball_proj##" + std::to_string(i);
                                draw_ball_center(
                                    detection2d[i].ball2d.proj_center[0],
                                    cameras_params[i].height,
                                    (ImVec4)ImColor::HSV(0.5, 1.0f, 1.0f),
                                    ball_proj_name, ImPlotMarker_Cross, 8.0);
                            }
                        }

                        ImPlot::EndPlot();
                    }
                    ImGui::End();
                }
            }
        }

        if (camera_control->open && show_realtime_plot) {
            ImGui::Begin("Realtime Plots");
            {
                static float t = 0;
                t += ImGui::GetIO().DeltaTime;
                for (int i = 0; i < num_cameras; i++) {
                    get_senstemp_value(&ecams[i].camera, &cameras_params[i]);
                    realtime_plot_data[i].AddPoint(t,
                                                   cameras_params[i].sens_temp);
                }

                static float history = 10.0f;
                ImGui::SliderFloat("History", &history, 1, 30, "%.1f s");

                static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickMarks;
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                if (ImPlot::BeginPlot("Camera Sensor Temperature",
                                      avail_size)) {
                    ImPlot::SetupAxes(nullptr, nullptr, flags, flags);
                    ImPlot::SetupAxisLimits(ImAxis_X1, t - history, t,
                                            ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 30, 90);
                    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);

                    for (int i = 0; i < num_cameras; i++) {
                        std::string line_name =
                            std::string(cameras_params[i].camera_serial);
                        ImPlot::PlotLine(
                            line_name.c_str(), &realtime_plot_data[i].Data[0].x,
                            &realtime_plot_data[i].Data[0].y,
                            realtime_plot_data[i].Data.size(), 0,
                            realtime_plot_data[i].Offset, 2 * sizeof(float));
                    }
                    ImPlot::EndPlot();
                }
                ImGui::End();
            }
        }
        if (show_error) {
            ImGui::OpenPopup("Error");
            show_error = false; // Reset the flag so it only opens once
        }

        if (ImGui::BeginPopupModal("Error", NULL,
                                   ImGuiWindowFlags_AlwaysAutoResize)) {
            ImGui::Text("%s", error_message.c_str());
            ImGui::Separator();

            if (ImGui::Button("OK")) {
                ImGui::CloseCurrentPopup();
                show_error = false;
            }

            ImGui::EndPopup();
        }

        render_a_frame(window);
    }

    if (camera_control->open) {
        for (int i = 0; i < num_cameras; i++) {
            close_camera(&ecams[i].camera, &cameras_params[i]);
        }
        delete[] cameras_params;
        delete[] ecams;
        delete[] cameras_select;
    }

    HostClient_StopNetThread();
    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    ctx.net.stop();

    return 0;
}
