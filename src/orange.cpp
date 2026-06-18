#include "camera.h"
#include "enet_utils.h"
#include "galvo_calib.h"
#include "global.h"
#include "gui.h"
#include "host_client_imgui.h"
#include "imgui.h"
#include "implot.h"
#include "realtime_tool.h"
#include "server_endpoints.h"
#include "utils.h"
#include "video_capture.h"
#include <ImGuiFileDialog.h>
#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <sys/stat.h>

// Control flags
std::atomic<bool> g_workerRunning{false};
std::atomic<bool> g_workerShouldStop{false};

// Optional: store last measurement for UI display
std::mutex g_resultsMutex;
std::vector<int> g_lastOffsets; // size = num_cameras

namespace {
constexpr std::chrono::seconds kIdleCameraRefreshInterval{3};

bool same_camera_device(const GigEVisionDeviceInfo &a,
                        const GigEVisionDeviceInfo &b) {
    return std::strcmp(a.serialNumber, b.serialNumber) == 0 &&
           std::strcmp(a.currentIp, b.currentIp) == 0 &&
           std::strcmp(a.nic.ip4Address, b.nic.ip4Address) == 0;
}

bool same_camera_list(const GigEVisionDeviceInfo *a, int a_count,
                      const GigEVisionDeviceInfo *b, int b_count) {
    if (a_count != b_count) {
        return false;
    }
    for (int i = 0; i < a_count; i++) {
        if (!same_camera_device(a[i], b[i])) {
            return false;
        }
    }
    return true;
}

void refresh_idle_camera_list(int max_cameras,
                              GigEVisionDeviceInfo *unsorted_device_info,
                              GigEVisionDeviceInfo *device_info,
                              int &cam_count, std::vector<bool> &check,
                              bool select_all_cameras) {
    std::vector<GigEVisionDeviceInfo> refreshed_unsorted(max_cameras);
    std::vector<GigEVisionDeviceInfo> refreshed_sorted(max_cameras);
    int refreshed_count =
        scan_cameras(max_cameras, refreshed_unsorted.data(), false);
    sort_cameras_ip(refreshed_unsorted.data(), refreshed_sorted.data(),
                    refreshed_count);

    if (same_camera_list(device_info, cam_count, refreshed_sorted.data(),
                         refreshed_count)) {
        return;
    }

    std::vector<std::string> selected_serials;
    selected_serials.reserve(check.size());
    for (int i = 0; i < cam_count && i < static_cast<int>(check.size()); i++) {
        if (check[i]) {
            selected_serials.emplace_back(device_info[i].serialNumber);
        }
    }

    cam_count = refreshed_count;
    for (int i = 0; i < cam_count; i++) {
        device_info[i] = refreshed_sorted[i];
    }
    for (int i = 0; i < cam_count; i++) {
        unsorted_device_info[i] = refreshed_unsorted[i];
    }

    check.assign(cam_count, false);
    for (int i = 0; i < cam_count; i++) {
        check[i] =
            select_all_cameras ||
            std::find(selected_serials.begin(), selected_serials.end(),
                      std::string(device_info[i].serialNumber)) !=
                selected_serials.end();
    }

    std::cout << "Camera list refreshed: " << cam_count << " camera"
              << (cam_count == 1 ? "" : "s") << " found." << std::endl;
}
} // namespace

void poll_ptp_offset_and_dump(int num_cameras, CameraEmergent *ecams,
                              CameraParams *cameras_params) {
    // Open CSV in append mode; create if not exists
    std::ofstream ofs("ptp_offsets.csv", std::ios::app);
    if (!ofs) {
        printf("Failed to open ptp_offsets.csv\n");
        g_workerRunning = false;
        return;
    }

    // Check if file is empty -> write header once
    bool needHeader = (ofs.tellp() == std::streampos(0));
    if (needHeader) {
        // Optional first column: timestamp
        ofs << "timestamp";
        for (int i = 0; i < num_cameras; ++i) {
            ofs << ","
                << cameras_params[i].camera_serial; // adjust field name/type
        }
        ofs << "\n";
        ofs.flush();
    }

    // NOTE: this worker must not issue any GVCP request (no ptp_camera_sync, no
    // EVT_CameraGetInt32Param) — the capture threads already talk GVCP to these
    // cameras every frame, and a second thread doing concurrent GVCP on the same
    // camera collides (GVCP ACK error 0300) and crashes the EVT SDK. PtpMode is
    // enabled automatically at stream start (PTP sync is always on); here we
    // only READ the cached per-frame offsets the capture threads publish.
    (void)ecams;

    // Prepare buffer for offsets
    std::vector<int> offsets(num_cameras);

    // Main loop
    while (!g_workerShouldStop.load()) {

        for (int i = 0; i < num_cameras && i < kMaxCameras; i++) {
            offsets[i] = g_cam_ptp_offset[i].load(std::memory_order_relaxed);
        }

        // 3) Append row to CSV
        // Optional timestamp in seconds since epoch
        auto now = std::chrono::system_clock::now();
        auto now_s = std::chrono::duration_cast<std::chrono::seconds>(
                         now.time_since_epoch())
                         .count();
        ofs << now_s;
        for (int i = 0; i < num_cameras; ++i) {
            ofs << "," << offsets[i];
        }
        ofs << "\n";
        ofs.flush(); // ensure data is written promptly

        // 4) Update latest offsets for UI (optional)
        {
            std::lock_guard<std::mutex> lock(g_resultsMutex);
            g_lastOffsets = offsets;
        }

        // 5) Throttle: PtpOffset only changes slowly, so ~10 Hz keeps the CSV
        //    small and readable instead of writing thousands of rows/second.
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    g_workerRunning = false;
}

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

    std::string orange_root_dir_str;
    std::string encoder_codec;
    std::string recording_root_dir_str;
    prepare_application_folders(orange_root_dir_str, recording_root_dir_str,
                                encoder_codec);

    std::string input_folder = recording_root_dir_str + "/exp/unsorted";
    std::string calib_yaml_folder = orange_root_dir_str + "/calib_yaml";

    std::vector<bool> check;
    for (int i = 0; i < cam_count; i++) {
        check.push_back(false);
    }
    CameraParams *cameras_params = nullptr;
    CameraEachSelect *cameras_select = nullptr;
    CameraEmergent *ecams = nullptr;
    std::vector<std::thread> camera_threads;
    GL_Texture *tex_gl = nullptr;
    int num_cameras = 0;
    CameraControl *camera_control =
        new CameraControl{false, false, false, false, false};

    int evt_buffer_size{100};
    PTPParams *ptp_params =
        new PTPParams{0, 0, 0, 0, false, false, false, false};

    std::string encoder_preset = "p1";

    ScrollingBuffer *realtime_plot_data = nullptr;
    bool show_realtime_plot = false;
    // PTP sync always on: all cameras share the single NIC (one PTP domain) and
    // elect a master among themselves, so every preview/record is synchronized
    // without a host-side grandmaster. (Recording force-enabled it anyway.)
    bool ptp_stream_sync = true;

    AppContext ctx; // ENetGuard constructed here (enet_initialize)

    const std::string endpoints_path =
        orange_root_dir_str + "/config/network/endpoints.json";

    std::vector<ServerEndpoint> endpoints;
    if (std::filesystem::exists(endpoints_path)) {
        try {
            endpoints = load_server_endpoints(endpoints_path);
        } catch (const std::exception &e) {
            fprintf(stderr, "warning: %s\n", e.what());
            fprintf(stderr,
                    "network mode unavailable until endpoints.json is fixed\n");
        }
    }

    std::vector<std::pair<std::string, int>> cams;
    cams.reserve(endpoints.size());
    for (const auto &ep : endpoints) {
        cams.emplace_back(ep.host, ep.port);
    }

    host_client_start_net_thread(ctx); // start dispatcher thread
    host_client_init(ctx, cams);

    std::vector<std::string> network_config_folders;
    int network_config_select = -1;
    std::string selected_network_folder = "rig_new";
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
    std::string calib_save_folder = recording_root_dir_str + "/exp";

    int local_config_select = 0;
    bool select_all_cameras = false;
    char *temp_string = (char *)malloc(128);
    *temp_string = '\0';
    bool save_image_all_ready{true};

    std::vector<std::string> color_temps = {"CT_Off",   "CT_2800K", "CT_3000K",
                                            "CT_4000K", "CT_5000K", "CT_6500K",
                                            "CT_Custom"};
    std::thread detection3d_thread;
    bool show_error = false;
    std::string error_message;

    int current_picture_format = 0;
    const char *picture_format_items[] = {"jpg", "tiff", "png"};
    std::string selected_picture_format = picture_format_items[0];

    HostClientCtx client_ctx{&selected_picture_format,
                             &calib_save_folder,
                             &network_config_select,
                             &network_config_folders,
                             &selected_network_folder,
                             device_info,
                             &cam_count,
                             &check,
                             &num_cameras,
                             &cameras_params,
                             &cameras_select,
                             &ecams,
                             &realtime_plot_data,
                             camera_control,
                             &detection3d_thread,
                             &calib_yaml_folder,
                             &input_folder,
                             &camera_threads,
                             ptp_params,
                             &encoder_codec,
                             &encoder_preset,
                             &evt_buffer_size,
                             &display_gpu_id,
                             &tex_gl,
                             &ptp_stream_sync};

    set_host_client_ctx(&client_ctx);

    auto last_idle_camera_refresh =
        std::chrono::steady_clock::now() - kIdleCameraRefreshInterval;

    while (!glfwWindowShouldClose(window->render_target)) {
        host_client_tick();
        create_new_frame();

        if (!camera_control->open) {
            auto now = std::chrono::steady_clock::now();
            if (now - last_idle_camera_refresh >=
                kIdleCameraRefreshInterval) {
                refresh_idle_camera_list(max_cameras, unsorted_device_info,
                                         device_info, cam_count, check,
                                         select_all_cameras);
                last_idle_camera_refresh = now;
            }
        }

        host_client_draw_gui();

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
                        if (encoder_codec == codecs[i]) {
                            codec_current = i;
                            break;
                        }
                    }
                }

                if (ImGui::Combo("Codec", &codec_current, codecs,
                                 IM_ARRAYSIZE(codecs))) {
                    encoder_codec = codecs[codec_current];
                }
            }

            {
                const char *presets[] = {"p1", "p3", "p5", "p7"};
                static int preset_current = -1;

                if (preset_current == -1) {
                    for (int i = 0; i < IM_ARRAYSIZE(presets); ++i) {
                        if (encoder_preset == presets[i]) {
                            preset_current = i;
                            break;
                        }
                    }
                }

                if (ImGui::Combo("Preset", &preset_current, presets,
                                 IM_ARRAYSIZE(presets))) {
                    encoder_preset = presets[preset_current];
                }
            }

            int fps_temp =
                streaming_target_fps.load(); // get the current atomic value

            if (ImGui::InputInt("Streaming FPS", &fps_temp)) {
                // Clamp if necessary
                if (fps_temp < 1)
                    fps_temp = 1;
                if (fps_temp > 240)
                    fps_temp = 240;
                streaming_target_fps.store(fps_temp);
            }

            if (camera_control->subscribe) {
                ImGui::EndDisabled();
            }

            if (camera_control->open) {
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

                if (camera_control->record_video) {
                    ImGui::BeginDisabled();
                }

                set_camera_properties(ecams, cameras_params, cameras_select,
                                      num_cameras, color_temps);

                if (camera_control->record_video) {
                    ImGui::EndDisabled();
                }

                ImGui::Checkbox("Show camera temperature", &show_realtime_plot);

                if (ImGui::TreeNode("Galvo Target Streaming")) {
                    static std::string galvo_ip = galvo_sender_params.target_ip;
                    static int galvo_port = galvo_sender_params.target_port;

                    HelpMarker(
                        "Streams the triangulated 3D target (world mm) over "
                        "UDP to the Windows galvo motor-control app. "
                        "Requires a camera pair with 3DStandoff detection "
                        "running.");

                    if (galvo_sender_params.enabled) {
                        ImGui::BeginDisabled();
                    }
                    input_text("Target IP", galvo_ip, 0);
                    ImGui::InputInt("Target Port", &galvo_port);
                    if (galvo_sender_params.enabled) {
                        ImGui::EndDisabled();
                    }

                    bool enabled = galvo_sender_params.enabled;
                    if (ImGui::Checkbox("Enable target streaming", &enabled)) {
                        if (enabled) {
                            if (galvo_sender_start(galvo_ip, galvo_port)) {
                                galvo_sender_params.target_ip = galvo_ip;
                                galvo_sender_params.target_port = galvo_port;
                                galvo_sender_params.enabled = true;
                            } else {
                                error_message =
                                    "Failed to start galvo target sender.";
                                show_error = true;
                            }
                        } else {
                            galvo_sender_stop_dummy_target();
                            galvo_sender_params.dummy_mode = false;
                            galvo_sender_stop();
                            galvo_sender_params.enabled = false;
                        }
                    }

                    if (!galvo_sender_params.enabled) {
                        ImGui::BeginDisabled();
                    }
                    bool dummy_mode = galvo_sender_params.dummy_mode;
                    if (ImGui::Checkbox("Send dummy target (test circle)",
                                        &dummy_mode)) {
                        if (dummy_mode) {
                            galvo_sender_start_dummy_target();
                        } else {
                            galvo_sender_stop_dummy_target();
                        }
                        galvo_sender_params.dummy_mode = dummy_mode;
                    }
                    HelpMarker(
                        "For testing the link without cameras: streams a "
                        "synthetic 300mm-radius circle (z=1000mm) instead of "
                        "real triangulated points -- the same test pattern "
                        "the receiver itself draws when idle.");
                    if (!galvo_sender_params.enabled) {
                        ImGui::EndDisabled();
                    }

                    if (galvo_sender_params.enabled) {
                        ImGui::TextColored(
                            ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                            "Streaming %s to %s:%d",
                            galvo_sender_params.dummy_mode ? "dummy target"
                                                           : "3D target",
                            galvo_sender_params.target_ip.c_str(),
                            galvo_sender_params.target_port);
                    }

                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Galvo Control Link")) {
                    static std::string link_ip = galvo_link_params.target_ip;
                    static int link_port = galvo_link_params.target_port;
                    static double aim_pan = 0.0, aim_tilt = 0.0;

                    HelpMarker(
                        "Request/reply control channel (GCC1/GCS1) to the "
                        "galvo motor-control app: query status, command raw "
                        "mirror angles, toggle calib mode, upload "
                        "calibration. Requires 'Allow remote control' "
                        "enabled on the galvo app.");

                    if (galvo_link_params.enabled) {
                        ImGui::BeginDisabled();
                    }
                    input_text("Control IP", link_ip, 0);
                    ImGui::InputInt("Control Port", &link_port);
                    if (galvo_link_params.enabled) {
                        ImGui::EndDisabled();
                    }

                    bool link_enabled = galvo_link_params.enabled;
                    if (ImGui::Checkbox("Enable control link",
                                        &link_enabled)) {
                        if (link_enabled) {
                            if (galvo_link_start(link_ip, link_port)) {
                                galvo_link_params.target_ip = link_ip;
                                galvo_link_params.target_port = link_port;
                                galvo_link_params.enabled = true;
                            } else {
                                error_message =
                                    "Failed to start galvo control link.";
                                show_error = true;
                            }
                        } else {
                            galvo_link_stop();
                            galvo_link_params.enabled = false;
                        }
                    }

                    if (galvo_link_params.enabled) {
                        GalvoStatus gstat;
                        double age = 0.0;
                        bool have =
                            galvo_link_last_status(&gstat, &age) && age < 2.0;
                        if (have) {
                            ImGui::TextColored(
                                ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                                "Link OK  pan %.2f  tilt %.2f deg  %s%s%s",
                                gstat.pan_deg, gstat.tilt_deg,
                                gstat.in_position ? "[in position] " : "",
                                gstat.calib_mode ? "[calib mode] " : "",
                                gstat.motors_enabled ? "" : "[motors off] ");
                            if (!gstat.remote_allowed) {
                                ImGui::TextColored(
                                    ImVec4(1.0f, 0.6f, 0.2f, 1.0f),
                                    "Remote control disabled on the galvo "
                                    "app -- motion commands will be "
                                    "refused.");
                            }
                            ImGui::Text(
                                "Travel limits: pan [%.1f, %.1f]  tilt "
                                "[%.1f, %.1f] deg",
                                gstat.pan_min, gstat.pan_max, gstat.tilt_min,
                                gstat.tilt_max);
                        } else {
                            ImGui::TextColored(
                                ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                                "No reply from %s:%d",
                                galvo_link_params.target_ip.c_str(),
                                galvo_link_params.target_port);
                        }

                        ImGui::SeparatorText("Manual aim (raw motor deg)");
                        if (!have) {
                            ImGui::BeginDisabled();
                        }
                        ImGui::InputDouble("pan (deg)", &aim_pan);
                        ImGui::InputDouble("tilt (deg)", &aim_tilt);
                        if (ImGui::Button("Move")) {
                            GalvoStatus rep;
                            if (!galvo_link_set_angles(aim_pan, aim_tilt,
                                                       &rep)) {
                                error_message =
                                    "Galvo move refused (err " +
                                    std::to_string(rep.err) + ").";
                                show_error = true;
                            }
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Stop Motion")) {
                            galvo_link_stop_motion();
                        }
                        ImGui::SameLine();
                        bool calib_mode = have && gstat.calib_mode;
                        if (ImGui::Checkbox("Calib mode", &calib_mode)) {
                            GalvoStatus rep;
                            if (!galvo_link_calib_mode(calib_mode, &rep)) {
                                error_message =
                                    "Galvo calib mode change refused (err " +
                                    std::to_string(rep.err) + ").";
                                show_error = true;
                            }
                        }
                        HelpMarker(
                            "Calib mode pauses target-stream aiming on the "
                            "galvo app so manual/sweep moves aren't fought. "
                            "It auto-expires there after 5s without control "
                            "traffic; the link's 2 Hz status poll keeps it "
                            "alive.");
                        if (!have) {
                            ImGui::EndDisabled();
                        }
                    }

                    ImGui::TreePop();
                }

                if (ImGui::TreeNode("Galvo Calibration (ChArUco)")) {
                    static GalvoCalibConfig gcal_cfg;
                    static int gcal_cam = -1;
                    static int gcal_dict = 3; // DICT_5X5_100
                    static bool gcal_inited = false;
                    static std::string gcal_board_path =
                        orange_root_dir_str + "/config/galvo_board.json";
                    int num_dicts;
                    const char *const *dict_names =
                        charuco_dictionary_names(&num_dicts);
                    if (!gcal_inited) {
                        gcal_inited = true;
                        charuco_board_spec_load(gcal_board_path,
                                                &gcal_cfg.board);
                        for (int i = 0; i < num_dicts; i++) {
                            if (gcal_cfg.board.dictionary == dict_names[i]) {
                                gcal_dict = i;
                            }
                        }
                        gcal_cfg.out_folder = calib_yaml_folder;
                        galvo_calib_load_persisted(gcal_cfg);
                    }
                    gcal_cfg.num_cameras = num_cameras;
                    gcal_cfg.cameras_select = cameras_select;
                    gcal_cfg.cameras_params = cameras_params;
                    gcal_cfg.galvo_cam = gcal_cam;
                    gcal_cfg.out_folder = calib_yaml_folder;

                    GalvoCalibStatusView cs = galvo_calib_status();
                    bool gcal_busy = galvo_calib_busy();

                    HelpMarker(
                        "Automated galvo <-> world calibration: the mirrors "
                        "sweep a pan/tilt grid while a ChArUco board is "
                        "detected in the galvo camera (gaze point) and a "
                        "calibrated fixed camera (world pose). The fit is "
                        "uploaded to the galvo app over the control link. "
                        "See docs/galvo_calibration_plan.md.");

                    // --- preflight -------------------------------------
                    GalvoStatus gstat;
                    double gage = 0.0;
                    bool link_ok = galvo_link_params.enabled &&
                                   galvo_link_last_status(&gstat, &gage) &&
                                   gage < 2.0;
                    bool streaming = camera_control->subscribe;
                    bool fixed_ok = false;
                    if (streaming && detection2d != nullptr) {
                        for (int i = 0; i < num_cameras; i++) {
                            if (i != gcal_cam &&
                                cameras_select[i].stream_on &&
                                detection2d[i].has_calibration_results) {
                                fixed_ok = true;
                            }
                        }
                    }
                    auto checkline = [](bool ok, const char *label) {
                        ImGui::TextColored(ok ? ImVec4(0.2f, 1.0f, 0.2f, 1.0f)
                                              : ImVec4(1.0f, 0.4f, 0.4f, 1.0f),
                                           "%s %s", ok ? "[ok]" : "[--]",
                                           label);
                    };
                    checkline(link_ok, "control link alive");
                    checkline(link_ok && gstat.remote_allowed,
                              "remote control allowed on the galvo app");
                    checkline(streaming, "cameras streaming");
                    checkline(gcal_cam >= 0, "galvo camera selected");
                    checkline(fixed_ok,
                              "a calibrated fixed camera is streaming");
                    bool preflight = link_ok && gstat.remote_allowed &&
                                     streaming && gcal_cam >= 0 && fixed_ok;

                    // --- galvo camera + board ---------------------------
                    const char *cam_preview =
                        gcal_cam >= 0 && gcal_cam < num_cameras
                            ? cameras_params[gcal_cam].camera_serial.c_str()
                            : "(select)";
                    if (ImGui::BeginCombo("Galvo camera", cam_preview)) {
                        for (int i = 0; i < num_cameras; i++) {
                            if (ImGui::Selectable(
                                    cameras_params[i].camera_serial.c_str(),
                                    gcal_cam == i)) {
                                gcal_cam = i;
                            }
                        }
                        ImGui::EndCombo();
                    }
                    if (ImGui::TreeNode("Board")) {
                        ImGui::InputInt("squares x", &gcal_cfg.board.squares_x);
                        ImGui::InputInt("squares y", &gcal_cfg.board.squares_y);
                        ImGui::InputFloat("square (mm)",
                                          &gcal_cfg.board.square_mm);
                        ImGui::InputFloat("marker (mm)",
                                          &gcal_cfg.board.marker_mm);
                        if (ImGui::Combo("dictionary", &gcal_dict, dict_names,
                                         num_dicts)) {
                            gcal_cfg.board.dictionary = dict_names[gcal_dict];
                        }
                        if (ImGui::Button("Save board config")) {
                            charuco_board_spec_save(gcal_board_path,
                                                    gcal_cfg.board);
                        }
                        ImGui::SameLine();
                        if (!streaming || gcal_busy) {
                            ImGui::BeginDisabled();
                        }
                        if (ImGui::Button("Test board detection")) {
                            galvo_calib_test_detection(gcal_cfg);
                        }
                        if (!streaming || gcal_busy) {
                            ImGui::EndDisabled();
                        }
                        HelpMarker(
                            "Grabs one frame from every streaming camera and "
                            "reports markers/corners found. Zero everywhere "
                            "almost always means the wrong dictionary; a few "
                            "markers but zero corners means squares x/y (or "
                            "their order) doesn't match the print.");
                        ImGui::TreePop();
                    }

                    // --- sweep ------------------------------------------
                    ImGui::SetNextItemWidth(80.0f);
                    ImGui::InputDouble("pan range +/- deg",
                                       &gcal_cfg.pan_half_range_deg);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80.0f);
                    ImGui::InputDouble("tilt range +/- deg",
                                       &gcal_cfg.tilt_half_range_deg);
                    HelpMarker(
                        "The sweep grid is centered on the zeroed/home pose "
                        "(0,0) -- home the mirrors facing the arena first. "
                        "Keep the range within where the camera actually "
                        "sees out of the mirrors; the travel limits are "
                        "mechanical, not optical.");
                    ImGui::SetNextItemWidth(80.0f);
                    ImGui::InputInt("grid pan", &gcal_cfg.grid_pan);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80.0f);
                    ImGui::InputInt("grid tilt", &gcal_cfg.grid_tilt);
                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(80.0f);
                    ImGui::InputInt("board positions",
                                    &gcal_cfg.placements_target);

                    bool can_start = preflight && !gcal_busy;
                    if (!can_start) {
                        ImGui::BeginDisabled();
                    }
                    if (ImGui::Button("Start calibration sweep")) {
                        galvo_calib_start_sweep(gcal_cfg);
                    }
                    if (!can_start) {
                        ImGui::EndDisabled();
                    }

                    if (cs.phase == GalvoCalib_Sweep ||
                        cs.phase == GalvoCalib_WaitBoard) {
                        ImGui::SameLine();
                        if (ImGui::Button("Abort")) {
                            galvo_calib_abort();
                        }
                        ImGui::Text("placement %d  grid %d/%d  accepted %d "
                                    "(total %d)",
                                    cs.placement, cs.grid_index, cs.grid_total,
                                    cs.accepted_this_placement,
                                    cs.accepted_total);
                        // coverage map: . pending  x skipped  O accepted
                        for (int it = 0; it < cs.grid_tilt; it++) {
                            std::string row;
                            for (int ip = 0; ip < cs.grid_pan; ip++) {
                                uint8_t v =
                                    cs.coverage[it * cs.grid_pan + ip];
                                row += v == 2 ? " O" : v == 1 ? " x" : " .";
                            }
                            ImGui::TextUnformatted(row.c_str());
                        }
                    }
                    if (cs.phase == GalvoCalib_WaitBoard) {
                        ImGui::TextColored(ImVec4(1.0f, 0.8f, 0.2f, 1.0f),
                                           "Move the board (new distance), "
                                           "then continue.");
                        if (ImGui::Button("Board moved -- continue")) {
                            galvo_calib_next_placement();
                        }
                        ImGui::SameLine();
                        if (ImGui::Button("Finish with current samples")) {
                            galvo_calib_finish_collection();
                        }
                    }

                    // --- status + result --------------------------------
                    if (!cs.message.empty()) {
                        ImGui::TextWrapped("%s", cs.message.c_str());
                    }
                    if (cs.fit_valid) {
                        ImGui::Separator();
                        ImGui::Text("Fit: RMS %.3f deg   base (%.0f, %.0f, "
                                    "%.0f) mm   rot (%.1f, %.1f, %.1f) deg",
                                    cs.fit_rms_deg, cs.model.base[0],
                                    cs.model.base[1], cs.model.base[2],
                                    cs.model.rot[0], cs.model.rot[1],
                                    cs.model.rot[2]);
                        ImGui::Text(
                            "pan %.0f x %.3f + %.2f   tilt %.0f x %.3f + %.2f",
                            cs.model.pan_sign, cs.model.pan_scale,
                            cs.model.pan_offset, cs.model.tilt_sign,
                            cs.model.tilt_scale, cs.model.tilt_offset);
                        static std::string upload_msg;
                        if (!link_ok || gcal_busy) {
                            ImGui::BeginDisabled();
                        }
                        if (ImGui::Button("Upload to galvo & save")) {
                            GalvoStatus rep;
                            if (galvo_link_set_calib(cs.model, &rep) &&
                                galvo_link_save_config(&rep)) {
                                upload_msg =
                                    "Uploaded and saved on the galvo app.";
                            } else {
                                upload_msg = "Upload failed (err " +
                                             std::to_string(rep.err) + ").";
                            }
                        }
                        if (!link_ok || gcal_busy) {
                            ImGui::EndDisabled();
                        }
                        ImGui::SameLine();
                        bool can_verify = preflight && !gcal_busy &&
                                          galvo_sender_params.enabled &&
                                          !galvo_sender_params.dummy_mode;
                        if (!can_verify) {
                            ImGui::BeginDisabled();
                        }
                        if (ImGui::Button("Verify (board center)")) {
                            galvo_calib_start_verify(gcal_cfg);
                        }
                        if (!can_verify) {
                            ImGui::EndDisabled();
                        }
                        HelpMarker(
                            "Streams the board center as a normal GCT1 "
                            "target and measures where it lands in the galvo "
                            "view. Needs target streaming enabled and 'aim "
                            "at network target' on the galvo app.");
                        if (!upload_msg.empty()) {
                            ImGui::TextUnformatted(upload_msg.c_str());
                        }
                        if (cs.verify_err_mm >= 0.0) {
                            ImGui::Text("verify error: %.1f mm on the board "
                                        "plane",
                                        cs.verify_err_mm);
                        }
                    }

                    ImGui::TreePop();
                }

                if (ImGui::Button("Start PTP Logging")) {
                    if (!g_workerRunning) {
                        g_workerShouldStop = false;
                        g_workerRunning = true;

                        // Start worker thread
                        std::thread([num_cameras, ecams, cameras_params] {
                            poll_ptp_offset_and_dump(num_cameras, ecams,
                                                     cameras_params);
                        }).detach();
                    }
                }

                ImGui::SameLine();
                if (ImGui::Button("Stop PTP Logging")) {
                    g_workerShouldStop = true;
                }

                if (g_workerRunning) {
                    ImGui::Text("Status: logging to ptp_offsets.csv");
                } else {
                    ImGui::Text("Status: stopped");
                }

                // Optional: display last offsets
                {
                    std::lock_guard<std::mutex> lock(g_resultsMutex);
                    if (!g_lastOffsets.empty()) {
                        for (int i = 0; i < num_cameras; ++i) {
                            ImGui::Text("Cam %d (%s): %d", i,
                                        cameras_params[i].camera_serial.c_str(),
                                        g_lastOffsets[i]);
                        }
                    }
                }

                if (camera_control->subscribe == true) {
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

                    ImGui::SameLine();
                    ImGui::SetNextItemWidth(100.0f);

                    if (ImGui::Combo("Picture format", &current_picture_format,
                                     picture_format_items,
                                     IM_ARRAYSIZE(picture_format_items))) {
                        selected_picture_format = std::string(
                            picture_format_items[current_picture_format]);
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].frame_save_format =
                                selected_picture_format;
                        }
                    }

                    save_image_all_ready = true;
                    for (int i = 0; i < num_cameras; i++) {
                        if (cameras_select[i].sigs->frame_save_state.load() !=
                            State_Frame_Idle) {
                            save_image_all_ready = false;
                            break;
                        }
                    }

                    if (!save_image_all_ready) {
                        ImGui::BeginDisabled();
                    }

                    ImGui::SameLine();
                    if (ImGui::Button("Save pictures")) {
                        make_folder(picture_save_folder);
                        std::string frame_save_name =
                            get_current_time_milliseconds();
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].frame_save_name = frame_save_name;
                            cameras_select[i].frame_save_format =
                                selected_picture_format;
                            cameras_select[i].picture_save_folder =
                                picture_save_folder;
                            cameras_select[i].sigs->frame_save_state.store(
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
                    std::vector<std::string> camera_config_files;
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
                        int opened_cameras = 0;
                        int attempted_camera = -1;
                        try {
                            cameras_params = new CameraParams[num_cameras];
                            cameras_select =
                                new CameraEachSelect[num_cameras];

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
                                        camera_config_files,
                                        selected_cameras[i], num_cameras)) {
                                    skip_setting_params[i] = true;
                                    cameras_params[i].camera_id =
                                        selected_cameras[i];
                                    cameras_params[i].num_cameras =
                                        num_cameras;
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

                                if (cameras_params[i].camera_name ==
                                    "shelter") {
                                    cameras_select[i].stream_on = true;
                                }
                            }

                            ecams = new CameraEmergent[num_cameras];
                            for (int i = 0; i < num_cameras; i++) {
                                attempted_camera = i;
                                if (!skip_setting_params[i]) {
                                    open_camera_with_params(
                                        &ecams[i].camera,
                                        &device_info
                                            [cameras_params[i].camera_id],
                                        &cameras_params[i]);
                                } else {
                                    update_camera_params(
                                        &ecams[i].camera,
                                        &device_info
                                            [cameras_params[i].camera_id],
                                        &cameras_params[i]);
                                }
                                opened_cameras++;
                            }
                            realtime_plot_data =
                                new ScrollingBuffer[num_cameras];
                            camera_control->open = true;
                        } catch (const CameraError &e) {
                            for (int i = 0; i < opened_cameras; i++) {
                                EVT_CameraClose(&ecams[i].camera);
                            }
                            if (attempted_camera >= opened_cameras &&
                                attempted_camera < num_cameras) {
                                EVT_CameraClose(
                                    &ecams[attempted_camera].camera);
                            }
                            delete[] realtime_plot_data;
                            delete[] cameras_params;
                            delete[] cameras_select;
                            delete[] ecams;
                            realtime_plot_data = nullptr;
                            cameras_params = nullptr;
                            cameras_select = nullptr;
                            ecams = nullptr;
                            camera_control->open = false;
                            num_cameras = 0;

                            if (e.error_code == EVT_ERROR_GVCP_ACK) {
                                error_message =
                                    "Camera communication failed with a GVCP "
                                    "ACK error.\n\nPower cycle the cameras, "
                                    "then try opening them again.";
                            } else {
                                error_message =
                                    "Camera open failed for " +
                                    e.camera_serial + ":\n" + e.what();
                            }
                            show_error = true;
                        } catch (const std::exception &e) {
                            for (int i = 0; i < opened_cameras; i++) {
                                EVT_CameraClose(&ecams[i].camera);
                            }
                            delete[] realtime_plot_data;
                            delete[] cameras_params;
                            delete[] cameras_select;
                            delete[] ecams;
                            realtime_plot_data = nullptr;
                            cameras_params = nullptr;
                            cameras_select = nullptr;
                            ecams = nullptr;
                            camera_control->open = false;
                            num_cameras = 0;

                            error_message =
                                std::string("Camera open failed:\n") +
                                e.what();
                            show_error = true;
                        }
                    }
                } else {
                    camera_control->open = false;
                    for (int i = 0; i < num_cameras; i++) {
                        close_camera(&ecams[i].camera, &cameras_params[i]);
                    }
                    delete[] realtime_plot_data;
                    delete[] cameras_params;
                    delete[] cameras_select;
                    delete[] ecams;
                    realtime_plot_data = nullptr;
                    cameras_params = nullptr;
                    cameras_select = nullptr;
                    ecams = nullptr;
                    num_cameras = 0;
                }
            }
            if (camera_control->subscribe) {
                ImGui::EndDisabled();
            }

            if (!camera_control->record_video && camera_control->open) {
                if (camera_control->subscribe) {
                    ImGui::BeginDisabled();
                }
                // PTP Stream Sync is always on (see ptp_stream_sync init). All
                // cameras sit on the single NIC, i.e. one PTP domain — they
                // elect a master among themselves (BMCA), so sync needs no
                // host-side grandmaster and no toggle here.
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
                            evt_buffer_size, ptp_stream_sync, "", "",
                            ptp_params, calib_yaml_folder, detection3d_thread,
                            &ctx);
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
                        std::string encoder_setup = "-codec " + encoder_codec +
                                                    " -preset " +
                                                    encoder_preset;
                        camera_control->record_video = true;
                        std::string folder_name =
                            input_folder + "/" + get_current_date_time();
                        make_folder(folder_name);
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
                            folder_name, ptp_params, calib_yaml_folder,
                            detection3d_thread, &ctx);
                    } else {
                        camera_control->subscribe = false;
                        stop_camera_streaming(
                            camera_threads, camera_control, ecams,
                            cameras_params, cameras_select, num_cameras,
                            evt_buffer_size, ptp_params, detection3d_thread);
                        // Keep PTP sync on for the next preview (always-PTP rig).
                        ptp_stream_sync = true;
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

                    using Clock = std::chrono::steady_clock;
                    auto now = Clock::now();
                    auto last =
                        cameras_select[i]
                            .camera_track_state->last_progress_time.load(
                                std::memory_order_relaxed);

                    float idle_seconds =
                        std::chrono::duration<float>(now - last).count();
                    bool hung =
                        idle_seconds > 3.0f; // e.g. hung if idle > 3 seconds

                    if (hung) {
                        ImGui::TextColored(ImVec4(1.0f, 0.2f, 0.2f, 1.0f),
                                           ICON_FK_CIRCLE); // red
                    } else {
                        ImGui::TextColored(ImVec4(0.2f, 1.0f, 0.2f, 1.0f),
                                           ICON_FK_CIRCLE); // green
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
                                    "##ball##" + std::to_string(i);
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

    galvo_sender_stop_dummy_target();
    galvo_sender_stop();
    galvo_link_stop();
    host_client_stop_net_thread();
    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    ctx.net.stop();

    return 0;
}
