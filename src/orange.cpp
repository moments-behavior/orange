#include "video_capture.h"
#include <iostream>
#include "camera.h"
#include "imgui.h"
#include "implot.h"
#include <ImGuiFileDialog.h>
#include "project.h"
#include "gui.h"
#include <sys/stat.h>
#include "NvEncoder/NvCodecUtils.h"
#include "network_base.h"
#include "enet_thread.h"
#include "global.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

#define display_gpu_id 0

int main(int argc, char **args) {
    ck(cudaSetDevice(display_gpu_id));

    gx_context *window = (gx_context *) malloc(sizeof(gx_context));
    *window = (gx_context){
        .swap_interval = 1, // use vsync
        .width = 1920,
        .height = 1080,
        .render_target_title = (char *) malloc(100), // window title
        .glsl_version = (char *) malloc(100)
    };

    render_initialize_target(window);

    int max_cameras = 20;
    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);

    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split(cwd, delimiter);
    std::string orange_root_dir_str = "/home/" + tokenized_path[2] + "/orange_data";
    prepare_application_folders(orange_root_dir_str);
    std::string input_folder = orange_root_dir_str + "/exp/unsorted";

    std::string yolo_model_folder = orange_root_dir_str + "/detect";
    std::string yolo_model = yolo_model_folder + "/rat_bbox.engine";

    bool check[cam_count]{0};
    CameraParams *cameras_params;
    CameraEachSelect *cameras_select;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    GL_Texture *tex;
    int num_cameras = 0;

    CameraControl *camera_control = new CameraControl{false, false, false, false};

    int evt_buffer_size{100};
    PTPParams *ptp_params = new PTPParams{0, 0, 0, 0, false, false, false, false};

    EncoderConfig *encoder_config = new EncoderConfig{
        "-codec h264 -preset p1 -fps ",
        "h264",
        "p1"
    };
    std::vector<std::string> camera_config_files;

    ScrollingBuffer *realtime_plot_data;
    bool show_realtime_plot = false;
    bool ptp_stream_sync = false;

    flatbuffers::FlatBufferBuilder *fb_builder = new flatbuffers::FlatBufferBuilder(1024);

    EnetContext server;
    if (enet_initialize(&server, 3333, 5)) {
        printf("Server Initiated\n");
    }
    ConnectedServer my_servers[2];
    intialize_servers(my_servers);

    INDIGOSignalBuilder indigo_signal_builder{};
    indigo_signal_builder = {
        .builder = fb_builder,
        .server = &server,
        .indigo_connection = nullptr
    };

    std::vector<std::string> network_config_folders;
    std::string network_start_folder_name = orange_root_dir_str + "/config/network";
    for (const auto &entry: std::filesystem::directory_iterator(network_start_folder_name)) {
        network_config_folders.push_back(entry.path().string());
    }
    int network_config_select = 0;

    std::vector<std::string> local_config_folders;
    std::string local_start_folder_name = orange_root_dir_str + "/config/local";
    for (const auto &entry: std::filesystem::directory_iterator(local_start_folder_name)) {
        local_config_folders.push_back(entry.path().string());
    }
    std::string picture_save_folder = orange_root_dir_str + "/pictures/" + get_current_date();
    std::string calib_save_folder = orange_root_dir_str + "/calibration/" + get_current_date();

    int local_config_select = 0;
    bool select_all_cameras = false;
    char *temp_string = (char *) malloc(64);
    *temp_string = '\0';
    bool save_image_all_ready = true;
    bool quite_enet = false;
    auto record_start_time = std::chrono::steady_clock::now();
    std::string elapsed_time;

    std::thread enet_thread = std::thread(&create_enet_thread, &server, my_servers, &indigo_signal_builder,
                                          &quite_enet);
    std::vector<std::string> color_temps = { "CT_Off", "CT_2800K", "CT_3000K", "CT_4000K", "CT_5000K", "CT_6500K", "CT_Custom"};
    
    while (!glfwWindowShouldClose(window->render_target)) {
        create_new_frame();
        if (ImGui::Begin("Network")) {
            if (ImGui::BeginTable("##Local Apps", 2,
                                  ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings |
                                  ImGuiTableFlags_Borders)) {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Indigo");
                ImGui::TableNextColumn();
                sprintf(temp_string, "Not connected");
                if (indigo_signal_builder.indigo_connection != nullptr) {
                    if (indigo_signal_builder.indigo_connection->state == ENET_PEER_STATE_CONNECTED) {
                        sprintf(temp_string, "Connected");
                    }
                }
                ImGui::Text("%s", temp_string);
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("Calibration");
                ImGui::TableNextColumn();
                if (indigo_signal_builder.indigo_connection != nullptr) {
                    ImGui::Text("%s", enum_names_calib_state()[calib_state]);
                } else {
                    ImGui::Text("%s", "Not connected");
                }
                ImGui::EndTable();
            }

            if (ImGui::BeginTable("Servers", 4,
                                  ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings |
                                  ImGuiTableFlags_Borders)) {
                for (int i = 0; i < 2; i++) {
                    sprintf(temp_string, "##servers%d", i);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", my_servers[i].name);
                    ImGui::TableNextColumn();

                    if (my_servers[i].peer != nullptr) {
                        if (my_servers[i].peer->state == ENET_PEER_STATE_CONNECTED) {
                            my_servers[i].connected = true;
                        }
                    } else {
                        my_servers[i].connected = false;
                    }

                    if (ImGui::Button(my_servers[i].connected ? "Disconnect" : "Connect")) {
                        if (my_servers[i].connected) {
                            enet_peer_disconnect(my_servers[i].peer, 0);
                        } else {
                            my_servers[i].peer = connect_peer(&server,
                                                              my_servers[i].ip_add[0],
                                                              my_servers[i].ip_add[1],
                                                              my_servers[i].ip_add[2],
                                                              my_servers[i].ip_add[3],
                                                              my_servers[i].port);
                        }
                    }
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", std::to_string(my_servers[i].num_cameras).c_str());
                    ImGui::TableNextColumn();

                    if (my_servers[i].connected) {
                        ImGui::Text("%s", FetchGame::EnumNamesManagerState()[my_servers[i].server_state]);
                    } else {
                        ImGui::Text("%s", "Not connected");
                    }
                }
                ImGui::EndTable();
            }

            for (int i = 0; i < network_config_folders.size(); i++) {
                std::vector<std::string> folder_token = string_split(network_config_folders[i], "/");
                sprintf(temp_string, folder_token.back().c_str());
                ImGui::RadioButton(temp_string, &network_config_select, i);
                if (i != network_config_folders.size() - 1)
                    ImGui::SameLine();
            }

            if (!camera_control->open && my_servers[0].server_state == FetchGame::ManagerState_IDLE && my_servers[1].
                server_state == FetchGame::ManagerState_IDLE && my_servers[0].connected && my_servers[1].connected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if (ImGui::Button("Open Cameras")) {
                    update_camera_configs(camera_config_files, network_config_folders[network_config_select]);
                    select_cameras_have_configs(camera_config_files, device_info, check, cam_count);
                    host_broadcast_open_cameras(fb_builder, &server, network_config_folders[network_config_select]);
                    // open cameras
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
                            set_camera_params(&cameras_params[i], &device_info[selected_cameras[i]],
                                              camera_config_files, selected_cameras[i], num_cameras);
                        }

                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].stream_on = false;
                            if (cameras_params[i].camera_name == "Cam16") {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].yolo = true;
                            }
                            if (cameras_params[i].camera_name == "shelter") {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].yolo = false;
                            }
                        }

                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id],
                                                    &cameras_params[i]);
                        }

                        realtime_plot_data = new ScrollingBuffer[num_cameras];
                    }
                    camera_control->open = true;
                }
                ImGui::PopStyleColor(1);
                ImGui::SameLine();
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.0f, 0.7f, 1.0f));
                if (ImGui::Button("Save to")) {
                    IGFD::FileDialogConfig config;
                    config.countSelectionMax = 1;
                    config.path = input_folder;
                    ImGuiFileDialog::Instance()->OpenDialog("ChooseRecordingDir", "Choose a Directory", nullptr, config);
                }
                ImGui::PopStyleColor(1);
                ImGui::SameLine();
                ImGui::SetWindowFontScale(1.5f); // 1.0 is default
                ImGui::Text("%s", input_folder.c_str());
                ImGui::SetWindowFontScale(1.0f); // Reset to normal
            }

            if (!camera_control->subscribe && my_servers[0].server_state == FetchGame::ManagerState_WAITTHREAD &&
                my_servers[1].server_state == FetchGame::ManagerState_WAITTHREAD) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if (ImGui::Button("Clients start camera threads")) {
                    std::string encoder_setup =
                            "-codec " + encoder_config->encoder_codec + " -preset " + encoder_config->encoder_preset +
                            " -fps ";
                    encoder_config->folder_name = input_folder + "/" + get_current_date_time();
                    make_folder(encoder_config->folder_name);
                    ptp_params->network_sync = true;
                    host_broadcast_start_threads(fb_builder, &server, encoder_config->folder_name, encoder_setup);
                    camera_control->record_video = true;

                    cudaSetDevice(display_gpu_id);
                    tex = new GL_Texture[num_cameras];
                    for (int i = 0; i < num_cameras; i++) {
                        if (cameras_select[i].stream_on) {
                            cudaStreamCreate(&tex[i].streams);
                            create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                            register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                            map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                            cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size,
                                                       &tex[i].cuda_resource);
                            create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                        }
                    }

                    start_camera_streaming(camera_threads, camera_control, ecams, cameras_params, cameras_select, tex,
                                           num_cameras, evt_buffer_size, true, encoder_setup,
                                           encoder_config->folder_name, ptp_params,
                                           &indigo_signal_builder, yolo_model);
                    record_start_time = std::chrono::steady_clock::now();
                    camera_control->subscribe = true;
                }
                ImGui::PopStyleColor(1);
            }

            if (my_servers[0].server_state == FetchGame::ManagerState_WAITSTART && my_servers[1].server_state ==
                FetchGame::ManagerState_WAITSTART) {
                // check network servers are ready as well as local computer
                if (ptp_params->ptp_counter == num_cameras) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                    if (ImGui::Button("Start Recording")) {
                        record_start_time = std::chrono::steady_clock::now();
                        // get the host ready, and then set global ptp time to start recording
                        unsigned long long ptp_time = get_current_PTP_time(&ecams[0].camera);
                        int delay_in_second = 3;
                        ptp_params->ptp_global_time = ((unsigned long long) delay_in_second) * 1000000000 + ptp_time;
                        host_broadcast_set_start_ptp(fb_builder, &server, ptp_params->ptp_global_time);
                        ptp_params->network_set_start_ptp = true;
                    }
                    ImGui::PopStyleColor(1);
                }
            }

            if (!ptp_params->network_set_stop_ptp && ptp_params->ptp_start_reached && my_servers[0].server_state ==
                FetchGame::ManagerState_WAITSTOP && my_servers[1].server_state == FetchGame::ManagerState_WAITSTOP) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if (ImGui::Button("Stop Recording")) {
                    unsigned long long ptp_time = get_current_PTP_time(&ecams[0].camera);
                    int delay_in_second = 3;
                    ptp_params->ptp_stop_time = ((unsigned long long) delay_in_second) * 1000000000 + ptp_time;
                    std::cout << ptp_params->ptp_stop_time << std::endl;
                    fb_builder->Clear();
                    FetchGame::ServerBuilder server_builder(*fb_builder);
                    server_builder.add_control(FetchGame::ServerControl_STOPRECORDING);
                    server_builder.add_ptp_global_time(ptp_params->ptp_stop_time);
                    auto my_server = server_builder.Finish();
                    fb_builder->Finish(my_server);
                    uint8_t *server_buffer = fb_builder->GetBufferPointer();
                    size_t server_buf_size = fb_builder->GetSize();
                    ENetPacket *enet_packet = enet_packet_create(server_buffer, server_buf_size, 0);
                    enet_host_broadcast(server.m_pNetwork, 0, enet_packet);
                    ptp_params->network_set_stop_ptp = true;
                }
                ImGui::PopStyleColor(1);
            }

            if (my_servers[0].server_state == FetchGame::ManagerState_IDLE && my_servers[1].server_state ==
                FetchGame::ManagerState_IDLE) {
                if (ImGui::Button("Clients close")) {
                    // broadcast data
                    fb_builder->Clear();
                    FetchGame::ServerBuilder server_builder(*fb_builder);
                    server_builder.add_control(FetchGame::ServerControl_QUIT);
                    auto my_server = server_builder.Finish();
                    fb_builder->Finish(my_server);
                    uint8_t *server_buffer = fb_builder->GetBufferPointer();
                    size_t server_buf_size = fb_builder->GetSize();
                    ENetPacket *enet_packet = enet_packet_create(server_buffer, server_buf_size, 0);
                    enet_host_broadcast(server.m_pNetwork, 0, enet_packet);
                }
            }
        }
        ImGui::End();

        if (ptp_params->network_set_stop_ptp && ptp_params->ptp_stop_reached) {
            ptp_params->network_set_stop_ptp = false;

            for (int i = 0; i < num_cameras; i++) {
                int size_pic = cameras_params[i].width * cameras_params[i].height * sizeof(unsigned char) * 4;
                cudaMemset(tex[i].cuda_buffer, 0, size_pic);
            }

            for (int i = 0; i < num_cameras; i++) {
                bind_pbo(&tex[i].pbo);
                bind_texture(&tex[i].texture);
                upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height);
                // Needs no arguments because texture and PBO are bound
                unbind_pbo();
                unbind_texture();
            }

            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on) {
                    gx_delete_buffer(&tex[i].pbo);
                    unmap_cuda_resource(&tex[i].cuda_resource);
                    cuda_unregister_pbo(tex[i].cuda_resource);
                    cudaStreamDestroy(tex[i].streams);
                }
            }
            delete[] tex;

            for (auto &t: camera_threads)
                t.join();

            for (int i = 0; i < num_cameras; i++) {
                camera_threads.pop_back();
            }
            for (int i = 0; i < num_cameras; i++) {
                destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size, &cameras_params[i]);
                delete[] ecams[i].evt_frame;
                check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera), cameras_params[i].camera_serial.c_str());
            }

            for (int i = 0; i < num_cameras; i++) {
                ptp_sync_off(&ecams[i].camera, &cameras_params[i]);
            }
            camera_control->sync_camera = false;
            camera_control->record_video = false;

            ptp_params->ptp_global_time = 0;
            ptp_params->ptp_stop_time = 0;
            ptp_params->ptp_counter = 0;
            ptp_params->ptp_stop_counter = 0;
            ptp_params->network_sync = false;
            ptp_params->network_set_start_ptp = false;
            ptp_params->ptp_stop_reached = false;
            ptp_params->ptp_start_reached = false;

            for (int i = 0; i < num_cameras; i++) {
                close_camera(&ecams[i].camera, &cameras_params[i]);
            }

            camera_control->open = false;

            for (int i = 0; i < cam_count; i++) {
                check[i] = 0;
            }
        }

        if (ImGui::Begin("Orange", nullptr, ImGuiWindowFlags_MenuBar)) {
            // ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate,
            //            ImGui::GetIO().Framerate);

            if (camera_control->open) {
                ImGui::BeginDisabled();
            }

            if (ImGui::BeginTable("Cameras", 3,
                                  ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings |
                                  ImGuiTableFlags_Borders)) {
                for (int i = 0; i < cam_count; i++) {
                    sprintf(temp_string, "%d", i);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Selectable(temp_string, &check[i], ImGuiSelectableFlags_SpanAllColumns);
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", device_info[i].serialNumber);
                    ImGui::TableNextColumn();
                    ImGui::Text("%s", device_info[i].currentIp);
                }
                ImGui::EndTable();
            }

            if (ImGui::Button(select_all_cameras ? "Clear all" : "Select all")) {
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

            // selection for yolo model
            if (ImGui::Button("Select YOLO")) {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = yolo_model_folder;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseYOLOFile", "Choose File", ".engine", config);
            }
            ImGui::SameLine();
            ImGui::Text("%s", yolo_model.c_str());

            if (camera_control->subscribe) {
                ImGui::EndDisabled();
            }

            if (camera_control->record_video) {
                ImGui::BeginDisabled();
            }

            ImGui::PushStyleColor(ImGuiCol_Button, ImVec4(0.5f, 0.0f, 0.7f, 1.0f));
            if (ImGui::Button("Save to")) {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = input_folder;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseRecordingDir", "Choose a Directory", nullptr, config);
            }
            ImGui::PopStyleColor(1); 
            ImGui::SameLine();
            ImGui::Text("%s", input_folder.c_str()); 

            {
                const char *items[] = {"h264", "hevc"};
                static int item_current = 0;
                ImGui::Combo("codec", &item_current, items, IM_ARRAYSIZE(items));
                encoder_config->encoder_codec = items[item_current];
            } 
            {
                const char *items[] = {"p1", "p3", "p5", "p7"};
                static int item_current = 0;
                ImGui::Combo("preset", &item_current, items, IM_ARRAYSIZE(items));
                encoder_config->encoder_preset = items[item_current];
            }

            if (camera_control->record_video) {
                ImGui::EndDisabled();
            }

            if (camera_control->open) {
                if (camera_control->record_video) {
                    ImGui::BeginDisabled();
                }

                ImGui::Checkbox("Show camera temperature", &show_realtime_plot);

                set_camera_properties(ecams, cameras_params, num_cameras, color_temps);

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
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings |
                                      ImGuiTableFlags_Borders)) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("name");
                    ImGui::TableNextColumn();
                    ImGui::Text("serial");
                    ImGui::TableNextColumn();
                    ImGui::Text("stream "); ImGui::SameLine();
                    if(ImGui::Checkbox("all##stream", &stream_all_cameras)) 
                    {
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
                    ImGui::Text("record "); ImGui::SameLine();
                    if(ImGui::Checkbox("all##record", &record_all_cameras)) 
                    {
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
                        ImGui::Text("%s", cameras_params[i].camera_name.c_str());
                        ImGui::TableNextColumn();
                        ImGui::Text("%s", cameras_params[i].camera_serial.c_str());
                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##checkbox_stream%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].stream_on);
                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##checkbox_record%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].record);
                        ImGui::TableNextColumn();                        
                        sprintf(temp_string, "##checkbox_yolo%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].yolo);
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
                        ImGuiFileDialog::Instance()->OpenDialog("ChoosePictureDir", "Choose a Directory", nullptr,
                                                                config);
                    }
                    ImGui::SameLine();
                    ImGui::Text("%s", picture_save_folder.c_str());
                    static int current_picture_format = 0;
                    const char* picture_format_items[] = { "jpg", "tiff", "png"};
                    ImGui::Combo("Picture format", &current_picture_format, picture_format_items, IM_ARRAYSIZE(picture_format_items));
                    for (int i = 0; i < num_cameras; i++) {
                        cameras_select[i].frame_save_format = std::string(picture_format_items[current_picture_format]);
                    }

                    if (ImGui::TreeNode("Save pictures from capturing")) {
                        save_image_all_ready = true;
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].frame_save_state != State_Frame_Idle) {
                                save_image_all_ready = false;
                                break;
                            }  
                        }

                        // for (int i = 0; i < num_cameras; i++) {
                        //     ImGui::Checkbox(cameras_params[i].camera_name.c_str(),
                        //                     &cameras_select[i].selected_to_save);
                        //     ImGui::SameLine();
                        //     ImGui::TextColored(ImVec4{1.0, 0.0f, 0, 1.0f}, "%d", cameras_select[i].pictures_counter);
                        //     ImGui::SameLine();
                        // }

                        const int cols = 5;
                        for (int i = 0; i < num_cameras; ++i) {     
                            std::string label = cameras_params[i].camera_name + ": " + std::to_string(cameras_select[i].pictures_counter) + "##calibration_save";
                            if (ImGui::Selectable(label.c_str(), cameras_select[i].selected_to_save,
                                                  ImGuiSelectableFlags_None,
                                                  ImVec2(150, 50))) {
                                cameras_select[i].selected_to_save = !cameras_select[i].selected_to_save;
                            }

                            // Keep items on the same line until end of row
                            if ((i + 1) % cols != 0)
                                ImGui::SameLine();
                        }


                        if (!save_image_all_ready) {
                            ImGui::BeginDisabled();
                        }

                        ImGui::NewLine();
                        if (ImGui::Button("Save selected")) {
                            make_folder(picture_save_folder);
                            std::string frame_save_name = get_current_time_milliseconds();
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = frame_save_name;
                                cameras_select[i].picture_save_folder = picture_save_folder;
                                if (cameras_select[i].selected_to_save) {
                                    cameras_select[i].frame_save_state = State_Write_New_Frame;
                                }
                            }
                        }
                        ImGui::SameLine();

                        if (ImGui::Button("Save pictures all")) {
                            make_folder(picture_save_folder);
                            std::string frame_save_name = get_current_time_milliseconds();
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = frame_save_name;
                                cameras_select[i].picture_save_folder = picture_save_folder;
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
                        }

                        if (calib_state == CalibSavePictures) {
                            send_indigo_message(indigo_signal_builder.server, indigo_signal_builder.builder, indigo_signal_builder.indigo_connection, FetchGame::SignalType_CalibrationNextPose);
                            calib_state = CalibNextPose;
                        }

                        ImGui::BeginDisabled(calib_state!=CalibPoseReached);
                        if (ImGui::Button("Calib save all")) {
                            make_folder(calib_save_folder);
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = std::to_string(cameras_select[i].pictures_counter);
                                cameras_select[i].picture_save_folder = calib_save_folder;
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
                            calib_state = CalibSavePictures;
                        }
                        ImGui::EndDisabled();

                        if (ImGui::Button("Calib save global images")) {
                            make_folder(calib_save_folder);
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = std::to_string(cameras_select[i].pictures_counter);
                                cameras_select[i].picture_save_folder = calib_save_folder;
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
                        }
                        
                        if (!save_image_all_ready) {
                            ImGui::EndDisabled();
                        }

                        ImGui::TreePop();
                    }
                }
            }
        }
        ImGui::End();

        // file explorer display
        if (ImGuiFileDialog::Instance()->Display("ChooseYOLOFile")) {
            // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {
                // action if OK
                yolo_model = ImGuiFileDialog::Instance()->GetFilePathName();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseRecordingDir")) {
            // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {
                // action if OK
                auto selected_folder = ImGuiFileDialog::Instance()->GetSelection();
                input_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }


        if (ImGuiFileDialog::Instance()->Display("ChoosePictureDir")) {
            // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk()) {
                // action if OK
                auto selected_folder = ImGuiFileDialog::Instance()->GetSelection();
                picture_save_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }


        if (ImGui::Begin("Local")) {
            if (camera_control->open) {
                ImGui::BeginDisabled();
            }

            for (int i = 0; i < local_config_folders.size(); i++) {
                std::vector<std::string> folder_token = string_split(local_config_folders[i], "/");
                sprintf(temp_string, folder_token.back().c_str());
                ImGui::RadioButton(temp_string, &local_config_select, i);
                ImGui::SameLine();
            }
            ImGui::RadioButton("Null", &local_config_select, local_config_folders.size());

            if (camera_control->open) {
                ImGui::EndDisabled();
            }

            if (camera_control->subscribe) {
                ImGui::BeginDisabled();
            }

            if (ImGui::Button(camera_control->open ? "Close Camera" : "Open camera")) {
                if (!camera_control->open) {
                    if (local_config_select < local_config_folders.size()) {
                        update_camera_configs(camera_config_files, local_config_folders[local_config_select]);
                        select_cameras_have_configs(camera_config_files, device_info, check, cam_count);
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
                            if (!set_camera_params(&cameras_params[i], &device_info[selected_cameras[i]],
                                                   camera_config_files, selected_cameras[i], num_cameras)) {
                                skip_setting_params[i] = true;
                                cameras_params[i].camera_id = selected_cameras[i];
                                cameras_params[i].num_cameras = num_cameras;
                            } else {
                                skip_setting_params[i] = false;

                            }
                        }

                        
                        for (int i = 0; i < num_cameras; i++) {
                            cameras_select[i].stream_on = false;
                            if (cameras_params[i].camera_name == "ceiling_center") {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].yolo = false;
                            }

                            if (cameras_params[i].camera_name == "shelter") {
                                cameras_select[i].stream_on = true;
                            }

                        }

                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            if (!skip_setting_params[i]) {
                                open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id],
                                                    &cameras_params[i]);                                
                            } else {
                                update_camera_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id],
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
                if (camera_control->subscribe) {
                    ImGui::EndDisabled();
                }
                if (ImGui::Button(camera_control->subscribe ? "Stop streaming" : "Start streaming")) {
                    (camera_control->subscribe) = !(camera_control->subscribe);
                    if (camera_control->subscribe) {
                        cudaSetDevice(display_gpu_id);
                        tex = new GL_Texture[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                cudaStreamCreate(&tex[i].streams);
                                create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                                register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                                map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                                cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size,
                                                           &tex[i].cuda_resource);
                                create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                            }
                        }

                        start_camera_streaming(camera_threads, camera_control, ecams, cameras_params, cameras_select,
                                               tex, num_cameras,
                                               evt_buffer_size, ptp_stream_sync, "",
                                               encoder_config->folder_name, ptp_params,
                                               &indigo_signal_builder, yolo_model);
                    } else {
                        for (int i = 0; i < num_cameras; i++) {
                            int size_pic = cameras_params[i].width * cameras_params[i].height * sizeof(unsigned char) *
                                           4;
                            cudaMemset(tex[i].cuda_buffer, 0, size_pic);
                        }

                        for (int i = 0; i < num_cameras; i++) {
                            bind_pbo(&tex[i].pbo);
                            bind_texture(&tex[i].texture);
                            upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height);
                            // Needs no arguments because texture and PBO are bound
                            unbind_pbo();
                            unbind_texture();
                        }

                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                gx_delete_buffer(&tex[i].pbo);
                                unmap_cuda_resource(&tex[i].cuda_resource);
                                cuda_unregister_pbo(tex[i].cuda_resource);
                                cudaStreamDestroy(tex[i].streams);
                            }
                        }
                        delete[] tex;

                        stop_camera_streaming(camera_threads, camera_control, ecams, cameras_params, cameras_select,
                                              num_cameras,
                                              evt_buffer_size, ptp_params);
                    }
                }
            }

            if (camera_control->stop_record) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.5f, 0, 0, 1.0f});
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
            }

            if (camera_control->open) {
                if (ImGui::Button(camera_control->stop_record ? ICON_FK_PAUSE : ICON_FK_PLAY)) {
                    (camera_control->stop_record) = !(camera_control->stop_record);
                    if (camera_control->stop_record) {
                        if (camera_control->subscribe) {
                            camera_control->subscribe = false;
                            // already streaming, turn the camera off
                            for (int i = 0; i < num_cameras; i++) {
                                int size_pic = cameras_params[i].width * cameras_params[i].height * sizeof(unsigned
                                                   char) * 4;
                                cudaMemset(tex[i].cuda_buffer, 0, size_pic);
                            }

                            for (int i = 0; i < num_cameras; i++) {
                                bind_pbo(&tex[i].pbo);
                                bind_texture(&tex[i].texture);
                                upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height);
                                // Needs no arguments because texture and PBO are bound
                                unbind_pbo();
                                unbind_texture();
                            }

                            for (int i = 0; i < num_cameras; i++) {
                                if (cameras_select[i].stream_on) {
                                    gx_delete_buffer(&tex[i].pbo);
                                    unmap_cuda_resource(&tex[i].cuda_resource);
                                    cuda_unregister_pbo(tex[i].cuda_resource);
                                    cudaStreamDestroy(tex[i].streams);
                                }
                            }
                            delete[] tex;

                            stop_camera_streaming(camera_threads, camera_control, ecams, cameras_params, cameras_select,
                                                  num_cameras,
                                                  evt_buffer_size, ptp_params);
                        }

                        std::string encoder_setup = "-codec " + encoder_config->encoder_codec + " -preset " + encoder_config->encoder_preset+ " -fps ";
                        camera_control->record_video = true;
                        encoder_config->folder_name = input_folder + "/" + get_current_date_time();
                        make_folder(encoder_config->folder_name);
                        if (num_cameras > 1) {
                            ptp_stream_sync = true;
                        } else {
                            ptp_stream_sync = false;
                        }

                        cudaSetDevice(display_gpu_id);
                        tex = new GL_Texture[num_cameras];
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                cudaStreamCreate(&tex[i].streams);
                                create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                                register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                                map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                                cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size,
                                                           &tex[i].cuda_resource);
                                create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                            }
                        }

                        start_camera_streaming(camera_threads, camera_control, ecams, cameras_params, cameras_select,
                                               tex, num_cameras, evt_buffer_size, ptp_stream_sync, encoder_setup,
                                               encoder_config->folder_name, ptp_params,
                                               &indigo_signal_builder, yolo_model);
                        record_start_time = std::chrono::steady_clock::now();
                        camera_control->subscribe = true;
                    } else {
                        camera_control->subscribe = false;
                        ptp_stream_sync = false;

                        for (int i = 0; i < num_cameras; i++) {
                            int size_pic = cameras_params[i].width * cameras_params[i].height * sizeof(unsigned char) *
                                           4;
                            cudaMemset(tex[i].cuda_buffer, 0, size_pic);
                        }

                        for (int i = 0; i < num_cameras; i++) {
                            bind_pbo(&tex[i].pbo);
                            bind_texture(&tex[i].texture);
                            upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height);
                            // Needs no arguments because texture and PBO are bound
                            unbind_pbo();
                            unbind_texture();
                        }

                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                gx_delete_buffer(&tex[i].pbo);
                                unmap_cuda_resource(&tex[i].cuda_resource);
                                cuda_unregister_pbo(tex[i].cuda_resource);
                                cudaStreamDestroy(tex[i].streams);
                            }
                        }
                        delete[] tex;

                        stop_camera_streaming(camera_threads, camera_control, ecams, cameras_params, cameras_select,
                                              num_cameras,
                                              evt_buffer_size, ptp_params);
                        camera_control->record_video = false;
                    }
                }
            }

            ImGui::PopStyleColor(1);
        }
        ImGui::End();

        if (camera_control->subscribe) {
            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on) {
                    bind_pbo(&tex[i].pbo);
                    bind_texture(&tex[i].texture);
                    upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height);
                    // Needs no arguments because texture and PBO are bound
                    unbind_pbo();
                    unbind_texture();
                }
            }

            if (camera_control->record_video) {
                auto now = std::chrono::steady_clock::now();
                auto elapsed = std::chrono::duration_cast<std::chrono::seconds>(now - record_start_time);
                elapsed_time = format_elapsed_time(elapsed);
            }

            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on) {
                    std::string window_name = cameras_params[i].camera_name;
                    ImGui::Begin(window_name.c_str());
                    if (camera_control->record_video) {
                        ImGui::TextColored(ImVec4{1.0, 1.0f, 0, 1.0f}, "Elapsed Time: %s", elapsed_time.c_str());
                    }

                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    static ImVec2 bmin(0, 0);
                    static ImVec2 uv0(0, 0);
                    static ImVec2 uv1(1, 1);
                    static ImVec4 tint(1, 1, 1, 1);

                    // ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                    ImPlotAxisFlags axisFlags = ImPlotAxisFlags_NoTickLabels | ImPlotAxisFlags_NoTickMarks |
                                                ImPlotAxisFlags_NoGridLines;
                    if (ImPlot::BeginPlot("##no_plot_name", avail_size, ImPlotFlags_Equal | ImPlotAxisFlags_AutoFit)) {
                        ImPlot::SetupAxesLimits(0, cameras_params[i].width, 0, cameras_params[i].height);
                        ImPlot::SetupAxis(ImAxis_X1, nullptr, axisFlags); // X-axis
                        ImPlot::SetupAxis(ImAxis_Y1, nullptr, axisFlags); // Y-axis
                        ImPlot::PlotImage("##no_image_name", (void *) (intptr_t) tex[i].texture, ImVec2(0, 0),
                                          ImVec2(cameras_params[i].width, cameras_params[i].height));

                        ImPlot::EndPlot();
                    }
                    ImGui::End();
                }
            }
        }

        if (camera_control->open && show_realtime_plot) {
            ImGui::Begin("Realtime Plots"); {
                static float t = 0;
                t += ImGui::GetIO().DeltaTime;
                for (int i = 0; i < num_cameras; i++) {
                    get_senstemp_value(&ecams[i].camera, &cameras_params[i]);
                    realtime_plot_data[i].AddPoint(t, cameras_params[i].sens_temp);
                }

                static float history = 10.0f;
                ImGui::SliderFloat("History", &history, 1, 30, "%.1f s");

                static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickMarks;
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                if (ImPlot::BeginPlot("Camera Sensor Temperature", avail_size)) {
                    ImPlot::SetupAxes(nullptr, nullptr, flags, flags);
                    ImPlot::SetupAxisLimits(ImAxis_X1, t - history, t, ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1, 30, 90);
                    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL, 0.5f);

                    for (int i = 0; i < num_cameras; i++) {
                        std::string line_name = std::string(cameras_params[i].camera_serial);
                        ImPlot::PlotLine(line_name.c_str(), &realtime_plot_data[i].Data[0].x,
                                         &realtime_plot_data[i].Data[0].y, realtime_plot_data[i].Data.size(), 0,
                                         realtime_plot_data[i].Offset, 2 * sizeof(float));
                    }
                    ImPlot::EndPlot();
                }
                ImGui::End();
            }
        }

        render_a_frame(window);
    }

    if (camera_control->open) {
        for (int i = 0; i < num_cameras; i++) {
            close_camera(&ecams[i].camera, &cameras_params[i]);
        }
        delete[] cameras_params;
        delete[] ecams;
    }

    quite_enet = true;
    enet_thread.join();
    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    enet_release(&server);
    return 0;
}
