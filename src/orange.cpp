#include "video_capture.h"
#include <iostream>
#include "camera.h"
#include "imgui.h"
#include "implot.h"
#include <ImGuiFileDialog.h>
#include "project.h"
#include "gui.h"
#include <sys/stat.h>
#include <cuda.h>
#include "NvEncoder/NvCodecUtils.h"
#include "network_base.h"
#include "enet_thread.h"
#include "yolo_worker.h"
#include "global.h"
#include "gpu_video_encoder.h"
#include "opengldisplay.h"
#include "image_writer_worker.h"

std::vector<YOLOv8Worker*> yolo_workers; // For managing YOLO workers
ENetPeer* external_data_consumer_peer = nullptr; // Store the peer for YOLO data

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

#define display_gpu_id 0

int main(int argc, char **args) {
    // --- REFACTOR 1: ESTABLISH THE PRIMARY CUDA CONTEXT ---
    // This context will be created once and shared by all threads.
    ck(cuInit(0));
    CUdevice cuDevice;
    CUcontext cuContext;
    ck(cuDeviceGet(&cuDevice, display_gpu_id));
    ck(cuCtxCreate(&cuContext, 0, cuDevice));
    ck(cuCtxPushCurrent(cuContext)); // Set for the main thread

    gx_context *window = (gx_context *) malloc(sizeof(gx_context));
    *window = (gx_context){
        .swap_interval = 1,
        .width = 1920,
        .height = 1080,
        .render_target_title = (char *) "Orange",
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
    std::string yolo_model = yolo_model_folder + "/fish_jinyao.engine";
    
    bool camera_is_selected[cam_count]{0};
    CameraParams *cameras_params;
    CameraEachSelect *cameras_select;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    GL_Texture *tex;
    int num_cameras = 0;
    int stream_downsample = 1;
    CameraControl *camera_control = new CameraControl{false, false, false, false};

    int evt_buffer_size{100};
    PTPParams *ptp_params = new PTPParams{0, 0, 0, 0, false, false, false, false};
    const int ACQUIRE_WORK_ENTRIES_MAX = 120;
    WORKER_ENTRY* worker_entry_pool = nullptr;
    SafeQueue<WORKER_ENTRY*>* free_entries_queue = nullptr;
    SafeQueue<WORKER_ENTRY*>* recycle_queue = nullptr;
    COpenGLDisplay** openGLDisplayWorkers = nullptr;
    GPUVideoEncoder** gpuVideoEncoders = nullptr; // Define pointer, but allocate later
    ImageWriterWorker* image_writer = new ImageWriterWorker("ImageSaverThread", cuContext);
    image_writer->StartThread();

    EncoderConfig *encoder_config = new EncoderConfig{
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
    std::string calib_save_folder = orange_root_dir_str + "/exp/calibration/" + get_current_date();

    int local_config_select = 0;
    bool select_all_cameras = false;
    char *temp_string = (char *) malloc(64);
    *temp_string = '\0';
    bool save_image_all_ready = true;
    bool quite_enet = false;

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
                const std::string& label = folder_token.back();
            
                // Highlight "rig_new" in purple
                if (label == "rig_new") {
                    ImGui::PushStyleColor(ImGuiCol_Text, ImVec4(1.0f, 0.55f, 0.0f, 1.0f));  
                }
            
                sprintf(temp_string, label.c_str());
                ImGui::RadioButton(temp_string, &network_config_select, i);
            
                if (label == "rig_new") {
                    ImGui::PopStyleColor();
                }
            
                if (i != network_config_folders.size() - 1)
                    ImGui::SameLine();
            }

            if (!camera_control->open && my_servers[0].server_state == FetchGame::ManagerState_IDLE && my_servers[1].
                server_state == FetchGame::ManagerState_IDLE && my_servers[0].connected && my_servers[1].connected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if (ImGui::Button("Open Cameras")) {
                    update_camera_configs(camera_config_files, network_config_folders[network_config_select]);
                    select_cameras_have_configs(camera_config_files, device_info, camera_is_selected, cam_count);
                    host_broadcast_open_cameras(fb_builder, &server, network_config_folders[network_config_select]);
                    // open cameras
                    num_cameras = 0;
                    for (int i = 0; i < cam_count; i++) {
                        if (camera_is_selected[i]) {
                            num_cameras++;
                        }
                    }
                    if (num_cameras > 0) {
                        cameras_params = new CameraParams[num_cameras]();
                        cameras_select = new CameraEachSelect[num_cameras]();

                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++) {
                            if (camera_is_selected[i]) {
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
                    // --- This button now does two things ---
                    // 1. It sends the command to the clients to start their threads.
                    // 2. It starts its own local threads using the same robust, centralized logic.
            
                    // --- Logic for telling clients to start ---
                    std::string encoder_setup_for_clients =
                            "-codec " + encoder_config->encoder_codec + " -preset " + encoder_config->encoder_preset +
                            " -fps ";
                    encoder_config->folder_name = input_folder + "/" + get_current_date_time();
                    make_folder(encoder_config->folder_name);
                    ptp_params->network_sync = true;
                    host_broadcast_start_threads(fb_builder, &server, encoder_config->folder_name, encoder_setup_for_clients);
                    
                    // --- Centralized Startup Logic (for local cameras) ---
                    (camera_control->subscribe) = true; // Set the master flag to true
                    camera_control->record_video = true; // This button implies recording
            
                    // 1. ALLOCATE SHARED RESOURCES
                    worker_entry_pool = new WORKER_ENTRY[ACQUIRE_WORK_ENTRIES_MAX];
                    free_entries_queue = new SafeQueue<WORKER_ENTRY*>();
                    recycle_queue = new SafeQueue<WORKER_ENTRY*>();
                    openGLDisplayWorkers = new COpenGLDisplay*[num_cameras]();
                    gpuVideoEncoders = new GPUVideoEncoder*[num_cameras]();
            
                    for(int i = 0; i < num_cameras; ++i) {
                        openGLDisplayWorkers[i] = nullptr;
                        gpuVideoEncoders[i] = nullptr;
                    }
            
                    size_t max_frame_size_bytes = 0;
                    for (int i = 0; i < num_cameras; ++i) {
                        size_t current_size = static_cast<size_t>(cameras_params[i].width) * static_cast<size_t>(cameras_params[i].height);
                        if (current_size > max_frame_size_bytes) {
                            max_frame_size_bytes = current_size;
                        }
                    }
                    std::cout << "Allocating worker pool with max frame size: " << max_frame_size_bytes << " bytes" << std::endl;
            
                    for (int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
                        ck(cudaMalloc(&worker_entry_pool[i].d_image, max_frame_size_bytes));
                        free_entries_queue->push(&worker_entry_pool[i]);
                    }
            
                    // 2. SETUP GPU TEXTURES FOR DISPLAY
                    cudaSetDevice(display_gpu_id);
                    tex = new GL_Texture[num_cameras];
                    for (int i = 0; i < num_cameras; i++) {
                        if (cameras_select[i].stream_on) {
                            int camera_width = int(cameras_params[i].width / cameras_select[i].downsample);
                            int camera_height = int(cameras_params[i].height / cameras_select[i].downsample);
                            setup_texture(tex[i], camera_width, camera_height);
                        }
                    }
            
                    // 3. CREATE WORKER THREAD OBJECTS (ENCODERS, YOLO, DISPLAY)
                    yolo_workers.assign(num_cameras, nullptr);
                    for (int i = 0; i < num_cameras; i++) {
                        if (cameras_select[i].stream_on) {
                            std::string display_name = "OpenGLDisplay_Cam_" + cameras_params[i].camera_serial;
                            openGLDisplayWorkers[i] = new COpenGLDisplay(display_name.c_str(), cuContext, &cameras_params[i], &cameras_select[i], tex[i].cuda_buffer, &indigo_signal_builder, *recycle_queue);
                        }
                        if (cameras_select[i].yolo) {
                            std::string yolo_name = "YOLO_Worker_Cam_" + cameras_params[i].camera_serial;
                            cameras_select[i].yolo_model = yolo_model.c_str();
                            if (yolo_model.empty()) {
                                std::cerr << "YOLO model not selected. Please select a YOLO model." << std::endl;
                            } else {
                                yolo_workers[i] = new YOLOv8Worker(yolo_name.c_str(), cuContext, &cameras_params[i], &cameras_select[i], *recycle_queue);
                                if (openGLDisplayWorkers[i]) {
                                    yolo_workers[i]->SetDisplayWorker(openGLDisplayWorkers[i]);
                                }
                            }
                        }
                        if (cameras_select[i].record) {
                            std::string encoder_thread_name = "GPUEncoder_Cam_" + cameras_params[i].camera_serial;
                            bool encoder_ready_signal = false;
                            gpuVideoEncoders[i] = new GPUVideoEncoder(encoder_thread_name.c_str(), cuContext, &cameras_params[i], encoder_config->encoder_codec, encoder_config->encoder_preset, encoder_config->tuning_info, encoder_config->folder_name, &encoder_ready_signal, *recycle_queue);
                        }
                    }
            
                    // 4. START ALL WORKER THREADS
                    for (int i = 0; i < num_cameras; i++) {
                        if (openGLDisplayWorkers[i]) openGLDisplayWorkers[i]->StartThread();
                        if (yolo_workers[i]) yolo_workers[i]->StartThread();
                        if (gpuVideoEncoders[i]) gpuVideoEncoders[i]->StartThread();
                    }
            
                    // 5. PREPARE CAMERAS AND START ACQUISITION THREADS
                    for (int i = 0; i < num_cameras; i++) {
                        camera_open_stream(&ecams[i].camera, &cameras_params[i]);
                        ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
                        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], evt_buffer_size);
                    }
            
                    if (ptp_stream_sync) {
                        for (int i = 0; i < num_cameras; i++) {
                            ptp_camera_sync(&ecams[i].camera, &cameras_params[i]);
                        }
                        camera_control->sync_camera = true;
                    }
            
                    for (int i = 0; i < num_cameras; i++) {
                        camera_threads.emplace_back(
                            &acquire_frames,
                            cuContext,
                            &ecams[i],
                            &cameras_params[i],
                            &cameras_select[i],
                            camera_control,
                            ptp_params,
                            &indigo_signal_builder,
                            openGLDisplayWorkers[i],
                            gpuVideoEncoders[i],
                            yolo_workers[i],
                            image_writer,
                            free_entries_queue,
                            recycle_queue
                        );
                    }
                }
                ImGui::PopStyleColor(1);
            }

            if (my_servers[0].server_state == FetchGame::ManagerState_WAITSTART && my_servers[1].server_state ==
                FetchGame::ManagerState_WAITSTART) {
                // check network servers are ready as well as local computer
                if (ptp_params->ptp_counter == num_cameras) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                    if (ImGui::Button("Start Recording")) {
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
                if (cameras_select[i].stream_on) {
                    int camera_width = int(cameras_params[i].width / cameras_select[i].downsample);
                    int camera_height = int(cameras_params[i].height / cameras_select[i].downsample);
                    clear_upload_and_cleanup(tex[i], camera_width, camera_height);
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
                camera_is_selected[i] = 0;
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
                    ImGui::Selectable(temp_string, &camera_is_selected[i], ImGuiSelectableFlags_SpanAllColumns);
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
                        camera_is_selected[i] = true;
                    }
                } else {
                    for (int i = 0; i < cam_count; i++) {
                        camera_is_selected[i] = false;
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
                static int codec_current = 0;
                if (ImGui::Combo("codec", &codec_current, items, IM_ARRAYSIZE(items))) {
                    encoder_config->encoder_codec = items[codec_current];
                }
            }
            {
                const char *items[] = {"p1", "p3", "p5", "p7"};
                static int preset_current = 0;
                if (ImGui::Combo("preset", &preset_current, items, IM_ARRAYSIZE(items))) {
                    encoder_config->encoder_preset = items[preset_current];
                }
            }
            { // Scoped for clarity
                const char *items[] = {"hq", "ll", "ull", "lossless"};
                static int tuning_current = 1; // Default to "ll" (low latency)
                if (ImGui::Combo("tuning", &tuning_current, items, IM_ARRAYSIZE(items))) {
                    encoder_config->tuning_info = items[tuning_current];
                }
                // Ensure default is set on first run
                if (encoder_config->tuning_info.empty()) {
                    encoder_config->tuning_info = "ll";
                }
            }
            {
                const char *items[] = {"1", "2", "4", "8", "16"};
                static const int item_numbers[] = {1, 2, 4, 8, 16};
                static int downsample_current = 0;
                if(ImGui::Combo("downsample streaming", &downsample_current, items, IM_ARRAYSIZE(items))) {
                    for (int i = 0; i < num_cameras; i++) {
                        cameras_select[i].downsample = item_numbers[downsample_current];
                    }
                }
            }

            int fps_temp = streaming_target_fps.load(); // get the current atomic value

            if (ImGui::InputInt("streaming fps", &fps_temp)) {
                // Clamp if necessary
                if (fps_temp < 1) fps_temp = 1;
                if (fps_temp > 240) fps_temp = 240;
                streaming_target_fps.store(fps_temp); // write it back safely
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

                if (ImGui::BeginTable("Camera Control Setting", 7,
                                      ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings |
                                      ImGuiTableFlags_Borders)) {
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("Name");
                    ImGui::TableNextColumn();
                    ImGui::Text("Serial");
                    ImGui::TableNextColumn();
                    ImGui::Text("Stream "); ImGui::SameLine();
                    if(ImGui::Checkbox("All##stream", &stream_all_cameras))
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
                    ImGui::Text("Record "); ImGui::SameLine();
                    if(ImGui::Checkbox("All##record", &record_all_cameras))
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
                    ImGui::Text("YOLO "); ImGui::SameLine();

                    // New Columns for IPC and ENet selection for YOLO
                    ImGui::TableNextColumn();
                    ImGui::Text("YOLO IPC"); // New header

                    ImGui::TableNextColumn();
                    ImGui::Text("YOLO ENet"); // New header


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

                        // New Checkboxes for IPC and ENet
                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##yolo_ipc%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].send_yolo_via_ipc); // Assumes flag name from video_capture.h

                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##yolo_enet%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].send_yolo_via_enet); // Assumes flag name from video_capture.h
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
                        const int cols = 5;
                        for (int i = 0; i < num_cameras; ++i) {     
                            std::string label = cameras_params[i].camera_name + ": " + std::to_string(cameras_select[i].pictures_counter) + "##calibration_save";
                            if (ImGui::Selectable(label.c_str(), &cameras_select[i].selected_to_save,
                                                    ImGuiSelectableFlags_None,
                                                    ImVec2(150, 50))) {
                            }
    
                            // Keep items on the same line until end of row
                            if ((i + 1) % cols != 0)
                                ImGui::SameLine();
                        }
    
                        ImGui::NewLine();
    
                        // ===================================================================
                        // START: ADD THIS NEW DIAGNOSTIC BLOCK
                        // ===================================================================
                        ImGui::SeparatorText("Debug Info");
                        ImGui::Text("Window Focused: %d, Window Hovered: %d", ImGui::IsWindowFocused(ImGuiFocusedFlags_AnyWindow), ImGui::IsWindowHovered(ImGuiHoveredFlags_AnyWindow));
                        ImGui::Text("camera_control->subscribe = %s", camera_control->subscribe ? "true" : "false");
                        ImGui::Text("save_image_all_ready = %s", save_image_all_ready ? "true" : "false");
                        ImGui::Separator();
                        // ===================================================================
                        // END: ADD THIS NEW DIAGNOSTIC BLOCK
                        // ===================================================================
    
                        if (ImGui::Button("Save selected")) {
                            std::cout << "[GUI] 'Save selected' button clicked. Formatting save name." << std::endl;
                            make_folder(picture_save_folder);
                            std::string frame_save_name = get_current_time_milliseconds();
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = frame_save_name;
                                cameras_select[i].picture_save_folder = picture_save_folder;
                                if (cameras_select[i].selected_to_save) {
                                    cameras_select[i].frame_save_state = State_Write_New_Frame;
                                    std::cout << "[GUI]   - Flagging camera " << cameras_params[i].camera_serial << " to save frame." << std::endl;
                                }
                            }
                        }
                        ImGui::SameLine();
    
                        if (ImGui::Button("Save pictures all")) {
                            std::cout << "[GUI] 'Save pictures all' button clicked. Formatting save name." << std::endl;
                            make_folder(picture_save_folder);
                            std::string frame_save_name = get_current_time_milliseconds();
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = frame_save_name;
                                cameras_select[i].picture_save_folder = picture_save_folder;
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
                            std::cout << "[GUI]   - Flagging ALL cameras to save frame." << std::endl;
                        }
    

                        if (calib_state == CalibSavePictures) {
                            std::cout << "[GUI] Calibration state is 'CalibSavePictures', triggering next pose." << std::endl;
                            send_indigo_message(indigo_signal_builder.server, indigo_signal_builder.builder, indigo_signal_builder.indigo_connection, FetchGame::SignalType_CalibrationNextPose);
                            calib_state = CalibNextPose;
                        }
                        
                        if (calib_state == CalibPoseReached) {
                            std::cout << "[GUI] Calibration state is 'CalibPoseReached', triggering frame save for all cameras." << std::endl;
                            make_folder(calib_save_folder);
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = std::to_string(cameras_select[i].pictures_counter);
                                cameras_select[i].picture_save_folder = calib_save_folder;
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
                            calib_state = CalibSavePictures;
                        }

                        if (ImGui::Button("Calib save images with counter")) {
                            std::cout << "[GUI] 'Calib save images with counter' button clicked." << std::endl;
                            make_folder(calib_save_folder);
                            for (int i = 0; i < num_cameras; i++) {
                                cameras_select[i].frame_save_name = std::to_string(cameras_select[i].pictures_counter);
                                cameras_select[i].picture_save_folder = calib_save_folder;
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
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
                        select_cameras_have_configs(camera_config_files, device_info, camera_is_selected, cam_count);
                    }

                    num_cameras = 0;
                    for (int i = 0; i < cam_count; i++) {
                        if (camera_is_selected[i]) {
                            num_cameras++;
                        }
                    }
                    if (num_cameras > 0) {
                        camera_control->open = true;
                        cameras_params = new CameraParams[num_cameras];
                        cameras_select = new CameraEachSelect[num_cameras];

                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++) {
                            if (camera_is_selected[i]) {
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
                        // --- START STREAMING ---
                        std::cout << "STARTING STREAMING SESSION..." << std::endl;
                    
                        // Create recording folder if needed
                        if (std::any_of(cameras_select, cameras_select + num_cameras, [](const CameraEachSelect& cs){ return cs.record; })) {
                            encoder_config->folder_name = input_folder + "/" + get_current_date_time();
                            make_folder(encoder_config->folder_name);
                            std::cout << "Recording session folder: " << encoder_config->folder_name << std::endl;
                        }
                    
                        // Allocate shared resources
                        worker_entry_pool = new WORKER_ENTRY[ACQUIRE_WORK_ENTRIES_MAX];
                        free_entries_queue = new SafeQueue<WORKER_ENTRY*>();
                        recycle_queue = new SafeQueue<WORKER_ENTRY*>();
                        openGLDisplayWorkers = new COpenGLDisplay*[num_cameras]();
                        gpuVideoEncoders = new GPUVideoEncoder*[num_cameras]();  // This should already exist
                        tex = new GL_Texture[num_cameras];
                        yolo_workers.assign(num_cameras, nullptr);
                    
                        // Initialize all pointers to nullptr
                        for(int i = 0; i < num_cameras; ++i) {
                            openGLDisplayWorkers[i] = nullptr;
                            gpuVideoEncoders[i] = nullptr;  // This should already exist
                        }
                    
                        // Allocate worker entry pool
                        size_t max_frame_size_bytes = 0;
                        for (int i = 0; i < num_cameras; ++i) {
                            size_t current_size = static_cast<size_t>(cameras_params[i].width) * static_cast<size_t>(cameras_params[i].height);
                            if (current_size > max_frame_size_bytes) {
                                max_frame_size_bytes = current_size;
                            }
                        }
                        std::cout << "Allocating " << ACQUIRE_WORK_ENTRIES_MAX << " worker entries with max frame size: " << max_frame_size_bytes << " bytes" << std::endl;
                    
                        for (int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
                            ck(cudaMalloc(&worker_entry_pool[i].d_image, max_frame_size_bytes));
                            free_entries_queue->push(&worker_entry_pool[i]);
                        }
                    
                        // Setup textures for display
                        for (int i = 0; i < num_cameras; i++) {
                            if (cameras_select[i].stream_on) {
                                int camera_width = int(cameras_params[i].width / cameras_select[i].downsample);
                                int camera_height = int(cameras_params[i].height / cameras_select[i].downsample);
                                setup_texture(tex[i], camera_width, camera_height);
                            }
                        }
                    
                        // *** ADD THIS SECTION: Create worker threads ***
                        for (int i = 0; i < num_cameras; i++) {
                            // Create OpenGL Display workers
                            if (cameras_select[i].stream_on) {
                                std::string display_name = "OpenGLDisplay_Cam_" + cameras_params[i].camera_serial;
                                openGLDisplayWorkers[i] = new COpenGLDisplay(display_name.c_str(), cuContext, &cameras_params[i], &cameras_select[i], tex[i].cuda_buffer, &indigo_signal_builder, *recycle_queue);
                            }
                    
                            // Create YOLO workers
                            if (cameras_select[i].yolo) {
                                std::string yolo_name = "YOLO_Worker_Cam_" + cameras_params[i].camera_serial;
                                cameras_select[i].yolo_model = yolo_model.c_str();
                                if (!yolo_model.empty()) {
                                    yolo_workers[i] = new YOLOv8Worker(yolo_name.c_str(), cuContext, &cameras_params[i], &cameras_select[i], *recycle_queue);
                                    if (openGLDisplayWorkers[i]) {
                                        yolo_workers[i]->SetDisplayWorker(openGLDisplayWorkers[i]);
                                    }
                                }
                            }
                    
                            // *** CREATE GPU VIDEO ENCODERS - THIS IS WHAT'S MISSING ***
                            if (cameras_select[i].record) {
                                std::string encoder_thread_name = "GPUEncoder_Cam_" + cameras_params[i].camera_serial;
                                bool encoder_ready_signal = false;
                                
                                std::cout << "Creating GPUVideoEncoder for camera " << cameras_params[i].camera_serial 
                                          << " with codec: " << encoder_config->encoder_codec 
                                          << ", preset: " << encoder_config->encoder_preset 
                                          << ", tuning: " << encoder_config->tuning_info << std::endl;
                                
                                gpuVideoEncoders[i] = new GPUVideoEncoder(
                                    encoder_thread_name.c_str(), 
                                    cuContext, 
                                    &cameras_params[i], 
                                    encoder_config->encoder_codec, 
                                    encoder_config->encoder_preset, 
                                    encoder_config->tuning_info,  // Make sure this field exists in EncoderConfig
                                    encoder_config->folder_name, 
                                    &encoder_ready_signal, 
                                    *recycle_queue
                                );
                                
                                std::cout << "GPUVideoEncoder created for " << cameras_params[i].camera_serial << std::endl;
                            }
                        }
                    
                        // Start all worker threads
                        for (int i = 0; i < num_cameras; i++) {
                            if (openGLDisplayWorkers[i]) openGLDisplayWorkers[i]->StartThread();
                            if (yolo_workers[i]) yolo_workers[i]->StartThread();
                            if (gpuVideoEncoders[i]) gpuVideoEncoders[i]->StartThread();  // THIS IS CRITICAL
                        }
                    
                        // Setup camera streaming
                        for (int i = 0; i < num_cameras; i++) {
                            camera_open_stream(&ecams[i].camera, &cameras_params[i]);
                            ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
                            allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], evt_buffer_size);
                        }
                    
                        // PTP sync if needed
                        if (ptp_stream_sync) {
                            for (int i = 0; i < num_cameras; i++) {
                                ptp_camera_sync(&ecams[i].camera, &cameras_params[i]);
                            }
                            camera_control->sync_camera = true;
                        }
                    
                        // Start acquisition threads
                        std::string encoder_setup = "-codec " + encoder_config->encoder_codec + " -preset " + encoder_config->encoder_preset + " -fps ";
                    
                        for (int i = 0; i < num_cameras; i++) {
                            camera_threads.emplace_back(
                                &acquire_frames,
                                cuContext,                          // CUcontext cuda_context
                                &ecams[i],                         // CameraEmergent *ecam
                                &cameras_params[i],                // CameraParams *camera_params
                                &cameras_select[i],                // CameraEachSelect* camera_select
                                camera_control,                    // CameraControl* camera_control
                                ptp_params,                        // PTPParams* ptp_params
                                &indigo_signal_builder,            // INDIGOSignalBuilder* indigo_signal_builder
                                openGLDisplayWorkers[i],           // COpenGLDisplay* openGLDisplay
                                gpuVideoEncoders[i],               // GPUVideoEncoder* gpu_encoder
                                yolo_workers[i],                   // YOLOv8Worker* yolo_worker
                                image_writer,                      // ImageWriterWorker* image_writer
                                free_entries_queue,                // SafeQueue<WORKER_ENTRY*>* free_entries_queue
                                recycle_queue                      // SafeQueue<WORKER_ENTRY*>* recycle_queue
                            );
                        }
                    } else {
                        // --- STOP STREAMING ---
                        std::cout << "STOPPING STREAMING SESSION..." << std::endl;
    
                        // --- REFACTOR 4: ORDERLY SHUTDOWN AND CLEANUP ---
                        // 1. Stop the acquisition threads from producing more work
                        for (auto &t : camera_threads) {
                            if (t.joinable()) t.join();
                        }
                        camera_threads.clear();
    
                        // 2. Stop all worker threads
                        for (int i = 0; i < num_cameras; i++) {
                            if (yolo_workers[i]) { yolo_workers[i]->StopThread(); }
                            if (openGLDisplayWorkers[i]) { openGLDisplayWorkers[i]->StopThread(); }
                            if (gpuVideoEncoders[i]) { gpuVideoEncoders[i]->StopThread(); }
                        }
                         for (int i = 0; i < num_cameras; i++) {
                            if (yolo_workers[i]) { delete yolo_workers[i]; }
                            if (openGLDisplayWorkers[i]) { delete openGLDisplayWorkers[i]; }
                            if (gpuVideoEncoders[i]) { delete gpuVideoEncoders[i]; }
                        }
                        yolo_workers.clear();
                        if(openGLDisplayWorkers) delete[] openGLDisplayWorkers;
                        openGLDisplayWorkers = nullptr;
                        if(gpuVideoEncoders) delete[] gpuVideoEncoders;
                        gpuVideoEncoders = nullptr;
    
                        // 3. Clean up camera SDK resources
                        for (int i = 0; i < num_cameras; i++) {
                            destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size, &cameras_params[i]);
                            delete[] ecams[i].evt_frame;
                            ecams[i].evt_frame = nullptr;
                            check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera), cameras_params[i].camera_serial.c_str());
                        }
    
                        // 4. Clean up OpenGL textures
                        for (int i = 0; i < num_cameras; i++) {
                             if (cameras_select[i].stream_on) {
                                 int w = int(cameras_params[i].width / cameras_select[i].downsample);
                                 int h = int(cameras_params[i].height / cameras_select[i].downsample);
                                 clear_upload_and_cleanup(tex[i], w, h);
                             }
                        }
                        if(tex) delete[] tex;
                        tex = nullptr;
    
                        // 5. Clean up the shared resource pool
                        if (worker_entry_pool) {
                            for (int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
                                if (worker_entry_pool[i].d_image) cudaFree(worker_entry_pool[i].d_image);
                            }
                            delete[] worker_entry_pool;
                            worker_entry_pool = nullptr;
                        }
                        if (free_entries_queue) { delete free_entries_queue; free_entries_queue = nullptr; }
                        if (recycle_queue) { delete recycle_queue; recycle_queue = nullptr; }
                    }
                }
            }

            if (camera_control->stop_record) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.5f, 0, 0, 1.0f});
            } else {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
            }

            if (camera_control->open) {

                if (!camera_control->subscribe) {
                    ImGui::BeginDisabled();
                }

                if (ImGui::Button(camera_control->record_video ? ICON_FK_PAUSE : ICON_FK_PLAY)) {
                    camera_control->record_video = !camera_control->record_video;
                
                    if (camera_control->record_video) {
                        // The folder is now created when streaming starts.
                        // We can still start the timer here if we want.
                        try_start_timer();
                        std::cout << "Recording toggled ON." << std::endl;
                    } else {
                        try_stop_timer();
                        std::cout << "Recording toggled OFF." << std::endl;
                    }
                }
                
                if (!camera_control->subscribe) {
                    ImGui::EndDisabled();
                }
            }

            ImGui::PopStyleColor(1);
        }
        ImGui::End();

        if (camera_control->subscribe) {
            for (int i = 0; i < num_cameras; i++) {
                if (cameras_select[i].stream_on) {
                    int camera_width = int(cameras_params[i].width / cameras_select[i].downsample);
                    int camera_height = int(cameras_params[i].height / cameras_select[i].downsample);
                    upload_texture_from_pbo(tex[i], camera_width, camera_height);
                }
            }

            if (camera_control->record_video) {
                int64_t start_ns = record_start_time_ns.load();
                std::string g_formatted_elapsed_time;
                if (start_ns > 0) {
                    int64_t now_ns = std::chrono::duration_cast<std::chrono::nanoseconds>(
                        std::chrono::steady_clock::now().time_since_epoch()).count();
                
                    auto elapsed_sec = std::chrono::seconds((now_ns - start_ns) / 1'000'000'000);
                    g_formatted_elapsed_time = format_elapsed_time(elapsed_sec);
                }

                for (int i = 0; i < num_cameras; i++) {
                    if (cameras_select[i].stream_on) {
                        std::string window_name = cameras_params[i].camera_name;
                        ImGui::Begin(window_name.c_str());
                        
                        if (start_ns > 0) {
                            ImGui::TextColored(ImVec4{0.0, 1.0f, 0, 1.0f}, "Elapsed Time: %s", g_formatted_elapsed_time.c_str());
                        } else {
                            ImGui::TextColored(ImVec4{1.0, 1.0f, 0, 1.0f}, "Recording starting...");
                        }
                        ImGui::SameLine();
                        ImGui::Text("FPS: %.1f", streaming_fps.load());                    
            
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
            } else {
                for (int i = 0; i < num_cameras; i++) {
                    if (cameras_select[i].stream_on) {
                        std::string window_name = cameras_params[i].camera_name;
                        ImGui::Begin(window_name.c_str());
                        ImGui::TextColored(ImVec4{1.0, 0.0f, 0, 1.0f}, "NOT RECORDING, ");
                        ImGui::SameLine();
                        ImGui::Text("FPS: %.1f", streaming_fps.load());    
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

    // Pop and destroy the primary context
    ck(cuCtxPopCurrent(&cuContext));
    ck(cuCtxDestroy(cuContext));

    image_writer->StopThread();
    delete image_writer;

    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    enet_release(&server);
    return 0;
}