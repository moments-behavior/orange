#include "video_capture.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <ImGuiFileDialog.h>
#include "project.h"
#include "gui.h"
#include "realtime_tool.h"
#include <sys/stat.h>
#include "NvEncoder/NvCodecUtils.h"
#include "network_base.h"
#include "acquire_frames.h"
#include "enet_thread.h"

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();

int main(int argc, char **args)
{
    gx_context *window = (gx_context *)malloc(sizeof(gx_context));
    *window = (gx_context){
        .swap_interval = 1, // use vsync
        .width = 1920,
        .height = 1080,
        .render_target_title = (char *)malloc(100), // window title
        .glsl_version = (char *)malloc(100)};

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

    std::string home_directory = "/home/" + tokenized_path[2];
    std::string input_folder = home_directory + "/exp/unsorted";

    DetectionData* detection_data = new DetectionData;
    detection_data->calibration_folder = home_directory + "/Calibration/5cam";
    detection_data->yolo_model_folder = home_directory + "/detect";
    detection_data->yolo_model = detection_data->yolo_model_folder + "/ball.engine";

    bool check[cam_count] {0};
    CameraParams *cameras_params;
    CameraEachSelect *cameras_select;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    GL_Texture *tex;
    u32 num_cameras = 0;

    CameraControl *camera_control = new CameraControl;

    int evt_buffer_size {100};
    PTPParams* ptp_params = new PTPParams{0, 0, 0, 0, false, false, false, false};
    std::string encoder_setup;
    std::string encoder_basic_setup = "-codec h264 -preset p1 -fps ";
    std::string encoder_codec = "h264"; 
    std::string encoder_preset = "p1";
    std::string folder_name;

    std::vector<std::string> camera_config_files;

    ScrollingBuffer* realtime_plot_data;
    bool show_realtime_plot = false;
    bool ptp_stream_sync = false;
    
    flatbuffers::FlatBufferBuilder* fb_builder = new flatbuffers::FlatBufferBuilder(1024);

    EnetContext server;
    if (enet_initialize(&server, 3333, 5)) {
        printf("Server Initiated\n");
    }
    ConnectedServer my_servers[2] ;
    intialize_servers(my_servers);

    INDIGOSignalBuilder indigo_signal_builder;
    indigo_signal_builder = {.builder = fb_builder, 
        .server = &server,
        .indigo_connection = NULL};

    std::vector<std::string> network_config_folders;
    std::string network_start_folder_name = home_directory + "/config/network";
    for (const auto & entry : std::filesystem::directory_iterator(network_start_folder_name)) {
        network_config_folders.push_back(entry.path().string());
    }
    int network_config_select = 0;

    std::vector<std::string> local_config_folders;
    std::string local_start_folder_name = home_directory + "/config/local";
    for (const auto & entry : std::filesystem::directory_iterator(local_start_folder_name)) {
        local_config_folders.push_back(entry.path().string());
    }
    int local_config_select = 0;
    bool select_all_cameras = false;
    char* subfix_buf = (char*)malloc(64);
    *subfix_buf = '\0';
    char* temp_string = (char*)malloc(64);
    *temp_string = '\0';
    bool save_image_all_ready = true;
    bool quite_enet = false;
    
    std::thread enet_thread = std::thread(&create_enet_thread, &server, my_servers, &indigo_signal_builder, &quite_enet);

    while (!glfwWindowShouldClose(window->render_target))
    {
        create_new_frame();
        if (ImGui::Begin("Network")) 
        {
            
            if (ImGui::BeginTable("##Local Apps", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
            {
                ImGui::TableNextRow();
                ImGui::TableNextColumn();
                ImGui::Text("App");
                ImGui::TableNextColumn();
                ImGui::Text("Indigo");
                ImGui::TableNextColumn();
                sprintf(temp_string, "Not connected");
                if (indigo_signal_builder.indigo_connection != nullptr) {
                    if (indigo_signal_builder.indigo_connection->state == ENET_PEER_STATE_CONNECTED) {
                        sprintf(temp_string, "Connected");
                    } 
                }
                ImGui::Text(temp_string);
                ImGui::EndTable();
            }


            if (ImGui::BeginTable("Servers", 4, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
            {
                for (int i = 0; i < 2; i++)
                {
                    sprintf(temp_string, "##servers%d", i);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text(my_servers[i].name);
                    ImGui::TableNextColumn();

                    if (my_servers[i].peer != nullptr) {
                        if (my_servers[i].peer->state == ENET_PEER_STATE_CONNECTED) {
                            my_servers[i].connected = true;
                        }
                    } else {
                        my_servers[i].connected = false;
                    }
                    
                    if (ImGui::Button(my_servers[i].connected ? "Disconnect":"Connect"))
                    {   
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
                    ImGui::Text(std::to_string(my_servers[i].num_cameras).c_str());
                    ImGui::TableNextColumn();

                    if (my_servers[i].connected) {
                        ImGui::Text(FetchGame::EnumNamesManagerState()[my_servers[i].server_state]);
                    } else {
                        ImGui::Text("Not connected");
                    }
                }
                ImGui::EndTable();
            }
            
            for (int i = 0; i < network_config_folders.size(); i++)
            {
                std::vector<std::string> folder_token = string_split(network_config_folders[i].c_str(), "/");
                sprintf(temp_string, folder_token.back().c_str());
                ImGui::RadioButton(temp_string, &network_config_select, i); 
                if (i != network_config_folders.size()-1)
                    ImGui::SameLine();
            }
            
            if (my_servers[0].server_state == FetchGame::ManagerState_IDLE && my_servers[1].server_state == FetchGame::ManagerState_IDLE && my_servers[0].connected && my_servers[1].connected) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if(ImGui::Button("Open Cameras")) {
                    update_camera_configs(camera_config_files, network_config_folders[network_config_select]);
                    select_cameras_have_configs(camera_config_files, device_info, check, cam_count);
                    host_broadcast_open_cameras(fb_builder, &server, network_config_folders[network_config_select]);
                    // open cameras
                    num_cameras = 0;
                    for (int i = 0; i < cam_count; i++)
                    {
                        if (check[i])
                        {
                            num_cameras++;
                        }
                    }
                    if (num_cameras > 0) {
                        cameras_params = new CameraParams[num_cameras];
                        cameras_select = new CameraEachSelect[num_cameras];
                        
                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++)
                        {
                            if (check[i]) {
                                selected_cameras.push_back(i);
                            }
                        }
                        for (int i = 0; i < num_cameras; i++)
                        {
                            set_camera_params(&cameras_params[i], i, &device_info[selected_cameras[i]], camera_config_files, selected_cameras[i], num_cameras);
                        }

                        for (int i =0; i < num_cameras; i++) {
                            cameras_select[i].stream_on = false;
                            if (cameras_params[i].camera_name.compare("Cam16") == 0) {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].yolo = true;
                            }
                        }

                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++)
                        {
                            open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
                        }

                        realtime_plot_data = new ScrollingBuffer[num_cameras];
                    }
                    camera_control->open = true;
                }
                ImGui::PopStyleColor(1);
            }

            if (my_servers[0].server_state == FetchGame::ManagerState_WAITTHREAD && my_servers[1].server_state == FetchGame::ManagerState_WAITTHREAD) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if(ImGui::Button("Clients start camera threads")) {
                    encoder_basic_setup = "-codec " + encoder_codec + " -preset " + encoder_preset + " -fps ";
                    make_folder_for_recording(folder_name, input_folder, subfix_buf);
                    ptp_params->network_sync = true;
                    host_broadcast_start_threads(fb_builder, &server, folder_name, encoder_basic_setup);

                    // start local recording threads
                    allocate_camera_frame_buffers(ecams, cameras_params, evt_buffer_size, num_cameras);
        
                    tex = new GL_Texture[num_cameras];
                    for (int i = 0; i < num_cameras; i++)
                    {
                        if (cameras_select[i].stream_on) {
                            cudaStreamCreate(&tex[i].streams);
                            create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                            register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                            map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                            cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                            create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                        }
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        ptp_camera_sync(&ecams[i].camera);
                    }
                    camera_control->sync_camera = true;
                    camera_control->record_video = true;

                    // detection data allocate
                    detection_data->detect_per_cam = new DetectionDataPerCam[num_cameras];
                    for (int i = 0; i < num_cameras; i++) {
                        detection_data->detect_per_cam[i].yolo_model = detection_data->yolo_model;
                        detection_data->detect_per_cam[i].calibration_file = detection_data->calibration_folder + "/Cam" + cameras_params[i].camera_serial + ".yaml";
                        detection_data->detect_per_cam[i].have_calibration_results = load_camera_calibration_results(detection_data->detect_per_cam[i].calibration_file, &detection_data->detect_per_cam[i].camera_calib);
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &indigo_signal_builder, detection_data));
                    }
                    camera_control->subscribe = true;
                }
                ImGui::PopStyleColor(1);
            }

            if (my_servers[0].server_state == FetchGame::ManagerState_WAITSTART && my_servers[1].server_state == FetchGame::ManagerState_WAITSTART) {
                // check network servers are ready as well as local computer
                if (ptp_params->ptp_counter==num_cameras) {
                    ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                    if (ImGui::Button("Start Recording")) {
                        // get the host ready, and then set global ptp time to start recording  
                        unsigned long long ptp_time = get_current_PTP_time(&ecams[0].camera);
                        int delay_in_second = 3;
                        ptp_params->ptp_global_time = ((unsigned long long)delay_in_second) * 1000000000 + ptp_time;
                        host_broadcast_set_start_ptp(fb_builder, &server, ptp_params->ptp_global_time);
                        ptp_params->network_set_start_ptp = true;
                    }
                    ImGui::PopStyleColor(1);
                }
            }

            if (ptp_params->ptp_start_reached && my_servers[0].server_state == FetchGame::ManagerState_WAITSTOP && my_servers[1].server_state == FetchGame::ManagerState_WAITSTOP) {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
                if (ImGui::Button("Stop Recording")) {
                    unsigned long long ptp_time = get_current_PTP_time(&ecams[0].camera);
                    int delay_in_second = 3;
                    ptp_params->ptp_stop_time = ((unsigned long long)delay_in_second) * 1000000000 + ptp_time;
                    std::cout << ptp_params->ptp_stop_time << std::endl;
                    fb_builder->Clear();
                    FetchGame::ServerBuilder server_builder(*fb_builder);
                    server_builder.add_control(FetchGame::ServerControl_STOPRECORDING);
                    server_builder.add_ptp_global_time(ptp_params->ptp_stop_time);
                    auto my_server = server_builder.Finish();
                    fb_builder->Finish(my_server);
                    uint8_t *server_buffer = fb_builder->GetBufferPointer();
                    int server_buf_size = fb_builder->GetSize();
                    ENetPacket* enet_packet = enet_packet_create(server_buffer, server_buf_size, 0);
                    enet_host_broadcast(server.m_pNetwork, 0, enet_packet);
                    ptp_params->network_set_stop_ptp = true;
                }
                ImGui::PopStyleColor(1);
            }

            if (my_servers[0].server_state == FetchGame::ManagerState_IDLE && my_servers[1].server_state == FetchGame::ManagerState_IDLE) {
                if(ImGui::Button("Clients close")) {
                    // broadcast data
                    fb_builder->Clear();
                    FetchGame::ServerBuilder server_builder(*fb_builder);
                    server_builder.add_control(FetchGame::ServerControl_QUIT);
                    auto my_server = server_builder.Finish();
                    fb_builder->Finish(my_server);
                    uint8_t *server_buffer = fb_builder->GetBufferPointer();
                    int server_buf_size = fb_builder->GetSize();
                    ENetPacket* enet_packet = enet_packet_create(server_buffer, server_buf_size, 0);
                    enet_host_broadcast(server.m_pNetwork, 0, enet_packet);
                }
            }    
        }
        ImGui::End();

        if (ptp_params->network_set_stop_ptp && ptp_params->ptp_stop_reached) 
        {
            
            ptp_params->network_set_stop_ptp = false;

            for (auto &t : camera_threads)
                t.join();
            
            for (int i = 0; i < num_cameras; i++)
            {
                camera_threads.pop_back();
            }
            for (int i = 0; i < num_cameras; i++)
            {
                destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size);
                delete[] ecams[i].evt_frame;
                check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera));
                if (cameras_select[i].stream_on) {
                    gx_delete_buffer(&tex[i].pbo);
                    unmap_cuda_resource(&tex[i].cuda_resource);
                    cuda_unregister_pbo(tex[i].cuda_resource);
                }
            }
            
            delete[] tex;                     

            for (int i = 0; i < num_cameras; i++)
            {
                ptp_sync_off(&ecams[i].camera);
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

            for (int i = 0; i < num_cameras; i++)
            {
                close_camera(&ecams[i].camera);
            }

            camera_control->open = false;

            for (int i=0; i<cam_count; i++) {
                    check[i] = 0;
            }
        }

        if (ImGui::Begin("Orange", NULL, ImGuiWindowFlags_MenuBar))
        {

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Checkbox("Show realtime plot", &show_realtime_plot);

            if (ImGui::BeginTable("Cameras", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
            {

                for (int i = 0; i < cam_count; i++)
                {
                    sprintf(temp_string, "%d", i);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Selectable(temp_string, &check[i], ImGuiSelectableFlags_SpanAllColumns);
                    ImGui::TableNextColumn();
                    ImGui::Text(device_info[i].serialNumber);
                    ImGui::TableNextColumn();
                    ImGui::Text(device_info[i].currentIp);
                }
                ImGui::EndTable();
            }

            if (ImGui::Button(select_all_cameras ? "Clear all": "Select all")) {
                select_all_cameras  = !select_all_cameras;
                if (select_all_cameras) {
                    for (int i = 0; i < cam_count; i++)
                    {
                        check[i] = true;
                    }
                } else {
                    for (int i = 0; i < cam_count; i++)
                    {
                        check[i] = false;
                    }
                }
                
            }

            ImGui::InputText("subfix", subfix_buf, 64);

            if (ImGui::Button("Save to"))
            {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = input_folder;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseDirDlgKey", "Choose a Directory", nullptr, config);
            }

            ImGui::SameLine();
            ImGui::TextColored(ImVec4{1.0, 1.0f, 0, 1.0f}, input_folder.c_str());

            {
                const char* items[] = { "h264", "hevc"};
                static int item_current = 0;
                ImGui::Combo("codec", &item_current, items, IM_ARRAYSIZE(items));
                encoder_codec = items[item_current];
            }

            {
                const char* items[] = { "p1", "p3", "p5", "p7"};
                static int item_current = 0;
                ImGui::Combo("preset", &item_current, items, IM_ARRAYSIZE(items));
                encoder_preset = items[item_current];
            }

            // selection for yolo model
            if (ImGui::Button("Select YOLO"))
            {
                IGFD::FileDialogConfig config;
                config.countSelectionMax = 1;
                config.path = detection_data->yolo_model_folder;
                ImGuiFileDialog::Instance()->OpenDialog("ChooseFileDlgKey", "Choose File", ".engine", config);
            }
            ImGui::SameLine();
            ImGui::Text(detection_data->yolo_model.c_str());
            ImGui::Text(detection_data->calibration_folder.c_str());

            if (camera_control->open) {
                set_camera_properties(ecams, cameras_params, num_cameras);
                
                if (ImGui::BeginTable("Camera Control Setting", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
                {
                    
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Text("name"); 
                    ImGui::TableNextColumn();
                    ImGui::Text("stream");
                    ImGui::TableNextColumn();
                    ImGui::Text("yolo"); 

                    for (int i = 0; i < num_cameras; i++)
                    {
                        ImGui::TableNextRow();
                        ImGui::TableNextColumn();
                        ImGui::Text(cameras_params[i].camera_name.c_str());
                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##checkbox_control%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].stream_on);
                        ImGui::TableNextColumn();
                        sprintf(temp_string, "##checkbox_yolo%d", i);
                        ImGui::Checkbox(temp_string, &cameras_select[i].yolo);
                    }
                    ImGui::EndTable();
                }

            }
        }
        ImGui::End();

        // file explorer display
        if (ImGuiFileDialog::Instance()->Display("ChooseFileDlgKey"))
        { // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk())
            { // action if OK
                detection_data->yolo_model = ImGuiFileDialog::Instance()->GetFilePathName();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGuiFileDialog::Instance()->Display("ChooseDirDlgKey"))
        { // => will show a dialog
            if (ImGuiFileDialog::Instance()->IsOk())
            { // action if OK
                auto selected_folder = ImGuiFileDialog::Instance()->GetSelection();
                input_folder = ImGuiFileDialog::Instance()->GetCurrentPath();
            }
            // close
            ImGuiFileDialog::Instance()->Close();
        }

        if (ImGui::Begin("Local"))
        {

            for (int i = 0; i < local_config_folders.size(); i++)
            {
                std::vector<std::string> folder_token = string_split(local_config_folders[i].c_str(), "/");
                sprintf(temp_string, folder_token.back().c_str());
                ImGui::RadioButton(temp_string, &local_config_select, i);
                if (i != local_config_folders.size()-1)
                    ImGui::SameLine();
            }
        
            if (ImGui::Button(camera_control->open ? "Close Camera" : "Open camera")) {
                (camera_control->open) = !(camera_control->open);
                if (camera_control->open) 
                {
                    update_camera_configs(camera_config_files, local_config_folders[local_config_select]);
                    select_cameras_have_configs(camera_config_files, device_info, check, cam_count);

                    num_cameras = 0;
                    for (int i = 0; i < cam_count; i++)
                    {
                        if (check[i])
                        {
                            num_cameras++;
                        }
                    }
                    if (num_cameras > 0) {
                        cameras_params = new CameraParams[num_cameras];
                        cameras_select = new CameraEachSelect[num_cameras];

                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++)
                        {
                            if (check[i]) {
                                selected_cameras.push_back(i);
                            }
                        }
                        for (int i = 0; i < num_cameras; i++)
                        {
                            set_camera_params(&cameras_params[i], i, &device_info[selected_cameras[i]], camera_config_files, selected_cameras[i], num_cameras);
                        }

                        for (int i =0; i < num_cameras; i++) {
                            cameras_select[i].stream_on = false;
                            if (cameras_params[i].camera_name.compare("ceiling_center") == 0) {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].yolo = true;
                            }
                        }
                        
                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++)
                        {
                            open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
                        }
                        realtime_plot_data = new ScrollingBuffer[num_cameras];
                    }

                } else {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        close_camera(&ecams[i].camera);
                    }
                    delete[] cameras_params;
                    delete[] ecams;
                }
            }

            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Checkbox("PTP Stream Sync", &ptp_stream_sync); ImGui::SameLine();
            if (ImGui::Button(camera_control->subscribe ? "Stop streaming" : "Start streaming"))
            {
                (camera_control->subscribe) = !(camera_control->subscribe);
                if (camera_control->subscribe)
                {   
                    for (int i = 0; i < num_cameras; i++)
                    {               
                        camera_open_stream(&ecams[i].camera);
                        ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
                        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], evt_buffer_size);

                        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
                        {
                            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
                        }
                    }

                    tex = new GL_Texture[num_cameras];
                    for (int i = 0; i < num_cameras; i++)
                    {
                        if (cameras_select[i].stream_on) {
                            cudaStreamCreate(&tex[i].streams);
                            create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                            register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                            map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                            cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                            create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                        }
                    }

                    if (ptp_stream_sync){
                        for (int i = 0; i < num_cameras; i++)
                        {
                            ptp_camera_sync(&ecams[i].camera);
                        }
                        camera_control->sync_camera = true;
                    }

                    // detection data allocate
                    detection_data->detect_per_cam = new DetectionDataPerCam[num_cameras];
                    for (int i = 0; i < num_cameras; i++) {
                        detection_data->detect_per_cam[i].yolo_model = detection_data->yolo_model;
                        detection_data->detect_per_cam[i].calibration_file = detection_data->calibration_folder + "/Cam" + cameras_params[i].camera_serial + ".yaml";
                        detection_data->detect_per_cam[i].have_calibration_results = load_camera_calibration_results(detection_data->detect_per_cam[i].calibration_file, &detection_data->detect_per_cam[i].camera_calib);
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &indigo_signal_builder, detection_data));
                    }

                } else {
                    for (auto &t : camera_threads)
                        t.join();
                    
                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.pop_back();
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size);
                        delete[] ecams[i].evt_frame;
                        check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera));
                        
                        if (cameras_select[i].stream_on) {
                            gx_delete_buffer(&tex[i].pbo);
                            unmap_cuda_resource(&tex[i].cuda_resource);
                            cuda_unregister_pbo(tex[i].cuda_resource);
                        }
                    }
                    delete[] tex;
                    
                    if (num_cameras > 1) {
                        for (int i = 0; i < num_cameras; i++)
                        {
                            ptp_sync_off(&ecams[i].camera);
                        }
                        ptp_params->ptp_counter = 0;
                        ptp_params->ptp_global_time = 0;
                        camera_control->sync_camera = false;
                    }
                }
            }

            save_image_all_ready = true;
            if (camera_control->subscribe == true)
            {
                for (int i=0; i < num_cameras; i++)
                {
                    if (cameras_select[i].frame_save_state != State_Frame_Idle) {
                        save_image_all_ready = false;
                        break;
                    }
                }

                ImGui::Text("Save image index:");
                for (int i = 0; i < num_cameras; i++)
                {
                    sprintf(temp_string, "save_image_index%d", i);
                    ImGui::InputInt(temp_string, &cameras_select[i].frame_save_idx);
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    std::string label_save_input_checkbox;
                    label_save_input_checkbox = "s_" + cameras_params[i].camera_name;
                    ImGui::Checkbox(label_save_input_checkbox.c_str(), &cameras_select[i].selected_to_save);
                    ImGui::SameLine();
                }
                
                if (save_image_all_ready) {
                    if (ImGui::Button("Save selected"))
                    {
                        for (int i = 0; i < num_cameras; i++)
                        {
                            if (cameras_select[i].selected_to_save)
                            {
                                cameras_select[i].frame_save_state = State_Write_New_Frame;
                            }
                        }
                    }
                    
                    if (ImGui::Button("Save images all"))
                    {
                        for (int i = 0; i < num_cameras; i++)
                        {
                            cameras_select[i].frame_save_state = State_Write_New_Frame;
                        }
                    }
                }
            }
            ImGui::Separator();
            ImGui::Spacing();


            if (camera_control->stop_record)
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.5f, 0, 0, 1.0f});
            }
            else
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
            }

            if (ImGui::Button(camera_control->stop_record ? ICON_FK_PAUSE : ICON_FK_PLAY))
            {
                (camera_control->stop_record) = !(camera_control->stop_record);
                if (camera_control->stop_record)
                {
                    encoder_basic_setup = "-codec " + encoder_codec + " -preset " + encoder_preset + " -fps ";
                    camera_control->record_video = true;
                    make_folder_for_recording(folder_name, input_folder, subfix_buf);

                    for (int i = 0; i < num_cameras; i++)
                    {               
                        camera_open_stream(&ecams[i].camera);
                        ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
                        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], evt_buffer_size);
                        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
                        {
                            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
                        }
                    }
                    
                    tex = new GL_Texture[num_cameras];
                    for (int i = 0; i < num_cameras; i++)
                    {
                        if (cameras_select[i].stream_on) {
                            cudaStreamCreate(&tex[i].streams);
                            create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                            register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                            map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                            cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                            create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                        }
                    }

                    if (num_cameras > 1){
                        for (int i = 0; i < num_cameras; i++)
                        {
                            ptp_camera_sync(&ecams[i].camera);
                        }
                        camera_control->sync_camera = true;
                    }

                    // detection data allocate
                    detection_data->detect_per_cam = new DetectionDataPerCam[num_cameras];
                    for (int i = 0; i < num_cameras; i++) {
                        detection_data->detect_per_cam[i].yolo_model = detection_data->yolo_model;
                        detection_data->detect_per_cam[i].calibration_file = detection_data->calibration_folder + "/Cam" + cameras_params[i].camera_serial + ".yaml";
                        detection_data->detect_per_cam[i].have_calibration_results = load_camera_calibration_results(detection_data->detect_per_cam[i].calibration_file, &detection_data->detect_per_cam[i].camera_calib);
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        encoder_setup = encoder_basic_setup + std::to_string(cameras_params[i].frame_rate);
                        camera_threads.push_back(std::thread(&acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &indigo_signal_builder, detection_data));
                    }
                    
                    camera_control->subscribe = true;                    
                } else {
                    camera_control->subscribe = false;

                    for (auto &t : camera_threads)
                        t.join();
                    
                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.pop_back();
                    }
                    for (int i = 0; i < num_cameras; i++)
                    {
                        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, evt_buffer_size);
                        delete[] ecams[i].evt_frame;
                        check_camera_errors(EVT_CameraCloseStream(&ecams[i].camera));
                        if (cameras_select[i].stream_on) {
                            gx_delete_buffer(&tex[i].pbo);
                            unmap_cuda_resource(&tex[i].cuda_resource);
                            cuda_unregister_pbo(tex[i].cuda_resource);
                        }
                    }
                    
                    delete[] tex;                     

                    if (num_cameras > 1) {
                        for (int i = 0; i < num_cameras; i++)
                        {
                            ptp_sync_off(&ecams[i].camera);
                        }
                        ptp_params->ptp_counter = 0;
                        ptp_params->ptp_global_time = 0;
                        camera_control->sync_camera = false;
                    }
                    camera_control->record_video = false;
                }
            }
            ImGui::PopStyleColor(1);
        }
        ImGui::End();

        if (camera_control->subscribe)
        {
            for (int i = 0; i < num_cameras; i++)
            {
                if (cameras_select[i].stream_on) {
                    bind_pbo(&tex[i].pbo);
                    bind_texture(&tex[i].texture);
                    upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height); // Needs no arguments because texture and PBO are bound
                    unbind_pbo();
                    unbind_texture();
                }
            }
            
            for (int i = 0; i < num_cameras; i++)
            {
                if (cameras_select[i].stream_on) {
                    std::string window_name = cameras_params[i].camera_name;
                    ImGui::Begin(window_name.c_str());
                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    static ImVec2 bmin(0,0);
                    static ImVec2 uv0(0,0);
                    static ImVec2 uv1(1,1);
                    static ImVec4 tint(1,1,1,1);

                    // ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                    if (ImPlot::BeginPlot("##no_plot_name", avail_size, ImPlotFlags_Equal | ImPlotAxisFlags_AutoFit))
                    {
                        ImPlot::SetupAxesLimits(0, cameras_params[i].width, 0, cameras_params[i].height);
                        ImPlot::PlotImage("##no_image_name", (void *)(intptr_t)tex[i].texture, ImVec2(0, 0), ImVec2(cameras_params[i].width, cameras_params[i].height));                    
                        
                        if (detection_data->detect_per_cam[i].have_calibration_results) {
                            gui_plot_world_coordinates(&detection_data->detect_per_cam[i].camera_calib, &cameras_params[i]);
                        }
                        
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
                ImGui::SliderFloat("History",&history,1,30,"%.1f s");

                static ImPlotAxisFlags flags = ImPlotAxisFlags_NoTickMarks;
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                if (ImPlot::BeginPlot("Camera Sensor Temperature", avail_size)) {
                    ImPlot::SetupAxes(nullptr, nullptr, flags, flags);
                    ImPlot::SetupAxisLimits(ImAxis_X1,t - history, t, ImGuiCond_Always);
                    ImPlot::SetupAxisLimits(ImAxis_Y1,30,90);
                    ImPlot::SetNextFillStyle(IMPLOT_AUTO_COL,0.5f);

                    for (int i = 0; i < num_cameras; i++) {
                        std::string line_name = "Cam" + std::to_string(cameras_params[i].camera_id);
                        ImPlot::PlotLine(line_name.c_str(), &realtime_plot_data[i].Data[0].x, &realtime_plot_data[i].Data[0].y, realtime_plot_data[i].Data.size(), 0, realtime_plot_data[i].Offset, 2*sizeof(float));
                    }
                    ImPlot::EndPlot();
                }
                ImGui::End();
            }
        }

        render_a_frame(window);
    }
    
    quite_enet = true;
    enet_thread.join();
    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    enet_release(&server);
    return 0;
}
