#include "video_capture.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include "implot.h"
#include <imfilebrowser.h>
#include "project.h"
#include "gui.h"
#include <sys/stat.h>
#include "NvEncoder/NvCodecUtils.h"
#include "network_base.h"
#include "acquire_frames.h"
#include "enet_thread.h"

#include "lj_helper.h"  // labjack helper

simplelogger::Logger *logger = simplelogger::LoggerFactory::CreateConsoleLogger();


LabJackState lj_state;
Pulse pulse;
float duty_cycle = pulse.dutyCycle;
float fps = pulse.frequency;

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

    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory | ImGuiFileBrowserFlags_CreateNewDir);
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split(cwd, delimiter);

    std::string home_directory = "/home/" + tokenized_path[2];
    std::string input_folder = home_directory + "/exp/unsorted";
    file_dialog.SetPwd(input_folder);
    file_dialog.SetTitle("My files");

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
    bool hw_sync = true;
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
        if (ImGui::Begin("Labjack")) 
        {
            if(ImGui::Button(lj_state.is_connected? "disconnect T7": "connect to T7"))
            {
               
                if (lj_state.is_connected)  {
                    std::cout<<"disconnected from labjack" << std::endl;
                    close_labjack(&lj_state);
                }
                else                {
                    printf("connecting to labjack\n");
                    std::cout<<"connected to labjack"<<std::endl;
                    open_labjack(&lj_state);
                }

                
            }
            if(lj_state.is_connected) 
            {
                if(ImGui::Button(lj_state.pulse_on? "stop pulse": "start pulse"))
                {
                    if (lj_state.pulse_on) {
                        lj_state.pulse_on = false;
                        stop_pulsing(&lj_state);                    
                        

                    }
                    else {
                        lj_state.pulse_on = true;
                        start_pulsing(&lj_state, &pulse);                    
                    }
                }

                ImGui::SliderFloat("duty cycle", &duty_cycle, 10.0f, 20.0f, "%.3f");
                ImGui::SliderFloat("fps", &fps, 1.0f, 180.0f, "%.3f");
                
                update_pulse(&pulse, fps, duty_cycle);
            }
        }
        ImGui::End();




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
                file_dialog.Open();
            }
            ImGui::SameLine();
            ImGui::Text(input_folder.c_str());


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
                    // this the configs and updates the config values in this namespace(?)
                    update_camera_configs(camera_config_files, local_config_folders[local_config_select]);
                    select_cameras_have_configs(camera_config_files, device_info, check, cam_count);

                    num_cameras = 0;
                    for (int i = 0; i < cam_count; i++) {
                        if (check[i])   {
                            num_cameras++;
                        }
                    }
                    if (num_cameras > 0) {
                        cameras_params = new CameraParams[num_cameras];
                        cameras_select = new CameraEachSelect[num_cameras];

                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++) {
                            if (check[i]) {
                                selected_cameras.push_back(i);
                            }
                        }

                        for (int i = 0; i < num_cameras; i++)   {
                            set_camera_params(&cameras_params[i], &device_info[selected_cameras[i]], camera_config_files, selected_cameras[i], num_cameras);
                        }

                        for (int i =0; i < num_cameras; i++) {
                            cameras_select[i].stream_on = false;
                            if (cameras_params[i].camera_name.compare("ceiling_center") == 0) {
                                cameras_select[i].stream_on = true;
                                cameras_select[i].yolo = true;
                            }
                        }
                        
                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++)   {
                            
                            open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);

                        }
                        realtime_plot_data = new ScrollingBuffer[num_cameras];
                    }

                } 
                else {
                    for (int i = 0; i < num_cameras; i++){
                        close_camera(&ecams[i].camera);
                    }
                    delete[] cameras_params;
                    delete[] ecams;
                }
            }

            ImGui::Separator();
            ImGui::Spacing();

            ImGui::Checkbox("HW Stream Sync", &hw_sync); ImGui::SameLine();            
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

                    // image rendering?
                    tex = new GL_Texture[num_cameras];                    
                    for (int i = 0; i < num_cameras; i++)
                    {
                        if (cameras_select[i].stream_on) {
                            std::cout << "streaming on" << std::endl;
                            cudaStreamCreate(&tex[i].streams);
                            create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                            register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                            map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                            cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                            create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                        }
                    }

                    // configure hardware sync
                    for (int i = 0; i < num_cameras; i++)   {
                            set_hw_sync_evt_nic(&ecams[i].camera);
                    }
                    
                    // start acquiring frames
                    for (int i = 0; i < num_cameras; i++){
                        camera_threads.push_back(std::thread(&acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &indigo_signal_builder));
                    }

                } 
                else {
                    
                    // force cameras to stop acquiring frames (may not be the best way to do this as each camera may have different frame count)
                    for (int i = 0; i < num_cameras; i++)   {                    
                        check_camera_errors(EVT_CameraExecuteCommand(&ecams[i].camera, "AcquisitionStop"));
                    }

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

                    // configure hardware sync
                    for (int i = 0; i < num_cameras; i++)   {
                            set_hw_sync_evt_nic(&ecams[i].camera);
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        encoder_setup = encoder_basic_setup + std::to_string(cameras_params[i].frame_rate);
                        camera_threads.push_back(std::thread(&acquire_frames, &ecams[i], &cameras_params[i], &cameras_select[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &indigo_signal_builder));
                    }
                    camera_control->subscribe = true;                    
                } 
                else {
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

        file_dialog.Display();
        if (file_dialog.HasSelected())
        {
            input_folder = file_dialog.GetSelected().string();
            file_dialog.ClearSelected();
        }

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
