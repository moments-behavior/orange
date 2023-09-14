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
#include "utils.h"
#include <sys/stat.h>
#include "NvEncoder/NvCodecUtils.h"

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

    int max_cameras = 16;
    int cam_count;
    GigEVisionDeviceInfo unsorted_device_info[max_cameras];
    cam_count = scan_cameras(max_cameras, unsorted_device_info);
    GigEVisionDeviceInfo device_info[max_cameras];
    sort_cameras_ip(unsorted_device_info, device_info, cam_count);

    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory | ImGuiFileBrowserFlags_CreateNewDir);
    std::filesystem::path cwd = std::filesystem::current_path();
    std::string delimiter = "/";
    std::vector<std::string> tokenized_path = string_split(cwd, delimiter);
    std::string start_folder_name = "/home/" + tokenized_path[2] + "/exp";
    file_dialog.SetPwd(start_folder_name);
    file_dialog.SetTitle("Select working directory");

    file_dialog.SetPwd(start_folder_name);
    std::string input_folder = file_dialog.GetPwd();
    file_dialog.SetTitle("My files");

    bool check[cam_count] = {};

    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<std::thread> camera_threads;
    GL_Texture *tex;
    u32 num_cameras = 0;

    CameraControl *camera_control = new CameraControl;

    int evt_buffer_size {150};
    PTPParams* ptp_params = new PTPParams{0, 0};
    std::string encoder_setup;
    std::string encoder_basic_setup = "-codec h264 -preset p1 -fps ";
    std::string encoder_codec = "h264"; 
    std::string encoder_preset = "p1";
    std::string folder_name;

    bool load_camera_config = false;
    std::vector<std::string> camera_config_files;
    std::vector<std::string> camera_config_names;

    ScrollingBuffer* realtime_plot_data;
    bool show_realtime_plot = false;

    while (!glfwWindowShouldClose(window->render_target))
    {
        create_new_frame();


        if (ImGui::Begin("Orange Streaming", NULL, ImGuiWindowFlags_MenuBar))
        {

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::Checkbox("Show realtime plot", &show_realtime_plot);

            if (ImGui::BeginTable("Cameras", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
            {
                for (int i = 0; i < cam_count; i++)
                {
                    char label[32];
                    sprintf(label, "##checkbox%d", i);
                    ImGui::TableNextRow();
                    ImGui::TableNextColumn();
                    ImGui::Checkbox(label, &check[i]);
                    ImGui::TableNextColumn();
                    ImGui::Text(device_info[i].serialNumber);
                    ImGui::TableNextColumn();
                    ImGui::Text(device_info[i].currentIp);
                }
                ImGui::EndTable();
            }

            if (ImGui::Button("Select all")) {
                for (int i = 0; i < cam_count; i++)
                {
                    check[i] = true;
                }
            }

            if (ImGui::Button("Load camera config")) {
                load_camera_config = true;          
                file_dialog.Open();  
            }

        
            if (ImGui::Button(camera_control->open ? "Close Camera" : "Open camera")) {
                (camera_control->open) = !(camera_control->open);
                if (camera_control->open) 
                {
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
                        std::vector<int> selected_cameras;
                        for (int i = 0; i < cam_count; i++)
                        {
                            if (check[i]) {
                                selected_cameras.push_back(i);
                            }
                        }

                        for (int i = 0; i < num_cameras; i++)
                        {
                            // first checkt to see if it is in the config files 
                            cameras_params[i].camera_serial.append(device_info[selected_cameras[i]].serialNumber);
                            cameras_params[i].camera_name = cameras_params[i].camera_serial;

                            auto it = std::find(camera_config_names.begin(), camera_config_names.end(), cameras_params[i].camera_serial + ".json");
                            if (it == camera_config_names.end())
                            {
                                if (strcmp(device_info[selected_cameras[i]].modelName, "HB-65000GM")==0) {
                                    int gpu_id = 0;
                                    init_65MP_camera_params_mono(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 1000, gpu_id, 400); //458 
                                } else if (strcmp(device_info[selected_cameras[i]].modelName, "HB-7000SC")==0) {
                                    int gpu_id = 0;
                                    init_7MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 1500, 2000, gpu_id, 30); // 2000, 3000
                                } else if (strcmp(device_info[selected_cameras[i]].modelName, "HB-65000GC")==0) {
                                    int gpu_id = 0;
                                    init_65MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 28000, gpu_id, 10); 
                                } else if (strcmp(device_info[selected_cameras[i]].modelName, "HB-7000SM")==0) {
                                    int gpu_id = 0;
                                    init_7MP_camera_params_mono(&cameras_params[i], selected_cameras[i], num_cameras, 1000, 3000, gpu_id, 30); // 2000, 3000
                                } else {
                                    printf("Camera not supported...Exit");
                                    return 1;
                                }
                            } else {
                                auto config_idx = std::distance(camera_config_names.begin(), it);
                                std::cout << "Load camera json file: " << camera_config_files[config_idx] << std::endl;
                                load_camera_json_config_files(camera_config_files[config_idx], &cameras_params[i], selected_cameras[i], num_cameras); 
                            }
                        }
                        ecams = new CameraEmergent[num_cameras];
                        for (int i = 0; i < num_cameras; i++)
                        {
                            open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
                            // mcast
                            // string multicast_ip = "239.255.255.255"; // + std::to_string(i);
                            // ecams[i].camera.multicastAddress = multicast_ip.c_str(); 
                            // std::cout << ecams[i].camera.multicastAddress << std::endl;
                            // ecams[i].camera.portMulticast = 60646 + i;
                            // ecams[i].camera.multicastMasterSubscribe = true; 
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

            if (camera_control->open) {
                set_camera_properties(ecams, cameras_params, num_cameras);
            }

            if (ImGui::Button(camera_control->subscribe ? "Stop streaming" : "Start streaming"))
            {
                (camera_control->subscribe) = !(camera_control->subscribe);
                if (camera_control->subscribe)
                {   
                    camera_control->stream = true;     
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
                        cudaStreamCreate(&tex[i].streams);
                        create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                        register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                        map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                        cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                        create_texture(&tex[i].texture, cameras_params[i].width, cameras_params[i].height);
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&aquire_frames, &ecams[i], &cameras_params[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params));
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
                        gx_delete_buffer(&tex[i].pbo);
                        unmap_cuda_resource(&tex[i].cuda_resource);
                        cuda_unregister_pbo(tex[i].cuda_resource);
                    }
                    delete[] tex;
                }
            }

            ImGui::Separator();
            ImGui::Spacing();

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

            if (ImGui::Button("Update Encoding Params"))
            {
                encoder_basic_setup = "-codec " + encoder_codec + " -preset " + encoder_preset + " -fps ";
                std::cout << encoder_basic_setup << std::endl;
            }


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
                    camera_control->record_video = true;
                    std::string folder_string = current_date_time();
                    folder_name = file_dialog.GetSelected().string() + "/" + folder_string;

                    if (mkdir(folder_name.c_str(), 0777) == -1)
                    {
                        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
                        return 0;
                    }
                    else
                    {
                        std::cout << "Recorded video saves to : " << folder_name << std::endl;
                    }
                    
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
                    
                    // camera_control->stream = false;

                    if (camera_control->stream) {
                        tex = new GL_Texture[num_cameras];

                        for (int i = 0; i < num_cameras; i++)
                        {
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

                    for (int i = 0; i < num_cameras; i++)
                    {
                        encoder_setup = encoder_basic_setup + std::to_string(cameras_params[i].frame_rate);
                        camera_threads.push_back(std::thread(&aquire_frames, &ecams[i], &cameras_params[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params));
                    }
                    camera_control->subscribe = true;                    
                } else {
                    camera_control->subscribe = false;
                    camera_control->sync_camera = false;

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
                        if (camera_control->stream) {
                            gx_delete_buffer(&tex[i].pbo);
                            unmap_cuda_resource(&tex[i].cuda_resource);
                            cuda_unregister_pbo(tex[i].cuda_resource);
                        }
                    }
                    if (camera_control->stream) {
                        delete[] tex;                     
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

            if (load_camera_config)
            {
                std::string camera_config_dir = file_dialog.GetSelected().string();

                // load camera_config
                for (const auto &entry : std::filesystem::directory_iterator(camera_config_dir))
                {
                    camera_config_files.push_back(entry.path().string());
                }
                std::sort(camera_config_files.begin(), camera_config_files.end());

                for (auto &camera_serial : camera_config_files) {
                    // get the serial number
                    std::string delimiter = "/";
                    std::vector<std::string> tokenized_path = string_split(camera_serial, delimiter);
                    camera_config_names.push_back(tokenized_path.back());
                }
                file_dialog.ClearSelected();

            } else {
                input_folder = file_dialog.GetSelected().string();
                std::cout << input_folder << std::endl;
                file_dialog.ClearSelected();
            }

        }

        if (camera_control->subscribe && camera_control->stream)
        {
            for (int i = 0; i < num_cameras; i++)
            {
                bind_pbo(&tex[i].pbo);
                bind_texture(&tex[i].texture);
                upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height); // Needs no arguments because texture and PBO are bound
                unbind_pbo();
                unbind_texture();
            }
            
            for (int i = 0; i < num_cameras; i++)
            {
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

    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    return 0;
}
