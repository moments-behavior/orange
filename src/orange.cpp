#include "video_capture_gpu.h"
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
#include "yolo_detection.h"
#define ENET_IMPLEMENTATION
#include "aruco_detection.h"
#include "network.h"
#include "utils.h"
#define MAX_CLIENTS 32

// A nice way of printing out the system time
std::string CurrentTimeStr()
{
    time_t now = time(NULL);
    return std::string(ctime(&now));
}
#define CURRENT_TIME_STR CurrentTimeStr().c_str()

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
    std::vector<std::string> tokenized_path = string_split (cwd, delimiter);
    std::string start_folder_name = "/home/" + tokenized_path[2] + "/exp";

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
    std::string encoder_basic_setup = "-codec hevc -preset p1 -fps "; // h264
    std::string encoder_codec = "hevc";  // h264
    std::string encoder_preset = "p1";
    std::string folder_name;

    CPURender *cpu_buffers;
    CameraCalibResults* calib_results;

    Settings calib_setting;

    std::vector<int> image_save_index;
    bool *selected_images_to_save;

    bool show_cpu_buffer = false;

    ArucoMarker2d marker2d_all_cams;
    ArucoMarker3d marker3d;
    bool have_calibration_results = false;

    yolo_param yolo_setting = yolo_param();
    std::vector<std::vector<cv::Rect>> yolo_boxes;
    std::vector<std::vector<std::string>> yolo_labels;
    std::vector<std::vector<int>> yolo_classid;
    cv::dnn::Net yolo_net;
    std::map<unsigned int, cv::Point3f> yolo_obj_3d;

    bool draw_yolo_detection = false;
    bool draw_aruco_detection = false;

    thread aruco_detection_thread;

    // network and protocal
    if (enet_initialize() != 0)
    {
        printf("An error occurred while initializing ENet.\n");
        return 1;
    }

    ENetAddress address;
    enet_address_set_ip(&address, "127.0.0.1");
    address.port = 6005;

    ENetHost *server = enet_host_create(&address, MAX_CLIENTS, 2, 0, 0, 1024);
    if (server == NULL)
    {
        fprintf(stderr,
                "An error occurred while trying to create an ENet server host.\n");
        exit(EXIT_FAILURE);
    } else {
        printf("Started a server...\n");
    }

    ENetEvent event;

    flatbuffers::FlatBufferBuilder builder(1024);

    while (!glfwWindowShouldClose(window->render_target))
    {

        while (enet_host_service(server, &event, 0) > 0) {
            switch (event.type) {
                case ENET_EVENT_TYPE_CONNECT:
                    printf("\nA new client connected from %x:%u. %s\n", event.peer->host, event.peer->address.port, CURRENT_TIME_STR);
                    /* Store any relevant client information here. */
                    break;

                case ENET_EVENT_TYPE_RECEIVE:
                    printf("\nA packet of length %lu containing %s was received from %s on channel %u. %s.\n",
                        event.packet->dataLength,
                        event.packet->data,
                        event.peer->data,
                        event.channelID,
                        CURRENT_TIME_STR);

                    /* Clean up the packet now that we're done using it. */
                    enet_packet_destroy(event.packet);
                    break;
                case ENET_EVENT_TYPE_DISCONNECT:
                    printf("\n%s disconnected. %s\n", event.peer->data, CURRENT_TIME_STR);
                    /* Reset the peer's client information. */
                    event.peer->data = NULL;
                    break;
                case ENET_EVENT_TYPE_NONE:
                    break;                    
            }
        }

        create_new_frame();

        if (ImGui::Begin("Orange Streaming"))
        {

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

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
                            cameras_params[i].camera_name.append(device_info[selected_cameras[i]].serialNumber);
                            if (strcmp(device_info[selected_cameras[i]].modelName, "HB-65000GM")==0) {
                                int gpu_id = 0;
                                init_65MP_camera_params_mono(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 1000, gpu_id, 400); //458 
                            } else if (strcmp(device_info[selected_cameras[i]].modelName, "HB-7000SC")==0) {
                                int gpu_id = 0;
                                // init_7MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 1500, 2000, gpu_id, 30); 
                                init_7MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 3000, gpu_id, 30); // 2000, 3000
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

                    cpu_buffers = new CPURender[num_cameras];

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &cpu_buffers[i].display_buffer));
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
                const char* items[] = { "hevc", "h264"};
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


            if (camera_control->record_video)
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0.5f, 0, 0, 1.0f});
            }
            else
            {
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{0, 0.5f, 0, 1.0f});
            }

            if (ImGui::Button(camera_control->record_video ? ICON_FK_PAUSE : ICON_FK_PLAY))
            {
                (camera_control->record_video) = !(camera_control->record_video);
                if (camera_control->record_video)
                {
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

                    // if (num_cameras > 1){
                    //     for (int i = 0; i < num_cameras; i++)
                    //     {
                    //         ptp_camera_sync(&ecams[i].camera);
                    //     }
                    //     camera_control->sync_camera = true;
                    // }
                    
                    for (int i = 0; i < num_cameras; i++)
                    {
                        encoder_setup = encoder_basic_setup + std::to_string(cameras_params[i].frame_rate);
                        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].cuda_buffer, encoder_setup, folder_name, ptp_params, &cpu_buffers[i].display_buffer));
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
                }
            }
            ImGui::PopStyleColor(1);
        }
        ImGui::End();

        file_dialog.Display();
        if (file_dialog.HasSelected())
        {
            input_folder = file_dialog.GetSelected().string();
            std::cout << input_folder << std::endl;
            file_dialog.ClearSelected();
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
                std::string window_name = std::to_string(cameras_params[i].camera_id);
                ImGui::Begin(window_name.c_str());
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                // ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size))
                {
                    ImPlot::PlotImage("##no_image_name", (void *)(intptr_t)tex[i].texture, ImVec2(0, 0), ImVec2(cameras_params[i].width, cameras_params[i].height));

                    if (draw_aruco_detection) {
                        draw_aruco_markers(&marker3d, i);
                        send_serial_data(server, builder, &marker3d, yolo_obj_3d);
                    }

                    ImPlot::EndPlot();
                }
                ImGui::End();
            }
        }

        if (ImGui::Begin("Realtime Tools")) 
        {        
            if (ImGui::Button("Allocate Real Time Resources")) {
                
                // allocate cpu buffers
                for (int i = 0; i < num_cameras; i++) {
                    allocate_cpu_render_resources(&cpu_buffers[i], cameras_params[i].width, cameras_params[i].height);
                }
                camera_control->copy_to_cpu = true;

                for (int i = 0; i < num_cameras; i++)
                {
                    image_save_index.push_back(0);
                }

                selected_images_to_save = new bool[num_cameras];
                for (int i = 0; i < num_cameras; i++)
                {
                    selected_images_to_save[i] = false;
                }                      
                show_cpu_buffer = true;
            }


            if (ImGui::Button("Start Aruco Detection")) {

                calib_results = new CameraCalibResults[num_cameras];
                for (int i = 0; i < num_cameras; i++)
                {
                    load_camera_calibration_results(&calib_results[i], &cameras_params[i]);
                    have_calibration_results = true;
                    // print_calibration_results(&calib_results[i]);
                }

                for (int i = 0; i < num_cameras; i++) {
                    std::vector<cv::Point2f> marker_per_cam;
                    for (int j = 0; j < 4; j++) {
                        cv::Point2f each_corner;
                        marker_per_cam.push_back(each_corner);
                    }
                    marker3d.proj_corners.push_back(marker_per_cam);
                }
                aruco_detection_thread = std::thread(marker_detection_thread, cpu_buffers, &marker2d_all_cams, &marker3d, cameras_params, calib_results, camera_control, num_cameras);
                draw_aruco_detection = true;
            }

            if (show_cpu_buffer) {
                
                if (ImGui::Button("Load Yolo Models")) {
                    std::string yolov5_onnx = "/home/user/dev/clips0/yolo_models/best.onnx";
                    std::string yolov5_labelname = "/home/user/dev/clips0/yolo_models/label.names";
                    read_yolo_labels(yolov5_labelname, &yolo_setting);

                    yolo_net = cv::dnn::readNet(yolov5_onnx);
                    yolo_net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
                    yolo_net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
                    std::cout << "model loaded" << std::endl;

                    for (int i = 0; i < num_cameras; i++)
                    {
                        std::vector<cv::Rect> yolo_box_per_cam;
                        std::vector<std::string> yolo_label_per_cam;
                        std::vector<int> yolo_classid_per_cam;
                        yolo_boxes.push_back(yolo_box_per_cam);
                        yolo_labels.push_back(yolo_label_per_cam);
                        yolo_classid.push_back(yolo_classid_per_cam);
                    }
                }

                if (ImGui::Button("Yolo Detect")) {
                    
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }


                    for (int i = 0; i < num_cameras; i++) {
                        yolo_detection(yolo_net, &yolo_setting, &cpu_buffers[i].display_buffer, i, yolo_boxes, yolo_labels, yolo_classid);
                    }

                    yolo_obj_3d = get_3d_coordinates(yolo_boxes, yolo_classid, calib_results);
                    for ( auto it = yolo_obj_3d.begin(); it != yolo_obj_3d.end(); ++it) {
                            std::cout << "yolo_object: " << it->first << ", " << it->second << std::endl;
                    }
                    draw_yolo_detection = true;
                }

                ImGui::Separator();
                ImGui::Spacing();

                if (ImGui::Button("Get new frame"))
                {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = true;
                    }
                    draw_yolo_detection = false;
                    draw_aruco_detection = false;
                }

                
                if (ImGui::Button("Save images all"))
                {

                    folder_name = file_dialog.GetSelected().string();

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i].display_buffer.frame).reshape(3, 2200);
                        std::string image_name = folder_name + "/Cam" + std::to_string(cameras_params[i].camera_id) + "_image" + std::to_string(image_save_index[i]) + ".tif";
                        // string image_name = "/home/user/Calibration/realtime_calib/" + std::to_string(cameras_params[i].camera_id) + "-" + std::to_string(image_save_index[i]) + ".png";
                        std::cout << image_name << std::endl;
                        
                        cv::imwrite(image_name, view);
                        image_save_index[i]++;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = true;
                    }
                }

               
                for (int i = 0; i < num_cameras; i++)
                {
                    ImGui::InputInt("Saving image index: ", &image_save_index[i]);
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    char label[32];
                    sprintf(label, "Cam%d", cameras_params[i].camera_id);
                    ImGui::Checkbox(label, &selected_images_to_save[i]);
                    ImGui::SameLine();
                }

                if (ImGui::Button("Save selected"))
                {
                    folder_name = file_dialog.GetSelected().string();

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        if (selected_images_to_save[i])
                        {
                            cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i].display_buffer.frame).reshape(3, 2200);
                            string image_name = folder_name + "/Cam" + std::to_string(cameras_params[i].camera_id) + "_image" + std::to_string(image_save_index[i]) + ".tif";
                            std::cout << image_name << std::endl;
                            cv::imwrite(image_name, view);
                            image_save_index[i]++;
                        }
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = true;
                    }
                }


                for(int i=0; i < num_cameras; i++){
                    bind_texture(&cpu_buffers[i].image_texture);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, cameras_params[i].width, cameras_params[i].height, GL_RGB, GL_UNSIGNED_BYTE, cpu_buffers[i].display_buffer.frame);
                    unbind_texture();
                }

                for (int i = 0; i < num_cameras; i++) {
                    string window_name = "CCam" + std::to_string(cameras_params[i].camera_id);            
                    ImGui::Begin(window_name.c_str());
                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    //ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                    if (ImPlot::BeginPlot("##no_plot_name", avail_size)){
                        ImPlot::PlotImage("##no_image_name", (void*)(intptr_t)cpu_buffers[i].image_texture, ImVec2(0,0), ImVec2(cameras_params[i].width, cameras_params[i].height));

                        if (have_calibration_results)
                        {
                            gui_plot_world_coordinates(&calib_results[i], i);
                        }

                        if (draw_yolo_detection) {
                            draw_yolo_boxes(yolo_boxes.at(i), yolo_labels.at(i), yolo_classid.at(i));
                        }

                        ImPlot::EndPlot();
                    }
                    ImGui::End();            
                }
            }
                     
        }
        ImGui::End();   

        render_a_frame(window);

    }

    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    return 0;
}
