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
#include "realtime_tool.h"
#include "yolo_detection.h"
#include "fetch_generated.h"
#define ENET_IMPLEMENTATION
#include "enet.h"

#define MAX_CLIENTS 32
#include <random>

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
    file_dialog.SetPwd("/home/user/exp");
    std::string input_folder = file_dialog.GetPwd();
    file_dialog.SetTitle("My files");

    bool check[cam_count] = {};

    CameraParams *cameras_params;
    CameraEmergent *ecams;
    std::vector<thread> camera_threads;
    GL_Texture *tex;
    u32 num_cameras = 0;

    CameraControl *camera_control = new CameraControl;

    // int buffer_size {500};
    PTPParams* ptp_params = new PTPParams{0, 0};
    std::string encoder_setup;
    std::string encoder_basic_setup = "-codec hevc -preset p1 -fps "; // h264
    std::string encoder_codec = "hevc";  // h264
    std::string encoder_preset = "p1";
    std::string folder_name;

    CPURender *cpu_buffers;
    CameraCalibResults* calib_results;
    vector<vector<vector<Point2f>>> calib_data;

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
                                init_7MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 1500, 2000, gpu_id, 30); 
                                // init_7MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 3000, gpu_id, 30); // 2000, 3000
                            } else if (strcmp(device_info[selected_cameras[i]].modelName, "HB-65000GC")==0) {
                                int gpu_id = 0;
                                init_65MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 18000, gpu_id, 30); 
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
                        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], 100);
                        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
                        {
                            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
                        }
                        set_frame_buffer(&ecams[i].frame_recv, &cameras_params[i]);
                    }

                    tex = new GL_Texture[num_cameras];
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cudaStreamCreate(&tex[i].streams);
                        int size_pic = cameras_params[i].width * cameras_params[i].height * 4 * sizeof(unsigned char);

                        create_texture(&tex[i].texture);
                        create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                        bind_pbo(&tex[i].pbo);

                        register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                        unbind_texture();
                        unbind_pbo();
                        cudaMalloc((void **)&tex[i].display_buffer, size_pic);
                    }

                    cpu_buffers = new CPURender[num_cameras];

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].display_buffer, encoder_setup, folder_name, ptp_params, &cpu_buffers[i].display_buffer));
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
                        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, 100);
                        EVT_CameraCloseStream(&ecams[i].camera);
                        gx_delete_buffer(&tex[i].pbo);
                        cudaFree(tex[i].display_buffer);  
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
                    string folder_string = current_date_time();
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
                        EVT_CameraOpenStream(&ecams[i].camera);
                        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], 100);
                        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct)
                        {
                            allocate_frame_reorder_buffer(&ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
                        }
                        set_frame_buffer(&ecams[i].frame_recv, &cameras_params[i]);
                    }
                    
                    // camera_control->stream = false;

                    if (camera_control->stream) {
                        tex = new GL_Texture[num_cameras];
                        for (int i = 0; i < num_cameras; i++)
                        {
                            cudaStreamCreate(&tex[i].streams);
                            int size_pic = cameras_params[i].width * cameras_params[i].height * 4 * sizeof(unsigned char);

                            create_texture(&tex[i].texture);
                            create_pbo(&tex[i].pbo, cameras_params[i].width, cameras_params[i].height);
                            bind_pbo(&tex[i].pbo);

                            register_pbo_to_cuda(&tex[i].pbo, &tex[i].cuda_resource);
                            unbind_texture();
                            unbind_pbo();
                            cudaMalloc((void **)&tex[i].display_buffer, size_pic);
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
                        encoder_setup = encoder_basic_setup + to_string(cameras_params[i].frame_rate);
                        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].display_buffer, encoder_setup, folder_name, ptp_params, &cpu_buffers[i].display_buffer));
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
                        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, 100);
                        EVT_CameraCloseStream(&ecams[i].camera);
                        if (camera_control->stream) {
                            gx_delete_buffer(&tex[i].pbo);
                            cudaFree(tex[i].display_buffer);  
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
                // Transfer to PBO then OpenGL texture
                // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
                map_cuda_resource(&tex[i].cuda_resource, tex[i].streams);
                cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                cudaMemcpy2DAsync(tex[i].cuda_buffer, cameras_params[i].width * 4, tex[i].display_buffer, cameras_params[i].width * 4, cameras_params[i].width * 4, cameras_params[i].height, cudaMemcpyDeviceToDevice, tex[i].streams);
                unmap_cuda_resource(&tex[i].cuda_resource);
                // CUDA-GL INTEROP ENDS HERE ---------------------------------------------------------------------------
                bind_pbo(&tex[i].pbo);
                bind_texture(&tex[i].texture);
                upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height); // Needs no arguments because texture and PBO are bound
                unbind_texture();
                unbind_pbo();
            }
            
            for (int i = 0; i < num_cameras; i++)
            {
                string window_name = std::to_string(cameras_params[i].camera_id);
                ImGui::Begin(window_name.c_str());
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                // ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size))
                {
                    ImPlot::PlotImage("##no_image_name", (void *)(intptr_t)tex[i].texture, ImVec2(0, 0), ImVec2(cameras_params[i].width, cameras_params[i].height));
                    ImPlot::EndPlot();
                }
                ImGui::End();
            }
        }

        if (ImGui::Begin("Realtime Tools")) 
        {
            
            if (ImGui::Button("Allocate cpu buffers")) {
                for (int i = 0; i < num_cameras; i++) {
                    allocate_cpu_render_resources(&cpu_buffers[i], cameras_params[i].width, cameras_params[i].height);
                }
                show_cpu_buffer = true;
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
                calib_results = new CameraCalibResults[num_cameras];

                for (int i = 0; i < num_cameras; i++)
                {
                    vector<vector<Point2f>> image_points_per_cam;
                    calib_data.push_back(image_points_per_cam);
                }
            }
            ImGui::Separator();
            ImGui::Spacing();

            if (show_cpu_buffer) {
                
                if (ImGui::Button("Load camera calibration")) {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        load_camera_calibration_results(&calib_results[i], &cameras_params[i]);
                        have_calibration_results = true;
                        // print_calibration_results(&calib_results[i]);
                    }
                }

                if (ImGui::Button("Detect Aruco Marker")) {

                    if (!marker2d_all_cams.detected_cameras.empty()) {
                        marker2d_all_cams.detected_cameras.clear();
                        marker2d_all_cams.detected_points.clear();
                        marker3d.corners.clear();
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }


                    for (int i = 0; i < num_cameras; i++) {
                        aruco_detection(&cpu_buffers[i].display_buffer, cameras_params, &marker2d_all_cams); 
                    } 

                    if(find_marker3d(&marker2d_all_cams, &marker3d, calib_results)) {
                        std::cout << "Marker tvec: " << marker3d.t_vec << std::endl;
                        std::cout << "Marker normal: " << marker3d.normal << std::endl;
                    }
                }

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

                if (ImGui::Button("Load Calibration Configure File")) {
                    std::string input_settings_file = "/home/user/src/orange/default.xml";
                    load_calibration_config_file(input_settings_file, &calib_setting);
                
                    for (int i = 0; i < num_cameras; i++)
                    {

                        Mat cameraMatrix = Mat::eye(3, 3, CV_64F);
                        // initialization
                        if (!calib_setting.intrinsicGuess)
                        {
                            if (!calib_setting.useFisheye && calib_setting.flag & CALIB_FIX_ASPECT_RATIO)
                                cameraMatrix.at<double>(0, 0) = calib_setting.aspectRatio;
                        }
                        else
                        {
                            cameraMatrix = (Mat_<double>(3, 3) << 2800, 0, 1100, 0, 2800, 1600, 0, 0, 1);
                        }
                        calib_results[i].k = cameraMatrix;

                        if (calib_setting.useFisheye)
                        {
                            Mat distCoeffs = Mat::zeros(4, 1, CV_64F);
                            calib_results[i].dist_coeffs = distCoeffs;
                        }
                        else
                        {
                            Mat distCoeffs = Mat::zeros(8, 1, CV_64F);
                            calib_results[i].dist_coeffs.push_back(distCoeffs);
                        }
                    }
                }

                if (ImGui::Button("Detect"))
                {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {

                        int winSize = 11; // Half of search window for cornerSubPix
                        // local the frame and process frame
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i].display_buffer.frame).reshape(3, 2200);

                        //! [find_pattern]
                        vector<Point2f> pointBuf;
                        bool found;
                        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

                        switch (calib_setting.calibrationPattern) // Find feature points on the input format
                        {
                        case Settings::CHESSBOARD:
                            found = findChessboardCorners(view, calib_setting.boardSize, pointBuf);
                            break;
                        case Settings::CIRCLES_GRID:
                            found = findCirclesGrid(view, calib_setting.boardSize, pointBuf);
                            break;
                        case Settings::ASYMMETRIC_CIRCLES_GRID:
                            found = findCirclesGrid(view, calib_setting.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
                            std::cout << "here?" << std::endl;
                            break;
                        default:
                            found = false;
                            break;
                        }

                        std::cout << "\n after finding corner?:" << found << std::endl;
                        //! [find_pattern]
                        //! [pattern_found]
                        if (found) // If done with success,
                        {
                            // improve the found corners' coordinate accuracy for chessboard
                            if (calib_setting.calibrationPattern == Settings::CHESSBOARD)
                            {
                                Mat viewGray;
                                cvtColor(view, viewGray, COLOR_BGR2GRAY);
                                cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
                                                Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
                            }

                            calib_data[i].push_back(pointBuf);
                            std::cout << pointBuf << std::endl;
                            // Draw the corners.
                            drawChessboardCorners(view, calib_setting.boardSize, Mat(pointBuf), found);
                            bitwise_not(view, view);
                        }
                    }
                }


                if (ImGui::Button("Get new frame"))
                {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = true;
                    }
                    draw_yolo_detection = false;
                    draw_aruco_detection = false;
                }

                for (int i = 0; i < num_cameras; i++)
                {
                    int no_frames = calib_data[i].size();
                    std::string no_frames_str = "Number of Frames: " + std::to_string(no_frames);
                    if (no_frames < 25)
                    {
                        ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), no_frames_str.c_str());
                    }
                    else
                    {
                        ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), no_frames_str.c_str());
                    }
                }

                if (ImGui::Button("Run calibration"))
                {
                    float grid_width = calib_setting.squareSize * (calib_setting.boardSize.width - 1);
                    Size imageSize = cv::Size(2200, 3200);
                    cout << "imageSize" << imageSize << endl;

                    for (int i = 0; i < num_cameras; i++)
                    {
                        string cam_calib_out = "/home/user/Calibration/realtime_calib/Cam" + std::to_string(cameras_params[i].camera_id) + ".xml";
                        if (runCalibrationAndSave(cam_calib_out, calib_setting, imageSize, calib_results[i].k, calib_results[i].dist_coeffs, calib_data[i], grid_width, calib_setting.release_object))
                        {
                            printf("Calibrated");
                        }
                    }
                }

                if (ImGui::Button("Load intrinsics"))
                {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        string input_intrinsic_files = "Cam" + std::to_string(cameras_params[i].camera_id) + ".xml";
                        loadIntrinsics(input_intrinsic_files, calib_results[i].k, calib_results[i].dist_coeffs);
                    }
                }

                if (ImGui::Button("Estimate camera pose"))
                {

                    // detect
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        int winSize = 11; // Half of search window for cornerSubPix
                        // local the frame and process frame
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i].display_buffer.frame).reshape(3, 2200);

                        //! [find_pattern]
                        vector<Point2f> pointBuf;
                        bool found;
                        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

                        switch (calib_setting.calibrationPattern) // Find feature points on the input format
                        {
                        case Settings::CHESSBOARD:
                            found = findChessboardCorners(view, calib_setting.boardSize, pointBuf);
                            break;
                        case Settings::CIRCLES_GRID:
                            found = findCirclesGrid(view, calib_setting.boardSize, pointBuf);
                            break;
                        case Settings::ASYMMETRIC_CIRCLES_GRID:
                            found = findCirclesGrid(view, calib_setting.boardSize, pointBuf, CALIB_CB_ASYMMETRIC_GRID);
                            std::cout << "here?" << std::endl;
                            break;
                        default:
                            found = false;
                            break;
                        }

                        std::cout << "\n after finding corner?:" << found << std::endl;
                        //! [find_pattern]
                        //! [pattern_found]
                        if (found) // If done with success,
                        {
                            // improve the found corners' coordinate accuracy for chessboard
                            if (calib_setting.calibrationPattern == Settings::CHESSBOARD)
                            {
                                Mat viewGray;
                                cvtColor(view, viewGray, COLOR_BGR2GRAY);
                                cornerSubPix(viewGray, pointBuf, Size(winSize, winSize),
                                                Size(-1, -1), TermCriteria(TermCriteria::EPS + TermCriteria::COUNT, 30, 0.0001));
                            }
                            calib_data[i].push_back(pointBuf);
                            // Draw the corners.
                            drawChessboardCorners(view, calib_setting.boardSize, Mat(pointBuf), found);
                            bitwise_not(view, view);

                            string cam_calib_estrinsics = "Cam" + std::to_string(cameras_params[i].camera_id) + "_extrinsics.xml";
                            // estimate extrinsics
                            if (estimatePose(cam_calib_estrinsics, calib_setting, pointBuf, calib_results[i].k, calib_results[i].dist_coeffs, SOLVEPNP_ITERATIVE))
                            {
                                std::cout << "Extrinsics estimated successfully." << std::endl;
                            }
                        }
                    }
                }


                ImGui::Separator();
                ImGui::Spacing();
                
                if (ImGui::Button("Save images all"))
                {

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i].display_buffer.frame).reshape(3, 2200);
                        string image_name = "/home/user/Calibration/rob_calibration/Cam" + std::to_string(cameras_params[i].camera_id) + "/image_" + std::to_string(image_save_index[i]) + ".tif";
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
                    sprintf(label, "Cam%d", i);
                    ImGui::Checkbox(label, &selected_images_to_save[i]);
                    ImGui::SameLine();
                }

                if (ImGui::Button("Save selected"))
                {
                    for (int i = 0; i < num_cameras; i++)
                    {
                        cpu_buffers[i].display_buffer.available_to_write = false;
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        if (selected_images_to_save[i])
                        {
                            cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, cpu_buffers[i].display_buffer.frame).reshape(3, 2200);
                            string image_name = "/home/user/Calibration/rob_calibration/Cam" + std::to_string(cameras_params[i].camera_id) + "/image_" + std::to_string(image_save_index[i]) + ".tif";
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

                        if (draw_aruco_detection) {
                            draw_aruco_markers(&marker2d_all_cams, i);
                        }

                        ImPlot::EndPlot();
                    }
                    ImGui::End();            
                }
            }
                     
        }
        ImGui::End();   

        {
            ImGui::Begin("Unity Rendering");

            if (ImGui::Button("Send Scene Info"))
            {
                
                // build the buffer
                auto position_ramp = FetchGame::Vec3(marker3d.t_vec.x, marker3d.t_vec.y, marker3d.t_vec.z);
                float orientation_ramp = marker3d.angle_x_axis;
                auto ramp_fb = CreateRamp(builder, &position_ramp, orientation_ramp);


                auto position_ball = FetchGame::Vec3(yolo_obj_3d[0].x, yolo_obj_3d[0].y, yolo_obj_3d[0].z); 
                auto ball_fb = CreateBall(builder, &position_ball);

                FetchGame::SceneBuilder scene_builder(builder);
                scene_builder.add_ramp(ramp_fb);
                scene_builder.add_ball(ball_fb);
                auto scene = scene_builder.Finish();
                builder.Finish(scene);
                uint8_t *buf = builder.GetBufferPointer();
                int size = builder.GetSize();

                ENetPacket *packet = enet_packet_create(buf,
                                                        size,
                                                        ENET_PACKET_FLAG_RELIABLE);
                /* Send the packet to the peer over channel id 0. */
                /* One could also broadcast the packet by         */
                enet_host_broadcast(server, 0, packet);
                // enet_peer_send(peer, 0, packet);

                // Receive some events
                // enet_host_service(client, &event, 0);
            }
            ImGui::End();
        }

        render_a_frame(window);

    }

    // Cleanup
    gx_cleanup(window);
    cudaDeviceReset();
    return 0;
}
