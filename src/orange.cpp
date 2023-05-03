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
#include "camera_calibration.h"

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
    std::string encoder_basic_setup = "-codec h264 -preset p1 -fps ";
    std::string encoder_codec = "h264"; 
    std::string encoder_preset = "p1";
    std::string folder_name;

    // for camera calibration
    
    // allocate display buffer on cpu, make this dynamic later
    int output_channels = 3;
    int size_pic = 3208 * 2200 * output_channels * sizeof(unsigned char);
    PictureBuffer display_buffer_cpu[4];
    for(int j=0; j<4; j++){
        display_buffer_cpu[j].frame = (unsigned char*)malloc(size_pic);
        clear_buffer_with_constant_image(display_buffer_cpu[j].frame, 3208, 2200);
        display_buffer_cpu[j].frame_number = 0;
        display_buffer_cpu[j].available_to_write = true;
    }

    GLuint texture[4];

    for(int j=0; j<4; j++){
        glGenTextures(1, &texture[j]);
        glBindTexture(GL_TEXTURE_2D, texture[j]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 3208, 2200, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        // Setup filtering parameters for display
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE); // This is required on WebGL for non power-of-two textures
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE); // Same
    }
    Settings calib_setting;
    vector<vector<Point2f>> imagePoints;
    Mat camera_matrices;
    Mat dist_coeffs;


    while (!glfwWindowShouldClose(window->render_target))
    {
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
                                init_7MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 1500, 2000, gpu_id, 30); // 2000, 3000
                            } else if (strcmp(device_info[selected_cameras[i]].modelName, "HB-65000GC")==0) {
                                int gpu_id = 0;
                                init_65MP_camera_params_color(&cameras_params[i], selected_cameras[i], num_cameras, 2000, 28000, gpu_id, 10); 
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

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].display_buffer, encoder_setup, folder_name, ptp_params, &display_buffer_cpu[i]));
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
                        camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].display_buffer, encoder_setup, folder_name, ptp_params, &display_buffer_cpu[i]));
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


        if (ImGui::Begin("Calibration"))
        {
            if (ImGui::Button("Load config file")){
                const string inputSettingsFile = "/home/user/src/orange/circle.xml";
                FileStorage fs(inputSettingsFile, FileStorage::READ);
                if (!fs.isOpened())
                {
                    std::cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << std::endl;
                    return -1;
                } 
                fs["Settings"] >> calib_setting;
                fs.release();

                if (!calib_setting.goodInput)
                {
                    cout << "Invalid input detected. Application stopping. " << endl;
                    return -1;
                } else {
                    std::cout << "Calibration configuration file loaded: \"" << inputSettingsFile << "\"" << std::endl;
                }
                camera_control->copy_to_cpu = true;
                camera_control->calibration = true;
            }

            if (camera_control->calibration) {
                
                if (ImGui::Button("Detect")) 
                {

                    for(int i=0; i < num_cameras; i++){
                        display_buffer_cpu[i].available_to_write = false;
                    }
                    
                    for(int i=0; i < num_cameras; i++) {

                        int winSize = 11;  // Half of search window for cornerSubPix
                        // local the frame and process frame 
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, display_buffer_cpu[i].frame).reshape(3, 2200);

                        //! [find_pattern]
                        vector<Point2f> pointBuf;
                        bool found;
                        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

                        switch(calib_setting.calibrationPattern) // Find feature points on the input format
                        {
                            case Settings::CHESSBOARD:
                                found = findChessboardCorners(view, calib_setting.boardSize, pointBuf);
                                break;
                            case Settings::CIRCLES_GRID:
                                found = findCirclesGrid(view, calib_setting.boardSize, pointBuf );
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
                        if (found)                // If done with success,
                        {
                            // improve the found corners' coordinate accuracy for chessboard
                                if(calib_setting.calibrationPattern == Settings::CHESSBOARD)
                                {
                                    Mat viewGray;
                                    cvtColor(view, viewGray, COLOR_BGR2GRAY);
                                    cornerSubPix( viewGray, pointBuf, Size(winSize,winSize),
                                        Size(-1,-1), TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001));
                                }
                                if (i==0){
                                    imagePoints.push_back(pointBuf);                                
                                }
                                std::cout << pointBuf << std::endl;
                                // Draw the corners.
                                drawChessboardCorners(view, calib_setting.boardSize, Mat(pointBuf), found);
                                bitwise_not(view, view);
                        }
                    }

                }

                if (ImGui::Button("Get new frame")) {
                    for(int i=0; i < num_cameras; i++){
                        display_buffer_cpu[i].available_to_write = true;
                    }
                }

                int no_frames = imagePoints.size();
                std::string no_frames_str = "Number of Frames: " + std::to_string(no_frames);
                if(no_frames < 25)
                {
                    ImGui::TextColored(ImVec4(1.0f, 0.0f, 1.0f, 1.0f), no_frames_str.c_str());
                } else {
                    ImGui::TextColored(ImVec4(0.0f, 1.0f, 1.0f, 1.0f), no_frames_str.c_str());
                }
                
                if(ImGui::Button("Run calibration"))
                {
                    float grid_width = calib_setting.squareSize * (calib_setting.boardSize.width - 1);
                    Size imageSize = cv::Size(2200, 3200);
                    cout << "imageSize" << imageSize << endl;

                    // for(int i=0; i < num_cameras; i++){
                        if(runCalibrationAndSave(calib_setting, imageSize, camera_matrices, dist_coeffs, imagePoints, grid_width, false)) {
                            printf("Calibrated");
                        }                                        
                    // } 
                }

                if(ImGui::Button("Estimate camera pose")){
                    
                    // detect 
                    for(int i=0; i < num_cameras; i++){
                        display_buffer_cpu[i].available_to_write = false;
                    }
                    
                    // for(int i=0; i < num_cameras; i++) 
                    {
                        int i = 0;
                        int winSize = 11;  // Half of search window for cornerSubPix
                        // local the frame and process frame 
                        cv::Mat view = cv::Mat(3208 * 2200 * 3, 1, CV_8U, display_buffer_cpu[i].frame).reshape(3, 2200);

                        //! [find_pattern]
                        vector<Point2f> pointBuf;
                        bool found;
                        int chessBoardFlags = CALIB_CB_ADAPTIVE_THRESH | CALIB_CB_NORMALIZE_IMAGE | CALIB_CB_FAST_CHECK;

                        switch(calib_setting.calibrationPattern) // Find feature points on the input format
                        {
                            case Settings::CHESSBOARD:
                                found = findChessboardCorners(view, calib_setting.boardSize, pointBuf);
                                break;
                            case Settings::CIRCLES_GRID:
                                found = findCirclesGrid(view, calib_setting.boardSize, pointBuf );
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
                        if (found)                // If done with success,
                        {
                            // improve the found corners' coordinate accuracy for chessboard
                                if(calib_setting.calibrationPattern == Settings::CHESSBOARD)
                                {
                                    Mat viewGray;
                                    cvtColor(view, viewGray, COLOR_BGR2GRAY);
                                    cornerSubPix(viewGray, pointBuf, Size(winSize,winSize),
                                        Size(-1,-1), TermCriteria(TermCriteria::EPS+TermCriteria::COUNT, 30, 0.0001));
                                }
                                if (i==0){
                                    imagePoints.push_back(pointBuf);                                
                                }
                                // Draw the corners.
                                drawChessboardCorners(view, calib_setting.boardSize, Mat(pointBuf), found);
                                bitwise_not(view, view);

                                // estimate extrinsics 
                                if(estimatePose(calib_setting, pointBuf, camera_matrices, dist_coeffs, SOLVEPNP_ITERATIVE)){
                                    std::cout << "Extrinsics estimated successfully." << std::endl;
                                }
                        }
                    }

                    

                }

                for(int j=0; j<num_cameras; j++){
                    // if the current frame is ready, upload for display, otherwise wait for the frame to get ready 
                    bind_texture(&texture[j]);
                    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, 3208, 2200, GL_RGB, GL_UNSIGNED_BYTE, display_buffer_cpu[j].frame);
                    unbind_texture();
                }

                for (int i = 0; i < num_cameras; i++) {
                    string window_name = "Cam" + std::to_string(cameras_params[i].camera_id);            
                    ImGui::Begin(window_name.c_str());
                    ImVec2 avail_size = ImGui::GetContentRegionAvail();

                    //ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                    if (ImPlot::BeginPlot("##no_plot_name", avail_size)){
                        ImPlot::PlotImage("##no_image_name", (void*)(intptr_t)texture[i], ImVec2(0,0), ImVec2(cameras_params[i].width, cameras_params[i].height));
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
