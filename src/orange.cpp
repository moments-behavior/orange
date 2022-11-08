#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>
#include <GL/glew.h>
#include "gl_helper.h"

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers

#include "shader_m.h"
#include "IconsForkAwesome.h"
#include "implot.h"
#include <imfilebrowser.h>
#include <unistd.h>
#include "buffer_utils.h"
#include "detection.h"

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    return buf;
}


void init_galvo_camera_params(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure)
{
    camera_params->width = 1280;
    camera_params->height = 1280;
    camera_params->frame_rate = 100;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "BayerRG8";
    camera_params->color_temp = "CT_3000K";
    camera_params->camera_id = camera_id;
    camera_params->gpu_id = 1;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
}




void init_25G_camera_params(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int gpu_id)
{
    camera_params->width = 3208;
    camera_params->height = 2200;
    camera_params->frame_rate = 25;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "BayerRG8";
    camera_params->color_temp = "CT_3000K";
    camera_params->camera_id = camera_id;
    camera_params->gpu_id = gpu_id;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
    camera_params->focus = 320;

}



int main(int argc, char **args) 
{
    // **************** camera resources ***************************************** 
    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    GigEVisionDeviceInfo ordered_device_info[max_cameras];

    int num_cameras = 4;

    int cam_count;
    cam_count = order_for_test_rig(max_cameras, device_info, ordered_device_info);
    if (cam_count < num_cameras) 
    {
        printf("Missing cameras...Exit\n");
        return 0;
    }


    // set_ip_persistent_with_open_close_camera(device_info, num_cameras);

    // esc to exit 
    int key_num;
    int* key_num_ptr = &key_num;  
    bool* record_video = new bool(false);
    bool* capture_pause = new bool(false);

    string folder_string = current_date_time();
    string folder_name = "/home/red/Videos/" + folder_string;
    
    // Creating a directory to save recorded video;
    if (mkdir(folder_name.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return 0;
    }
    else
        std::cout << "Recorded video saves to : " << folder_name << std::endl;


    PTPParams* ptp_params = new PTPParams{0, 0};
    int select_camera[] = {0, 1, 2, 3}; 
    int encode_gpu[] = {1, 1, 1, 1};


    int buffer_size {500};
    CameraParams cameras_params[num_cameras];


    for(int i = 0; i < num_cameras; i++)
    {
        int camera_id = select_camera[i];
        int gpu_id = encode_gpu[i];
        init_25G_camera_params(&cameras_params[i], camera_id, num_cameras, 2000, 3000, gpu_id);   
    }


    // init camera resources 
    Emergent::CEmergentCamera camera[num_cameras];
    Emergent::CEmergentFrame evt_frame[num_cameras][buffer_size]; 
    Emergent::CEmergentFrame frame_recv[num_cameras];

    string encoder_setup = "-preset p1 -fps " + to_string(cameras_params[0].frame_rate);
    const char *encoder_str = encoder_setup.c_str();
    std::vector<thread> camera_threads;

    for(int i = 0; i < num_cameras; i++)
    {
        open_camera_with_params(&camera[i], &ordered_device_info[select_camera[i]], &cameras_params[i]); 
        // open_camera_with_params(&camera[i], &device_info[i], camera_params[i]); 

        // sync 
        ptp_camera_sync(&camera[i]);
        EVT_CameraOpenStream(&camera[i]);
        allocate_frame_buffer(&camera[i], evt_frame[i], &cameras_params[i], buffer_size);
        set_frame_buffer(&frame_recv[i], &cameras_params[i]);
    }


    // *********************GUI*****************************************************
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);


    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Orange -- Video Capture App", NULL, NULL);
    if (window == NULL)
        return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync

    // Initialize OpenGL functions with GLEW
    glew_error_callback(glewInit());


    // ************* Dear Imgui ********************//
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImPlotContext* implotCtx = ImPlot::CreateContext();


    ImGuiIO& io = ImGui::GetIO(); (void)io;
    
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows


    ImGui::StyleColorsClassic();

    // When viewports are enabled we tweak WindowRounding/WindowBg so platform windows can look identical to regular ones.
    ImGuiStyle& style = ImGui::GetStyle();
    if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
    {
        style.WindowRounding = 0.0f;
        style.Colors[ImGuiCol_WindowBg].w = 1.0f;
    }


    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load a nice font
    io.Fonts->AddFontFromFileTTF("Roboto-Regular.ttf", 15.0f);
    // merge in icons from Font Awesome
    static const ImWchar icons_ranges[] = { ICON_MIN_FK, ICON_MAX_16_FK, 0 };
    ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF("forkawesome-webfont.ttf", 15.0f, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid

    GLuint texture[num_cameras];
    GLuint pbo[num_cameras];
    cudaGraphicsResource_t cuda_resource[num_cameras] {0};
    unsigned char* cuda_buffer[num_cameras];
    size_t cuda_pbo_storage_buffer_size[num_cameras];
    unsigned char *display_buffer[num_cameras];

    cudaStream_t streams[num_cameras];

    for(int i = 0; i < num_cameras; i++)
    {
        cudaStreamCreate(&streams[i]);
        int size_pic = cameras_params[i].width * cameras_params[i].height * 4 * sizeof(unsigned char);

        create_texture(&texture[i]);
        create_pbo(&pbo[i], cameras_params[i].width, cameras_params[i].height);
        bind_pbo(&pbo[i]);

        register_pbo_to_cuda(&pbo[i], &cuda_resource[i]);
        unbind_texture();
        unbind_pbo();
        cudaMalloc((void **)&display_buffer[i], size_pic);
    }

    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    

    // allocate display buffer
    int size_pic = 3208 * 2200 * 4 * sizeof(unsigned char);
    PictureBuffer display_buffer_cpu[4];
    for(int j=0; j<num_cameras; j++){
        display_buffer_cpu[j].frame = (unsigned char*)malloc(size_pic);
        clear_buffer_with_constant_image(display_buffer_cpu[j].frame, 3208, 2200);
        display_buffer_cpu[j].frame_number = 0;
        display_buffer_cpu[j].available_to_write = true;
    }
    


    
    for(int i = 0; i < num_cameras; i++)
    {
        camera_threads.push_back(std::thread(&aquire_frames_gpu_encode, &camera[i], &frame_recv[i], &cameras_params[i], encoder_str, key_num_ptr, ptp_params, folder_name, display_buffer[i], record_video, capture_pause, &display_buffer_cpu[i]));
    }

    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory | ImGuiFileBrowserFlags_CreateNewDir);
    file_dialog.SetTitle("My files");
    std::string input_folder;

    std::thread t_detection = std::thread(&yolo_detection, display_buffer_cpu, num_cameras, key_num_ptr);


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        for (int i = 0; i < num_cameras; i++) {
            // Transfer to PBO then OpenGL texture
            // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
            map_cuda_resource(&cuda_resource[i]);
            cuda_pointer_from_resource(&cuda_buffer[i], &cuda_pbo_storage_buffer_size[i], &cuda_resource[i]);
            cudaMemcpy2DAsync(cuda_buffer[i], cameras_params[i].width*4, display_buffer[i], cameras_params[i].width*4, cameras_params[i].width*4, cameras_params[i].height, cudaMemcpyDeviceToDevice, streams[i]);
            unmap_cuda_resource(&cuda_resource[i]);
            // CUDA-GL INTEROP ENDS HERE ---------------------------------------------------------------------------
            bind_pbo(&pbo[i]);
            bind_texture(&texture[i]);
            upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height); // Needs no arguments because texture and PBO are bound
            unbind_texture();
            unbind_pbo();
        }
        cudaDeviceSynchronize();

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();


        if (ImGui::Begin("Orange Streaming",  NULL, ImGuiWindowFlags_MenuBar))
        {

            if (ImGui::BeginMenuBar())
            {
                if (ImGui::BeginMenu("File"))
                {
                    if (ImGui::MenuItem("Open")) { file_dialog.Open(); };
                    ImGui::EndMenu();
                }
                ImGui::EndMenuBar();
            }

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);

            bool selected[cam_count] = {};

            // if (ImGui::BeginTable("Cameras", 3, ImGuiTableFlags_Resizable | ImGuiTableFlags_NoSavedSettings | ImGuiTableFlags_Borders))
            // {
            //     for (int i = 0; i < num_cameras; i++)
            //     {
            //         char label[32];
            //         sprintf(label, "Camera %d", cameras_params[i].camera_id);
            //         ImGui::TableNextRow();
            //         ImGui::TableNextColumn();
            //         ImGui::Selectable(label, &selected[i], ImGuiSelectableFlags_SpanAllColumns);
            //         ImGui::TableNextColumn();
            //         ImGui::Text(ordered_device_info[select_camera[i]].serialNumber);
            //         ImGui::TableNextColumn();
            //         ImGui::Text(ordered_device_info[select_camera[i]].currentIp);
            //     }
            //     ImGui::EndTable();
            // }


            ImGui::Separator();
            ImGui::Spacing();

            //if (ImGui::TreeNode("Camera Property"))
            {
                static int selected_camera = 0;
                static int slider_gain, slider_exposure, slider_frame_rate, slider_width, slider_height, OffsetX, OffsetY, slider_focus;

                for (int n = 0; n < num_cameras; n++)
                {
                    char buf[32];
                    sprintf(buf, "Camera %d", select_camera[n]);
                    if (ImGui::Selectable(buf, selected_camera == n))
                        selected_camera = n;
                        slider_gain = cameras_params[selected_camera].gain;
                        slider_focus = cameras_params[selected_camera].focus;
                        slider_width = cameras_params[selected_camera].width;
                        slider_height = cameras_params[selected_camera].height;
                        slider_exposure = cameras_params[selected_camera].exposure;
                        slider_frame_rate = cameras_params[selected_camera].frame_rate; 
                }
            

                // if(ImGui::SliderInt("Width", &slider_width, cameras_params[selected_camera].width_min, cameras_params[0].width_max, "%d"))
                // {
                //     slider_width = (slider_width / 16) * 16; // round to even number

                //     *capture_pause = true;
                //     update_width_value(&camera[selected_camera], slider_width, &cameras_params[selected_camera]);
                //     // set_frame_buffer(&frame_recv[selected_camera], &cameras_params[selected_camera]);
                //     // for(int frame_count=0;frame_count<buffer_size;frame_count++)
                //     // {
                //     //     set_frame_buffer(&evt_frame[selected_camera][frame_count], &cameras_params[selected_camera]);
                //     // }
                //     *capture_pause = false;
                    
                // }

                // if(ImGui::SliderInt("Height", &slider_height, cameras_params[selected_camera].height_min, cameras_params[0].height_max, "%d"))
                // {
                //     slider_height = (slider_height / 16) * 16; // round to even number
                //     update_height_value(&camera[selected_camera], slider_height, &cameras_params[selected_camera]);
                // }


                if(ImGui::SliderInt("OffsetX", &OffsetX, cameras_params[selected_camera].offsetx_min, cameras_params[selected_camera].offsetx_max, "%d"))
                {
                    // round to 16 
                    OffsetX = (OffsetX / 16) * 16; // round to even number
                    // update_offsetX_value(&camera[selected_camera], OffsetX, &cameras_params[selected_camera]);
                    EVT_CameraSetUInt32Param(camera, "OffsetX", OffsetX);

                }


                if(ImGui::SliderInt("OffsetY", &OffsetY, cameras_params[selected_camera].offsety_min, cameras_params[selected_camera].offsety_max, "%d"))
                {
                    // round to 16 
                    OffsetY = (OffsetY / 16) * 16; // round to even number
                    // update_offsetY_value(&camera[selected_camera], OffsetY, &cameras_params[selected_camera]);
                    EVT_CameraSetUInt32Param(camera, "OffsetY", OffsetY);

                }


                if(ImGui::SliderInt("Gain", &slider_gain, cameras_params[selected_camera].gain_min, cameras_params[selected_camera].gain_max, "%d"))
                {
                    update_gain_value(&camera[selected_camera], slider_gain, &cameras_params[selected_camera]);
                }


                if(ImGui::SliderInt("Focus", &slider_focus, cameras_params[selected_camera].focus_min, cameras_params[selected_camera].focus_max, "%d"))
                {
                    update_focus_value(&camera[selected_camera], slider_focus, &cameras_params[selected_camera]);
                }


                if(ImGui::SliderInt("Exposure", &slider_exposure, cameras_params[selected_camera].exposure_min, cameras_params[selected_camera].exposure_max, "%d"))
                {
                    update_exposure_value(&camera[selected_camera], slider_exposure, &cameras_params[selected_camera]);
                }


                if(ImGui::SliderInt("FrameRate", &slider_frame_rate, cameras_params[selected_camera].frame_rate_min, cameras_params[selected_camera].frame_rate_max, "%d"))
                {
                    update_frame_rate_value(&camera[selected_camera], slider_frame_rate, &cameras_params[selected_camera]);
                }
                //ImGui::TreePop();

            }

            ImGui::Separator();
            ImGui::Spacing();
            
            // if (ImGui::Button("Streaming")){}
            // ImGui::SameLine();

            if (*record_video)
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.5f, 0, 0, 1.0f });
            else
                ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0, 0.5f, 0, 1.0f });

            if (ImGui::Button(*record_video ? ICON_FK_PAUSE : ICON_FK_PLAY)){
                (*record_video) = !(*record_video);
            }
            ImGui::PopStyleColor(1);


            ImGui::End();        

        }

        file_dialog.Display();

        
        if (file_dialog.HasSelected())
        {
            input_folder = file_dialog.GetSelected().string();
            std::cout << input_folder << std::endl;
            file_dialog.ClearSelected();
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

        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

        // Update and Render additional Platform Windows
        // (Platform functions may change the current OpenGL context, so we save/restore it to make it easier to paste this code elsewhere.
        //  For this specific demo app we could also call glfwMakeContextCurrent(window) directly)
        if (io.ConfigFlags & ImGuiConfigFlags_ViewportsEnable)
        {
            GLFWwindow* backup_current_context = glfwGetCurrentContext();
            ImGui::UpdatePlatformWindows();
            ImGui::RenderPlatformWindowsDefault();
            glfwMakeContextCurrent(backup_current_context);
        }

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
 
    // exit 
    key_num = 27;


    // wait for threads to join
    for (auto& t : camera_threads)
            t.join();
    
    if (t_detection.joinable()) t_detection.join();

    for(int i = 0; i < num_cameras; i++)
    {
        destroy_frame_buffer(&camera[i], evt_frame[i], buffer_size);
        EVT_CameraOpenStream(&camera[i]);
        close_camera(&camera[i]);
    }

    std::cout << folder_name << std::endl;
    cudaDeviceReset();
    return 0;
}