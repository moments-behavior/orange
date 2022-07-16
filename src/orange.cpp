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
#include "IconsFontAwesome5.h"

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    return buf;
}


int main(int argc, char **args) 
{
    // **************** camera resources ***************************************** 
    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    GigEVisionDeviceInfo ordered_device_info[max_cameras];

    int num_cameras = 2;

    int cam_count;
    cam_count = check_cameras(max_cameras, device_info, ordered_device_info);
    if (cam_count < num_cameras) 
    {
        printf("Missing cameras...Exit\n");
        return 0;
    }


    // set_ip_persistent_with_open_close_camera(device_info, num_cameras);


    // popular change to camera settings 
    unsigned int width {3208}; 
    unsigned int height {2200};
    unsigned int frame_rate {30};
    unsigned int gain {1000}; 
    unsigned int exposure {4000};
    string pixel_format = "BayerRG8"; 
    string color_temp = "CT_2800K";

    // esc to exit 
    int key_num;
    int* key_num_ptr = &key_num;  

    string folder_string = current_date_time();
    string folder_name = "/home/jinyao/Videos/" + folder_string;
    
    // Creating a directory to save recorded video;
    if (mkdir(folder_name.c_str(), 0777) == -1)
    {
        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
        return 0;
    }
    else
        std::cout << "Recorded video saves to : " << folder_name << std::endl;


    PTPParams* ptp_params = new PTPParams{0, 0};
    int camera_orders[] = {0, 1, 2, 3, 4, 5, 6};  
    int camera_gpus[] = {0, 1, 0, 0, 1, 1, 1};
    
    
    int camera_id {0};
    int gpu_id {0};
    int buffer_size {100};

    CameraParams camera_params[num_cameras];

    for(int i = 0; i < num_cameras; i++)
    {
        camera_id = camera_orders[i];
        gpu_id = camera_gpus[camera_id];
        camera_params[i] = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp, camera_id, gpu_id, num_cameras, false);
    }

    // init camera resources 
    Emergent::CEmergentCamera camera[num_cameras];
    Emergent::CEmergentFrame evt_frame[num_cameras][buffer_size]; 
    Emergent::CEmergentFrame frame_recv[num_cameras];

    string encoder_setup = "-preset p1 -fps " + to_string(frame_rate);
    const char *encoder_str = encoder_setup.c_str();
    std::vector<thread> camera_threads;

    for(int i = 0; i < num_cameras; i++)
    {
        //open_camera_with_params(&camera[i], &ordered_device_info[i], camera_params[i]); 
        open_camera_with_params(&camera[i], &device_info[i], camera_params[i]); 

        // sync 
        ptp_camera_sync(&camera[i]);

        allocate_frame_buffer(&camera[i], evt_frame[i], camera_params[i], buffer_size);
        set_frame_buffer(&frame_recv[i], camera_params[i]);
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
    ImGuiIO& io = ImGui::GetIO(); (void)io;
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableKeyboard;       // Enable Keyboard Controls
    // io.ConfigFlags |= ImGuiConfigFlags_NavEnableGamepad;      // Enable Gamepad Controls
    io.ConfigFlags |= ImGuiConfigFlags_DockingEnable;           // Enable Docking
    // io.ConfigFlags |= ImGuiConfigFlags_ViewportsEnable;         // Enable Multi-Viewport / Platform Windows


    ImGui::StyleColorsDark();

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
    static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_16_FA, 0 };
    ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF( FONT_ICON_FILE_NAME_FAS, 15.0f, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid


    // cudaSetDevice(1);

    GLuint texture[num_cameras];
    GLuint pbo[num_cameras];
    cudaGraphicsResource_t cuda_resource[num_cameras] {0};
    unsigned char* cuda_buffer[num_cameras];
    size_t cuda_pbo_storage_buffer_size[num_cameras];
    unsigned char *d_debayer[num_cameras];
    int size_pic = width * height * 4 * sizeof(unsigned char);


    for(int i = 0; i < num_cameras; i++)
    {
        create_texture(&texture[i]);
        create_pbo(&pbo[i], width, height);
        bind_pbo(&pbo[i]);

        register_pbo_to_cuda(&pbo[i], &cuda_resource[i]);
        unbind_texture();
        unbind_pbo();
    }

    for(int i = 0; i < num_cameras; i++){
        cudaSetDevice(camera_gpus[i]);
        cudaMalloc((void **)&d_debayer[i], size_pic);
    }


    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);
    for(int i = 0; i < num_cameras; i++)
    {
        camera_threads.push_back(std::thread(&aquire_frames_gpu_encode, &camera[i], &frame_recv[i], camera_params[i], encoder_str, key_num_ptr, ptp_params, folder_name, d_debayer[i], false));
    }

    cudaSetDevice(0);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        for (int i = 0; i < num_cameras; i++) {

            // Transfer to PBO then OpenGL texture
            // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
            map_cuda_resource(&cuda_resource[i]);
            cuda_pointer_from_resource(&cuda_buffer[i], &cuda_pbo_storage_buffer_size[i], &cuda_resource[i]);
            cudaMemcpy2D(cuda_buffer[i], width*4, d_debayer[i], width*4, width*4, height, cudaMemcpyDeviceToDevice);
            unmap_cuda_resource(&cuda_resource[i]);

            // CUDA-GL INTEROP ENDS HERE ---------------------------------------------------------------------------
            bind_pbo(&pbo[i]);
            bind_texture(&texture[i]);
            upload_image_pbo_to_texture(width, height); // Needs no arguments because texture and PBO are bound
            unbind_texture();
            unbind_pbo();
        }


        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Orange Streaming");
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }
        
        for (int i = 0; i < num_cameras; i++) {
            string window_name = "Cam" + std::to_string(i);            
            ImGui::Begin(window_name.c_str());
            // ImGui::SetNextWindowSize(ImVec2(0, 0), 0); // Setting size to 0, 0 forces auto-fit
            ImGui::Text("pointer = %p", texture[i]);
            ImVec2 avail_size = ImGui::GetContentRegionAvail();
            ImGui::Image((void*)(intptr_t)texture[i], avail_size);
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
    

    for(int i = 0; i < num_cameras; i++)
    {
        destroy_frame_buffer(&camera[i], evt_frame[i], buffer_size);
        close_camera(&camera[i]);
    }


    std::cout << folder_name << std::endl;
    return 0;
}