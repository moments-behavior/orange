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
    string folder_name = "/home/user/Videos/" + folder_string;
    
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
        camera_params[i] = create_camera_params(width, height, frame_rate, gain, exposure, pixel_format, color_temp, camera_id, gpu_id, num_cameras);
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


    // *********************************my camera rendering pipeline ************************************//
    Shader screenShader("shader_picture.vs", "shader_picture.fs");
    float quadVertices[] = { // vertex attributes for a quad that fills the entire screen in Normalized Device Coordinates.
        // positions   // texCoords
        -1.0f,  1.0f,  0.0f, 1.0f,
        -1.0f, -1.0f,  0.0f, 0.0f,
         1.0f, -1.0f,  1.0f, 0.0f,

        -1.0f,  1.0f,  0.0f, 1.0f,
         1.0f, -1.0f,  1.0f, 0.0f,
         1.0f,  1.0f,  1.0f, 1.0f
    };

    unsigned int quadVAO, quadVBO;
    glGenVertexArrays(1, &quadVAO);
    glGenBuffers(1, &quadVBO);
    glBindVertexArray(quadVAO);
    glBindBuffer(GL_ARRAY_BUFFER, quadVBO);
    glBufferData(GL_ARRAY_BUFFER, sizeof(quadVertices), &quadVertices, GL_STATIC_DRAW);
    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)0);
    glEnableVertexAttribArray(1);
    glVertexAttribPointer(1, 2, GL_FLOAT, GL_FALSE, 4 * sizeof(float), (void*)(2 * sizeof(float)));

    screenShader.use();
    screenShader.setInt("screenTexture", 0);

    // framebuffer configuration
    // -------------------------
    unsigned int framebuffer[num_cameras];
    unsigned int textureColorbuffer[num_cameras];

    for (int j = 0; j < num_cameras; j++) {
        glGenFramebuffers(1, &framebuffer[j]);
        glBindFramebuffer(GL_FRAMEBUFFER, framebuffer[j]);
        // create a color attachment texture
        glGenTextures(1, &textureColorbuffer[j]);
        glBindTexture(GL_TEXTURE_2D, textureColorbuffer[j]);
        glTexImage2D(GL_TEXTURE_2D, 0, GL_RGB, 3208, 2200, 0, GL_RGB, GL_UNSIGNED_BYTE, NULL);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR);
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR);
        glFramebufferTexture2D(GL_FRAMEBUFFER, GL_COLOR_ATTACHMENT0, GL_TEXTURE_2D, textureColorbuffer[j], 0);

        if (glCheckFramebufferStatus(GL_FRAMEBUFFER) != GL_FRAMEBUFFER_COMPLETE)
            std::cout << "ERROR::FRAMEBUFFER:: Framebuffer is not complete!" << std::endl;
        glBindFramebuffer(GL_FRAMEBUFFER, 0);
    }

    // ************* Dear Imgui ********************//
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    // Load a nice font
    io.Fonts->AddFontFromFileTTF("Roboto-Regular.ttf", 15.0f);
    // merge in icons from Font Awesome
    static const ImWchar icons_ranges[] = { ICON_MIN_FA, ICON_MAX_16_FA, 0 };
    ImFontConfig icons_config; icons_config.MergeMode = true; icons_config.PixelSnapH = true;
    io.Fonts->AddFontFromFileTTF( FONT_ICON_FILE_NAME_FAS, 15.0f, &icons_config, icons_ranges);
    // use FONT_ICON_FILE_NAME_FAR if you want regular instead of solid

    GLuint texture;
    GLuint pbo;
    cudaGraphicsResource_t cuda_resource = 0;
    unsigned char* cuda_buffer;
    size_t cuda_pbo_storage_buffer_size;
    create_texture(&texture);
    create_pbo(&pbo, width, height);
    bind_pbo(&pbo);
    register_pbo_to_cuda(&pbo, &cuda_resource);
    unbind_texture();
    unbind_pbo();

    unsigned char *d_debayer[num_cameras];
    int size_pic = width * height * 4 * sizeof(unsigned char);
    for(int i = 0; i < num_cameras; i++){
        cudaMalloc((void **)&d_debayer[i], size_pic);
    }

    ImVec4 clear_color = ImVec4(0.0f, 0.0f, 0.0f, 1.00f);


    for(int i = 0; i < num_cameras; i++)
    {
        camera_threads.push_back(std::thread(&aquire_frames_gpu_encode, &camera[i], &frame_recv[i], camera_params[i], encoder_str, key_num_ptr, ptp_params, folder_name, d_debayer[i]));
    }


    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();

        // for (int i = 0; i < num_cameras; i++) {
        //     // Transfer to PBO then OpenGL texture
        //     // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
        //     map_cuda_resource(&cuda_resource);
        //     cuda_pointer_from_resource(&cuda_buffer, &cuda_pbo_storage_buffer_size, &cuda_resource);
        //     cudaMemcpy2D(cuda_buffer, width*4, d_debayer[i], width*4, width*4, height, cudaMemcpyDeviceToDevice);
        //     unmap_cuda_resource(&cuda_resource);

        //     // CUDA-GL INTEROP ENDS HERE ---------------------------------------------------------------------------
        //     bind_pbo(&pbo);
        //     bind_texture(&texture);
        //     upload_image_pbo_to_texture(width, height); // Needs no arguments because texture and PBO are bound
        //     unbind_texture();
        //     unbind_pbo();

        //     glBindFramebuffer(GL_FRAMEBUFFER, framebuffer[i]);
        //     // make sure we clear the framebuffer's content
        //     glViewport(0, 0, width, height);
        //     glClearColor(0.1f, 0.1f, 0.1f, 1.0f);
        //     glClear(GL_COLOR_BUFFER_BIT);

        //     screenShader.use();
        //     glBindVertexArray(quadVAO);
        //     glBindTexture(GL_TEXTURE_2D, texture);
        //     glDrawArrays(GL_TRIANGLES, 0, 6);
        //     glBindTexture(GL_TEXTURE_2D, 0);
        //     glBindVertexArray(0);
        //     glBindFramebuffer(GL_FRAMEBUFFER, 0);
        // }

        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Orange World!");                          
            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
            ImGui::End();
        }

        // {
        //     ImGui::Begin("Streaming");
        //     ImGui::SetNextWindowSize(ImVec2(3208, 1100), 0); // Setting size to 0, 0 forces auto-fit
            
        //     // left
        //     {
        //         ImGui::BeginChild("Cam0", ImVec2(1604, 1100), true);
        //         ImGui::Text("pointer = %p", textureColorbuffer[0]);
        //         ImVec2 avail_size = ImGui::GetContentRegionAvail();
        //         ImGui::Image((void*)(intptr_t)textureColorbuffer[0], avail_size);
        //         ImGui::EndChild();
        //     }
        //     ImGui::SameLine();


        //     {
        //         ImGui::BeginChild("Cam1", ImVec2(1604, 1100), true);
        //         ImGui::Text("pointer = %p", textureColorbuffer[1]);
        //         ImVec2 avail_size = ImGui::GetContentRegionAvail();
        //         ImGui::Image((void*)(intptr_t)textureColorbuffer[1], avail_size);
        //         ImGui::EndChild();
        //     }
            
        //     ImGui::End();
        // }



        // Rendering
        ImGui::Render();
        int display_w, display_h;
        glfwGetFramebufferSize(window, &display_w, &display_h);
        glViewport(0, 0, display_w, display_h);
        glClearColor(clear_color.x * clear_color.w, clear_color.y * clear_color.w, clear_color.z * clear_color.w, clear_color.w);
        glClear(GL_COLOR_BUFFER_BIT);
        ImGui_ImplOpenGL3_RenderDrawData(ImGui::GetDrawData());

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