#include "video_capture_gpu.h"
#include <iostream>
#include "camera.h"
#include <thread>

#include "imgui.h"
#include "imgui_impl_glfw.h"
#include "imgui_impl_opengl3.h"
#include <GLFW/glfw3.h> // Will drag system OpenGL headers


// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    return buf;
}


static void glfw_error_callback(int error, const char* description)
{
    fprintf(stderr, "Glfw Error %d: %s\n", error, description);
}



int main(int argc, char **args) 
{
    // **************** camera resources ***************************************** 
    short max_cameras {10};
    GigEVisionDeviceInfo device_info[max_cameras];
    GigEVisionDeviceInfo ordered_device_info[max_cameras];

    int num_cameras = 1;

    int cam_count;
    cam_count = check_cameras(max_cameras, device_info, ordered_device_info);
    if (cam_count < num_cameras) 
    {
        printf("Missing cameras...Exit\n");
        return 0;
    }

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
    int camera_gpus[] = {0, 0, 0, 0, 1, 1, 1};
    
    
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
        camera_threads.push_back(std::thread(&aquire_frames_gpu_encode, &camera[i], &frame_recv[i], camera_params[i], encoder_str, key_num_ptr, ptp_params, folder_name));
    }


    // *********************GUI*****************************************************
    // Setup window
    glfwSetErrorCallback(glfw_error_callback);
    if (!glfwInit())
        return 1;

    // GL 3.0 + GLSL 130
    const char* glsl_version = "#version 130";
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 0);


    // Create window with graphics context
    GLFWwindow* window = glfwCreateWindow(1280, 720, "Dear ImGui GLFW+OpenGL3 example", NULL, NULL);
    if (window == NULL)
        return 1;

    glfwMakeContextCurrent(window);
    glfwSwapInterval(1); // Enable vsync
    IMGUI_CHECKVERSION();
    ImGui::CreateContext();
    ImGuiIO& io = ImGui::GetIO(); (void)io;

    ImGui::StyleColorsDark();

    ImGui_ImplGlfw_InitForOpenGL(window, true);
    ImGui_ImplOpenGL3_Init(glsl_version);

    ImVec4 clear_color = ImVec4(0.45f, 0.55f, 0.60f, 1.00f);

    while (!glfwWindowShouldClose(window))
    {
        glfwPollEvents();
        ImGui_ImplOpenGL3_NewFrame();
        ImGui_ImplGlfw_NewFrame();
        ImGui::NewFrame();

        {
            ImGui::Begin("Orange World!");                          
            if (ImGui::Button("Stop Recording"))
                key_num = 27;

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);
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

        glfwSwapBuffers(window);
    }

    // Cleanup
    ImGui_ImplOpenGL3_Shutdown();
    ImGui_ImplGlfw_Shutdown();
    ImGui::DestroyContext();

    glfwDestroyWindow(window);
    glfwTerminate();
 

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