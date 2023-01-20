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


    // bool select_cameras = false; 
    int max_cameras = 16;
    GigEVisionDeviceInfo device_info[max_cameras];
    // GigEVisionDeviceInfo ordered_device_info[max_cameras];

    // int num_cameras = 4;

    int cam_count;
    cam_count = scan_cameras(max_cameras, device_info);
    
    
    // if (cam_count < num_cameras) 
    // {
    //     printf("Missing cameras...Exit\n");
    //     return 0;
    // }


    // // esc to exit 
    int key_num;
    int* key_num_ptr = &key_num;  
    // bool* record_video = new bool(false);
    // bool* capture_pause = new bool(false);

    // string folder_string = current_date_time();
    // string folder_name = "/home/red/Videos/" + folder_string;
    
    // // Creating a directory to save recorded video;
    // if (mkdir(folder_name.c_str(), 0777) == -1)
    // {
    //     std::cerr << "Error :  " << std::strerror(errno) << std::endl;
    //     return 0;
    // }
    // else
    //     std::cout << "Recorded video saves to : " << folder_name << std::endl;


    // int select_camera[] = {0, 1, 2, 3}; 
    // int encode_gpu[] = {1, 1, 1, 1};


    // int buffer_size {500};
    // PTPParams* ptp_params = new PTPParams{0, 0};


    // for(int i = 0; i < num_cameras; i++)
    // {
    //     int camera_id = select_camera[i];
    //     int gpu_id = encode_gpu[i];
    //     init_25G_camera_params(&cameras_params[i], camera_id, num_cameras, 2000, 3000, gpu_id);   
    // }


    // // init camera resources 
    // Emergent::CEmergentCamera camera[num_cameras];
    // Emergent::CEmergentFrame evt_frame[num_cameras][buffer_size]; 
    // Emergent::CEmergentFrame frame_recv[num_cameras];

    // string encoder_setup = "-preset p1 -fps " + to_string(cameras_params[0].frame_rate);
    // const char *encoder_str = encoder_setup.c_str();
    // std::vector<thread> camera_threads;

    // for(int i = 0; i < num_cameras; i++)
    // {
    //     open_camera_with_params(&camera[i], &ordered_device_info[select_camera[i]], &cameras_params[i]); 
    //     // open_camera_with_params(&camera[i], &device_info[i], camera_params[i]); 

    //     // sync 
    //     ptp_camera_sync(&camera[i]);
    //     EVT_CameraOpenStream(&camera[i]);
    //     allocate_frame_buffer(&camera[i], evt_frame[i], &cameras_params[i], buffer_size);
    //     set_frame_buffer(&frame_recv[i], &cameras_params[i]);
    // }


    // GLuint texture[num_cameras];
    // GLuint pbo[num_cameras];
    // cudaGraphicsResource_t cuda_resource[num_cameras] {0};
    // unsigned char* cuda_buffer[num_cameras];
    // size_t cuda_pbo_storage_buffer_size[num_cameras];
    // unsigned char *display_buffer[num_cameras];
    // cudaStream_t streams[num_cameras];

    // for(int i = 0; i < num_cameras; i++)
    // {
    //     cudaStreamCreate(&streams[i]);
    //     int size_pic = cameras_params[i].width * cameras_params[i].height * 4 * sizeof(unsigned char);

    //     create_texture(&texture[i]);
    //     create_pbo(&pbo[i], cameras_params[i].width, cameras_params[i].height);
    //     bind_pbo(&pbo[i]);

    //     register_pbo_to_cuda(&pbo[i], &cuda_resource[i]);
    //     unbind_texture();
    //     unbind_pbo();
    //     cudaMalloc((void **)&display_buffer[i], size_pic);
    // }

    
    
    // for(int i = 0; i < num_cameras; i++)
    // {
    //     camera_threads.push_back(std::thread(&aquire_frames_gpu_encode, &camera[i], &frame_recv[i], &cameras_params[i], encoder_str, key_num_ptr, ptp_params, folder_name, display_buffer[i], record_video, capture_pause, &cuda_buffer[i], &cuda_resource[i], &cuda_pbo_storage_buffer_size[i], &pbo[i], &texture[i]));
    // }

    ImGui::FileBrowser file_dialog(ImGuiFileBrowserFlags_SelectDirectory | ImGuiFileBrowserFlags_CreateNewDir);
    file_dialog.SetTitle("My files");
    std::string input_folder = file_dialog.GetPwd();
    
    bool check[cam_count] = {};
    bool streaming = false;

    CameraParams* cameras_params;

    // init camera resources 
    // Emergent::CEmergentCamera camera[num_cameras];
    // Emergent::CEmergentFrame evt_frame[num_cameras][buffer_size]; 
    // Emergent::CEmergentFrame frame_recv[num_cameras];

    // string encoder_setup = "-preset p1 -fps " + to_string(cameras_params[0].frame_rate);
    // const char *encoder_str = encoder_setup.c_str();
    CameraEmergent* ecams;
    std::vector<thread> camera_threads;
    GL_Texture* tex;
    u32 num_cameras = 0;

    while (!glfwWindowShouldClose(window->render_target))
    {
        create_new_frame();

        if(streaming) {

            for (int i = 0; i < num_cameras; i++) {
                // Transfer to PBO then OpenGL texture
                // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
                map_cuda_resource(&tex[i].cuda_resource);
                cuda_pointer_from_resource(&tex[i].cuda_buffer, &tex[i].cuda_pbo_storage_buffer_size, &tex[i].cuda_resource);
                cudaMemcpy2DAsync(tex[i].cuda_buffer, cameras_params[i].width*4, tex[i].display_buffer, cameras_params[i].width*4, cameras_params[i].width*4, cameras_params[i].height, cudaMemcpyDeviceToDevice, tex[i].streams);
                unmap_cuda_resource(&tex[i].cuda_resource);
                // CUDA-GL INTEROP ENDS HERE ---------------------------------------------------------------------------
                bind_pbo(&tex[i].pbo);
                bind_texture(&tex[i].texture);
                upload_image_pbo_to_texture(cameras_params[i].width, cameras_params[i].height); // Needs no arguments because texture and PBO are bound
                unbind_texture();
                unbind_pbo();
            }
            cudaDeviceSynchronize();
        } 


        if (ImGui::Begin("Orange Streaming",  NULL, ImGuiWindowFlags_MenuBar))
        {

            ImGui::Text("Application average %.3f ms/frame (%.1f FPS)", 1000.0f / ImGui::GetIO().Framerate, ImGui::GetIO().Framerate);


            if (ImGui::Button("Save to")){
                file_dialog.Open();
            }
            ImGui::SameLine();
            ImGui::Text(input_folder.c_str());
               
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

            num_cameras = 0;
            for (int i = 0; i < cam_count; i++){
                if(check[i]) { num_cameras++;}
            }
            std::cout << num_cameras << std::endl;

            if (ImGui::Button("Streaming")){
                // start streaming selected cameras 
                cameras_params = new CameraParams[num_cameras];

                for(int i = 0; i < num_cameras; i++)
                {
                    cameras_params[i].camera_name.append(device_info[i].serialNumber);
                    init_25G_camera_params(&cameras_params[i], i, num_cameras, 2000, 3000, 0);   
                }

                std::cout << "here?" << std::endl;

                ecams = new CameraEmergent[num_cameras];
                std::cout << "here?" << std::endl;
                for(int i = 0; i < num_cameras; i++)
                {
                    open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]); 
                    // open_camera_with_params(&camera[i], &device_info[i], camera_params[i]); 

                    // sync 
                    // ptp_camera_sync(&camera[i]);
                    EVT_CameraOpenStream(&ecams[i].camera);
                    allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, &cameras_params[i], 100);
                    set_frame_buffer(&ecams[i].frame_recv, &cameras_params[i]);
                }

                tex = new GL_Texture[num_cameras];
                for(int i = 0; i < num_cameras; i++)
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

    
                for(int i = 0; i < num_cameras; i++)
                {
                    camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], key_num_ptr, tex[i].display_buffer));
                }
                streaming = true;

            }


            ImGui::Separator();
            ImGui::Spacing();

        //     //if (ImGui::TreeNode("Camera Property"))
        //     {
        //         static int selected_camera = 0;
        //         static int slider_gain, slider_exposure, slider_frame_rate, slider_width, slider_height, OffsetX, OffsetY, slider_focus;

        //         for (int n = 0; n < num_cameras; n++)
        //         {
        //             char buf[32];
        //             sprintf(buf, "Camera %d", select_camera[n]);
        //             if (ImGui::Selectable(buf, selected_camera == n))
        //                 selected_camera = n;
        //                 slider_gain = cameras_params[selected_camera].gain;
        //                 slider_focus = cameras_params[selected_camera].focus;
        //                 slider_width = cameras_params[selected_camera].width;
        //                 slider_height = cameras_params[selected_camera].height;
        //                 slider_exposure = cameras_params[selected_camera].exposure;
        //                 slider_frame_rate = cameras_params[selected_camera].frame_rate; 
        //         }
            

        //         // if(ImGui::SliderInt("Width", &slider_width, cameras_params[selected_camera].width_min, cameras_params[0].width_max, "%d"))
        //         // {
        //         //     slider_width = (slider_width / 16) * 16; // round to even number

        //         //     *capture_pause = true;
        //         //     update_width_value(&camera[selected_camera], slider_width, &cameras_params[selected_camera]);
        //         //     // set_frame_buffer(&frame_recv[selected_camera], &cameras_params[selected_camera]);
        //         //     // for(int frame_count=0;frame_count<buffer_size;frame_count++)
        //         //     // {
        //         //     //     set_frame_buffer(&evt_frame[selected_camera][frame_count], &cameras_params[selected_camera]);
        //         //     // }
        //         //     *capture_pause = false;
                    
        //         // }

        //         // if(ImGui::SliderInt("Height", &slider_height, cameras_params[selected_camera].height_min, cameras_params[0].height_max, "%d"))
        //         // {
        //         //     slider_height = (slider_height / 16) * 16; // round to even number
        //         //     update_height_value(&camera[selected_camera], slider_height, &cameras_params[selected_camera]);
        //         // }


        //         if(ImGui::SliderInt("OffsetX", &OffsetX, cameras_params[selected_camera].offsetx_min, cameras_params[selected_camera].offsetx_max, "%d"))
        //         {
        //             // round to 16 
        //             OffsetX = (OffsetX / 16) * 16; // round to even number
        //             // update_offsetX_value(&camera[selected_camera], OffsetX, &cameras_params[selected_camera]);
        //             EVT_CameraSetUInt32Param(camera, "OffsetX", OffsetX);

        //         }


        //         if(ImGui::SliderInt("OffsetY", &OffsetY, cameras_params[selected_camera].offsety_min, cameras_params[selected_camera].offsety_max, "%d"))
        //         {
        //             // round to 16 
        //             OffsetY = (OffsetY / 16) * 16; // round to even number
        //             // update_offsetY_value(&camera[selected_camera], OffsetY, &cameras_params[selected_camera]);
        //             EVT_CameraSetUInt32Param(camera, "OffsetY", OffsetY);

        //         }


        //         if(ImGui::SliderInt("Gain", &slider_gain, cameras_params[selected_camera].gain_min, cameras_params[selected_camera].gain_max, "%d"))
        //         {
        //             update_gain_value(&camera[selected_camera], slider_gain, &cameras_params[selected_camera]);
        //         }


        //         if(ImGui::SliderInt("Focus", &slider_focus, cameras_params[selected_camera].focus_min, cameras_params[selected_camera].focus_max, "%d"))
        //         {
        //             update_focus_value(&camera[selected_camera], slider_focus, &cameras_params[selected_camera]);
        //         }


        //         if(ImGui::SliderInt("Exposure", &slider_exposure, cameras_params[selected_camera].exposure_min, cameras_params[selected_camera].exposure_max, "%d"))
        //         {
        //             update_exposure_value(&camera[selected_camera], slider_exposure, &cameras_params[selected_camera]);
        //         }


        //         if(ImGui::SliderInt("FrameRate", &slider_frame_rate, cameras_params[selected_camera].frame_rate_min, cameras_params[selected_camera].frame_rate_max, "%d"))
        //         {
        //             update_frame_rate_value(&camera[selected_camera], slider_frame_rate, &cameras_params[selected_camera]);
        //         }
        //         //ImGui::TreePop();

        //     }

        //     ImGui::Separator();
        //     ImGui::Spacing();
            
        //     // if (ImGui::Button("Streaming")){}
        //     // ImGui::SameLine();

        //     if (*record_video)
        //         ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0.5f, 0, 0, 1.0f });
        //     else
        //         ImGui::PushStyleColor(ImGuiCol_Button, ImVec4{ 0, 0.5f, 0, 1.0f });

        //     if (ImGui::Button(*record_video ? ICON_FK_PAUSE : ICON_FK_PLAY)){
        //         (*record_video) = !(*record_video);
        //     }
        //     ImGui::PopStyleColor(1);


            ImGui::End();        

        }

        file_dialog.Display();

        
        if (file_dialog.HasSelected())
        {
            input_folder = file_dialog.GetSelected().string();
            std::cout << input_folder << std::endl;
            file_dialog.ClearSelected();
        }

        if (streaming) {
            for (int i = 0; i < num_cameras; i++) {
                string window_name = cameras_params[i].camera_name; // "Cam" + std::to_string(cameras_params[i].camera_id);            
                ImGui::Begin(window_name.c_str());
                ImVec2 avail_size = ImGui::GetContentRegionAvail();

                //ImGui::Image((void*)(intptr_t)texture[i], avail_size);
                if (ImPlot::BeginPlot("##no_plot_name", avail_size)){
                    ImPlot::PlotImage("##no_image_name", (void*)(intptr_t)tex[i].texture, ImVec2(0,0), ImVec2(cameras_params[i].width, cameras_params[i].height));
                    ImPlot::EndPlot();
                }
                ImGui::End();        
            }
        }

       render_a_frame(window);
    }

    // Cleanup
    gx_cleanup(window);
 
    // exit 
    key_num = 27;


    // wait for threads to join
    for (auto& t : camera_threads)
            t.join();
    

    for(int i = 0; i < num_cameras; i++)
    {
        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, 100);
        EVT_CameraOpenStream(&ecams[i].camera);
        close_camera(&ecams[i].camera);
    }

    // std::cout << folder_name << std::endl;
    cudaDeviceReset();
    return 0;
}