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
    file_dialog.SetPwd("/home/jinyao/exp");
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

    while (!glfwWindowShouldClose(window->render_target))
    {
        create_new_frame();

        if (camera_control->streaming)
        {

            for (int i = 0; i < num_cameras; i++)
            {
                // Transfer to PBO then OpenGL texture
                // CUDA-GL INTEROP STARTS HERE -------------------------------------------------------------------------
                map_cuda_resource(&tex[i].cuda_resource);
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
            cudaDeviceSynchronize();
        }

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

            num_cameras = 0;
            for (int i = 0; i < cam_count; i++)
            {
                if (check[i])
                {
                    num_cameras++;
                }
            }

            if (ImGui::Button(camera_control->streaming ? "Stop streaming" : "Start streaming"))
            {
                cameras_params = new CameraParams[num_cameras];

                for (int i = 0; i < num_cameras; i++)
                {
                    cameras_params[i].camera_name.append(device_info[i].serialNumber);
                    init_25G_camera_params(&cameras_params[i], i, num_cameras, 2000, 3000, 0);
                }

                ecams = new CameraEmergent[num_cameras];
                for (int i = 0; i < num_cameras; i++)
                {
                    open_camera_with_params(&ecams[i].camera, &device_info[cameras_params[i].camera_id], &cameras_params[i]);
                    // sync
                    // ptp_camera_sync(&camera[i]);
                    EVT_CameraOpenStream(&ecams[i].camera);
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
                    camera_threads.push_back(std::thread(&aquire_frames_gpu, &ecams[i], &cameras_params[i], camera_control, tex[i].display_buffer));
                }
                camera_control->streaming = true;
            }

            ImGui::Separator();
            ImGui::Spacing();

            if (camera_control->streaming)
            {
                set_camera_properties(ecams, cameras_params, num_cameras);
            }

            ImGui::Separator();
            ImGui::Spacing();

            if (ImGui::Button("Save to"))
            {
                file_dialog.Open();
            }
            ImGui::SameLine();
            ImGui::Text(input_folder.c_str());

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
                    string folder_name = file_dialog.GetSelected().string() + "/" + folder_string;

                    if (mkdir(folder_name.c_str(), 0777) == -1)
                    {
                        std::cerr << "Error :  " << std::strerror(errno) << std::endl;
                        return 0;
                    }
                    else
                    {
                        std::cout << "Recorded video saves to : " << folder_name << std::endl;
                    }
                    string encoder_setup = "-preset p1 -fps " + to_string(cameras_params[0].frame_rate);
                    camera_control->streaming = false;
                    for (auto &t : camera_threads)
                        t.join();

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.pop_back();
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        ptp_camera_sync(&ecams[i].camera);
                    }

                    for (int i = 0; i < num_cameras; i++)
                    {
                        camera_threads.push_back(std::thread(&sync_aquire_frames_gpu_encode, &ecams[i], &cameras_params[i], camera_control, tex[i].display_buffer, encoder_setup, folder_name, ptp_params));
                    }
                    camera_control->streaming = true;                    
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

        if (camera_control->streaming)
        {
            for (int i = 0; i < num_cameras; i++)
            {
                string window_name = cameras_params[i].camera_name;
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

        render_a_frame(window);
    }

    // Cleanup
    gx_cleanup(window);

    camera_control->streaming = false;
    for (auto &t : camera_threads)
        t.join();

    for (int i = 0; i < num_cameras; i++)
    {
        destroy_frame_buffer(&ecams[i].camera, ecams[i].evt_frame, 100);
        EVT_CameraCloseStream(&ecams[i].camera);
        close_camera(&ecams[i].camera);
    }

    // std::cout << folder_name << std::endl;
    cudaDeviceReset();
    return 0;
}