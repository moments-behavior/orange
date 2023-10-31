#ifndef ORANGE_GUI
#define ORANGE_GUI
#include "gx_helper.h"
#include "camera.h"
#include <math.h>
#include "realtime_tool.h"

struct GL_Texture {
    GLuint texture;
    GLuint pbo;
    cudaGraphicsResource_t cuda_resource;
    unsigned char* cuda_buffer;
    size_t cuda_pbo_storage_buffer_size;
    cudaStream_t streams;
    int num_channels;
};

static void set_camera_properties(CameraEmergent* ecams, CameraParams* cameras_params, int num_cameras)
{
    if (ImGui::TreeNode("Camera Property"))
    {
        static int selected_camera = 0;
        static int slider_gain, slider_exposure, slider_frame_rate, slider_width, slider_height, OffsetX, OffsetY, slider_focus;

        for (int n = 0; n < num_cameras; n++)
        {
            if (ImGui::Selectable(cameras_params[n].camera_serial.c_str(), selected_camera == n))
                selected_camera = n;
                slider_gain = cameras_params[selected_camera].gain;
                slider_focus = cameras_params[selected_camera].focus;
                slider_width = cameras_params[selected_camera].width;
                slider_height = cameras_params[selected_camera].height;
                slider_exposure = cameras_params[selected_camera].exposure;
                slider_frame_rate = cameras_params[selected_camera].frame_rate; 
        }
    

        if(ImGui::SliderInt("Width", &slider_width, cameras_params[selected_camera].width_min, cameras_params[selected_camera].width_max, "%d"))
        {
            slider_width = (slider_width / 16) * 16; // round to even number
            update_width_value(&ecams[selected_camera].camera, slider_width, &cameras_params[selected_camera]);
        }

        if(ImGui::SliderInt("Height", &slider_height, cameras_params[selected_camera].height_min, cameras_params[selected_camera].height_max, "%d"))
        {
            slider_height = (slider_height / 16) * 16; // round to even number
            update_height_value(&ecams[selected_camera].camera, slider_height, &cameras_params[selected_camera]);
        }


        if(ImGui::SliderInt("OffsetX", &OffsetX, cameras_params[selected_camera].offsetx_min, cameras_params[selected_camera].offsetx_max, "%d"))
        {
            // round to 16 
            OffsetX = (OffsetX / 16) * 16; // round to even number
            update_offsetX_value(&ecams[selected_camera].camera, OffsetX, &cameras_params[selected_camera]);
        }


        if(ImGui::SliderInt("OffsetY", &OffsetY, cameras_params[selected_camera].offsety_min, cameras_params[selected_camera].offsety_max, "%d"))
        {
            // round to 16 
            OffsetY = (OffsetY / 16) * 16; // round to even number
            update_offsetY_value(&ecams[selected_camera].camera, OffsetY, &cameras_params[selected_camera]);
        }


        if(ImGui::SliderInt("Gain", &slider_gain, cameras_params[selected_camera].gain_min, cameras_params[selected_camera].gain_max, "%d"))
        {
            update_gain_value(&ecams[selected_camera].camera, slider_gain, &cameras_params[selected_camera]);
        }


        if(ImGui::SliderInt("Focus", &slider_focus, cameras_params[selected_camera].focus_min, cameras_params[selected_camera].focus_max, "%d"))
        {
            update_focus_value(&ecams[selected_camera].camera, slider_focus, &cameras_params[selected_camera]);
        }


        if(ImGui::SliderInt("Exposure", &slider_exposure, cameras_params[selected_camera].exposure_min, cameras_params[selected_camera].exposure_max, "%d"))
        {
            update_exposure_value(&ecams[selected_camera].camera, slider_exposure, &cameras_params[selected_camera]);
        }


        if(ImGui::SliderInt("FrameRate", &slider_frame_rate, cameras_params[selected_camera].frame_rate_min, cameras_params[selected_camera].frame_rate_max, "%d"))
        {
            update_frame_rate_value(&ecams[selected_camera].camera, slider_frame_rate, &cameras_params[selected_camera]);
        }

        ImGui::TreePop();

    }

}


// utility structure for realtime plot
struct ScrollingBuffer {
    int MaxSize;
    int Offset;
    ImVector<ImVec2> Data;
    ScrollingBuffer(int max_size = 2000) {
        MaxSize = max_size;
        Offset  = 0;
        Data.reserve(MaxSize);
    }
    void AddPoint(float x, float y) {
        if (Data.size() < MaxSize)
            Data.push_back(ImVec2(x,y));
        else {
            Data[Offset] = ImVec2(x,y);
            Offset =  (Offset + 1) % MaxSize;
        }
    }
    void Erase() {
        if (Data.size() > 0) {
            Data.shrink(0);
            Offset  = 0;
        }
    }
};

// utility structure for realtime plot
struct RollingBuffer {
    float Span;
    ImVector<ImVec2> Data;
    RollingBuffer() {
        Span = 10.0f;
        Data.reserve(2000);
    }
    void AddPoint(float x, float y) {
        float xmod = fmodf(x, Span);
        if (!Data.empty() && xmod < Data.back().x)
            Data.shrink(0);
        Data.push_back(ImVec2(xmod, y));
    }
};


static void gui_plot_world_coordinates(CameraCalibResults* cvp, int cam_id)
{
    double axis_x_values[4]; double axis_y_values[4]; 
    world_coordinates_projection_points(cvp, axis_x_values, axis_y_values, 50);
    ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, ImVec4(1.0, 1.0, 1.0,1.0));
    ImPlot::SetNextLineStyle(ImVec4(1.0, 1.0, 1.0,1.0), 3.0);
    std::string name = "World Origin";
    
    float one_axis_x[2];
    float one_axis_y[2];

    std::vector<triple_f> node_colors = {
        {1.0f, 1.0f, 1.0f},
        {1.0f, 0.0f, 0.0f},
        {0.0f, 1.0f, 0.0f},
        {0.0f, 0.0f, 1.0f}};
                
    for (u32 edge=0; edge < 3; edge++)
    {
        double xs[2] {axis_x_values[0], axis_x_values[edge+1]};
        double ys[2] {axis_y_values[0], axis_y_values[edge+1]};
        
        ImVec4 my_color; 
        my_color.w = 1.0f; 
        my_color.x = node_colors[edge+1].x;
        my_color.y = node_colors[edge+1].y;
        my_color.z = node_colors[edge+1].z;

        ImPlot::SetNextMarkerStyle(ImPlotMarker_Circle, 6.0, my_color);
        ImPlot::SetNextLineStyle(my_color, 3.0);
        ImPlot::PlotLine(name.c_str(), xs, ys, 2, ImPlotLineFlags_Segments);
    }
    
}

static void draw_aruco_markers(ArucoMarker3d* aruco_marker, int camera_id)
{
    double x[5] = {(double)aruco_marker->proj_corners[camera_id][0].x, 
        (double)aruco_marker->proj_corners[camera_id][1].x, 
        (double)aruco_marker->proj_corners[camera_id][2].x, 
        (double)aruco_marker->proj_corners[camera_id][3].x, 
        (double)aruco_marker->proj_corners[camera_id][0].x};
    
    double y[5] = {(double)2200 - (double)aruco_marker->proj_corners[camera_id][0].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][1].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][2].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][3].y, 
        (double)2200 - (double)aruco_marker->proj_corners[camera_id][0].y};
    
    ImPlot::SetNextLineStyle(ImVec4(1.0, 0.0, 1.0, 1.0), 3.0);
    ImPlot::PlotLine("##", &x[0], &y[0], 5); 
}


#endif