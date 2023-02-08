#ifndef ORANGE_PROJECT
#define ORANGE_PROJECT

#include <unistd.h>
#include <iostream>
#include "gx_helper.h"
#include "camera.h"

// Get current date/time, format is YYYY-MM-DD.HH:mm:ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y-%m-%d_%X", &tstruct);
    return buf;
}

void init_galvo_camera_params(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int frame_rate) {
    camera_params->width = 1280;
    camera_params->height = 1280;
    camera_params->frame_rate = frame_rate;
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



void init_25G_camera_params(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int gpu_id, int frame_rate) {
    camera_params->width = 3208;
    camera_params->height = 2200;
    camera_params->frame_rate = frame_rate;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "BayerRG8";
    camera_params->color_temp = "CT_3000K";
    camera_params->gpu_id = gpu_id;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
    camera_params->focus = 320;
    camera_params->camera_id = camera_id;
}


struct GL_Texture {
    GLuint texture;
    GLuint pbo;
    cudaGraphicsResource_t cuda_resource;
    unsigned char* cuda_buffer;
    size_t cuda_pbo_storage_buffer_size;
    unsigned char *display_buffer;
    cudaStream_t streams;
};

#endif