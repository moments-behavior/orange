#pragma once
#include <string>

struct CameraParams {
    int width;
    int height;
    int gain;
    int exposure;
    int frame_rate;
    int focus;
    int gpu_id;
    int camera_id;
    bool gpu_direct;
    bool color;
    bool need_reorder;
    std::string pixel_format;
    std::string color_temp;
    std::string camera_name;
    std::string camera_serial;
    unsigned int num_cameras;

    // Add max/min ranges for parameters
    unsigned int width_max;
    unsigned int width_min;
    unsigned int width_inc;
    unsigned int height_max;
    unsigned int height_min;
    unsigned int height_inc;
    unsigned int gain_max;
    unsigned int gain_min;
    unsigned int gain_inc;
    unsigned int exposure_max;
    unsigned int exposure_min;
    unsigned int exposure_inc;
    unsigned int frame_rate_max;
    unsigned int frame_rate_min;
    unsigned int frame_rate_inc;
    unsigned int focus_max;
    unsigned int focus_min;
    unsigned int focus_inc;
    unsigned int offsetx_max;
    unsigned int offsetx_min;
    unsigned int offsetx_inc;
    unsigned int offsety_max;
    unsigned int offsety_min;
    unsigned int offsety_inc;
    unsigned int offsetx;
    unsigned int offsety;
    unsigned int iris;
    unsigned int iris_max;
    unsigned int iris_min;
    unsigned int iris_inc;              // Iris increment value
    int sens_temp_max;
    int sens_temp_min;
    int sens_temp;

};
