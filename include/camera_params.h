#pragma once
#include <string>

namespace evt {

struct CameraParams {
    int width{0};
    int height{0};
    int gain{0};
    int exposure{0};
    int frame_rate{0};
    int focus{0};
    int gpu_id{0};
    int camera_id{0};
    bool gpu_direct{false};
    bool color{false};
    bool need_reorder{false};
    std::string pixel_format;
    std::string color_temp;
    std::string camera_name;
    std::string camera_serial;
    unsigned int num_cameras{0};

    // Add max/min ranges for parameters
    unsigned int width_max{0};
    unsigned int width_min{0};
    unsigned int width_inc{0};
    unsigned int height_max{0};
    unsigned int height_min{0};
    unsigned int height_inc{0};
    unsigned int gain_max{0};
    unsigned int gain_min{0};
    unsigned int gain_inc{0};
    unsigned int exposure_max{0};
    unsigned int exposure_min{0};
    unsigned int exposure_inc{0};
    unsigned int frame_rate_max{0};
    unsigned int frame_rate_min{0};
    unsigned int frame_rate_inc{0};
    unsigned int focus_max{0};
    unsigned int focus_min{0};
    unsigned int focus_inc{0};
    unsigned int offsetx_max{0};
    unsigned int offsetx_min{0};
    unsigned int offsetx_inc{0};
    unsigned int offsety_max{0};
    unsigned int offsety_min{0};
    unsigned int offsety_inc{0};
    unsigned int offsetx{0};
    unsigned int offsety{0};
    unsigned int iris{0};
    unsigned int iris_max{0};
    unsigned int iris_min{0};
    unsigned int iris_inc{0};
    int sens_temp_max{0};
    int sens_temp_min{0};
    int sens_temp{0};
};

} // namespace evt