#pragma once
#include <string>
#include "json.hpp"
#ifndef  EMERGENT_SDK
#include <gigevisiondeviceinfo.h>  // Add this include
#endif
#include "NvEncoder/Logger.h"  // Add this at the top

namespace evt {

struct CameraParams {
    // Add device info storage
    GigEVisionDeviceInfo device_info{};  // Add this field
    
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
    unsigned int width_min{8};
    unsigned int width_inc{0};
    unsigned int height_max{0};
    unsigned int height_min{16};
    unsigned int height_inc{0};
    unsigned int gain_max{0};
    unsigned int gain_min{8};
    unsigned int gain_inc{0};
    unsigned int exposure_max{0};
    unsigned int exposure_min{1};
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

    static CameraParams from_json(const nlohmann::json& j) {
        CameraParams params;
        
        // Basic parameters
        params.width = j.value("width", 0);
        params.height = j.value("height", 0);
        params.gain = j.value("gain", 0);
        params.exposure = j.value("exposure", 0);  // This should be picking up the value 10
        params.frame_rate = j.value("frame_rate", 0);
        params.focus = j.value("focus", 0);
        params.gpu_id = j.value("gpu_id", 0);
        params.gpu_direct = j.value("gpu_direct", false);
        params.color = j.value("color", false);
        
        // Most importantly - make sure we store the serial!
        params.camera_serial = j.value("device_serial_number", "");
        
        // Debug logging to verify values are read correctly
        LOG(INFO) << "Loaded camera config values:"
                << "\n  Exposure: " << params.exposure
                << "\n  Frame Rate: " << params.frame_rate
                << "\n  Gain: " << params.gain;
        
        return params;
    }
};

} // namespace evt