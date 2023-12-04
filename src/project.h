#ifndef ORANGE_PROJECT
#define ORANGE_PROJECT

#include <unistd.h>
#include <iostream>
#include "camera.h"
#include "json.hpp"
using json = nlohmann::json;

std::vector<std::string> string_split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}


std::vector<std::string> string_split_char(char* string_c, std::string delimiter) {
    std::string s = std::string(string_c);
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    return res;
}


void load_camera_json_config_files(std::string file_name, CameraParams* camera_params, int camera_id, int num_cameras) {
    
    std::ifstream f(file_name);
    json camera_config = json::parse(f);

    camera_params->camera_id = camera_id;
    camera_params->num_cameras = num_cameras;
    camera_params->need_reorder = false;

    camera_params->camera_name = camera_config["name"];
    camera_params->width = camera_config["width"];
    camera_params->height = camera_config["height"];
    camera_params->frame_rate = camera_config["frame_rate"];
    camera_params->gain = camera_config["gain"];
    camera_params->exposure = camera_config["exposure"];
    camera_params->pixel_format = camera_config["pixel_format"];
    camera_params->color_temp = camera_config["color_temp"];
    camera_params->gpu_id = camera_config["gpu_id"];
    camera_params->gpu_direct = camera_config["gpu_direct"];
    camera_params->color = camera_config["color"];
    camera_params->focus = camera_config["focus"];
}

// Get current date/time, format is YYYY_MM_DD_HH_mm_ss
const std::string current_date_time() {
    time_t     now = time(0);
    struct tm  tstruct;
    char       buf[80];
    tstruct = *localtime(&now);
    strftime(buf, sizeof(buf), "%Y:%m:%d:%X", &tstruct);
    
    std::string delimiter = ":";

    std::string s(buf);
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr (pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back (token);
    }

    res.push_back (s.substr (pos_start));
    std::string final_string;

    for (int i = 0; i < res.size(); i++) {
        if (i!=0) {
            final_string += "_";
        }
        final_string += res[i];
    }
    return final_string.c_str();
}

void init_galvo_camera_params(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure) 
{
    camera_params->width = 1280;
    camera_params->height = 1280;
    camera_params->frame_rate = 100;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "BayerRG8";
    camera_params->color_temp = "CT_3000K";
    camera_params->camera_id = camera_id;
    camera_params->gpu_id = 1;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
    camera_params->color = true;
}


void init_65MP_camera_params_mono(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int gpu_id, int frame_rate) 
{
    // camera_params->width = 9344;
    // camera_params->height = 7000;
    camera_params->width = 512;
    camera_params->height = 512;
    camera_params->frame_rate = frame_rate;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "Mono8";
    camera_params->gpu_id = gpu_id;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
    camera_params->focus = 4311;
    camera_params->camera_id = camera_id;
    camera_params->color = false;
}


void init_65MP_camera_params_color(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int gpu_id, int frame_rate) 
{
    camera_params->width = 8192; // 9344;
    camera_params->height = 7000; // 7000;
    camera_params->frame_rate = frame_rate;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "BayerGB8";
    camera_params->gpu_id = gpu_id;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
    camera_params->focus = 4419;
    camera_params->camera_id = camera_id;
    camera_params->color = true;
    camera_params->color_temp = "CT_3000K";
}


void init_7MP_camera_params_color(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int gpu_id, int frame_rate) 
{
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
    camera_params->focus = 345;
    camera_params->camera_id = camera_id;
    camera_params->color = true;
}


void init_7MP_camera_params_mono(CameraParams* camera_params, int camera_id, int num_cameras, int gain, int exposure, int gpu_id, int frame_rate) 
{
    camera_params->width = 3208;
    camera_params->height = 2200;
    camera_params->frame_rate = frame_rate;
    camera_params->gain = gain;
    camera_params->exposure = exposure;
    camera_params->pixel_format = "Mono8";
    camera_params->color_temp = "CT_3000K";
    camera_params->gpu_id = gpu_id;
    camera_params->num_cameras = num_cameras;
    camera_params->gpu_direct = false;
    camera_params->need_reorder = false;
    camera_params->focus = 4700;
    camera_params->camera_id = camera_id;
    camera_params->color = false;
}


void update_camera_configs(std::vector<std::string>& camera_config_files, std::string input_folder)
{   
    camera_config_files.clear();
    std::string camera_config_dir = input_folder;
    for (const auto &entry : std::filesystem::directory_iterator(camera_config_dir))
    {
        std::string entry_str = entry.path().string();
        if (entry_str.find(".json") != std::string::npos)
            camera_config_files.push_back(entry_str);
    }
    std::sort(camera_config_files.begin(), camera_config_files.end());
    for (int i=0; i < camera_config_files.size(); i++) {
        std::cout << camera_config_files[i] << std::endl;
    }
}

bool set_camera_params(CameraParams* camera_params, GigEVisionDeviceInfo* device_info, std::vector<std::string>& camera_config_files, int camera_idx, int num_cameras)
{
    // first checkt to see if it is in the config files 
    camera_params->camera_serial.append(device_info->serialNumber);
    camera_params->camera_name = camera_params->camera_serial;

    std::string sub_str = camera_params->camera_serial + ".json";
    auto it = std::find_if(camera_config_files.begin(), camera_config_files.end(), [&](const std::string& str) {return str.find(sub_str) != std::string::npos;});

    if (it == camera_config_files.end())
    {
        if (strcmp(device_info->modelName, "HB-65000GM")==0) {
            int gpu_id = 0;
            init_65MP_camera_params_mono(camera_params, camera_idx, num_cameras, 2000, 1000, gpu_id, 400); //458 
        } else if (strcmp(device_info->modelName, "HB-7000SC")==0) {
            int gpu_id = 0;
            init_7MP_camera_params_color(camera_params, camera_idx, num_cameras, 1500, 2000, gpu_id, 30); // 2000, 3000
        } else if (strcmp(device_info->modelName, "HB-65000GC")==0) {
            int gpu_id = 0;
            init_65MP_camera_params_color(camera_params, camera_idx, num_cameras, 2000, 28000, gpu_id, 10); 
        } else if (strcmp(device_info->modelName, "HB-7000SM")==0) {
            int gpu_id = 0;
            init_7MP_camera_params_mono(camera_params, camera_idx, num_cameras, 1000, 3000, gpu_id, 30); // 2000, 3000
        } else {
            printf("Camera not supported...Exit");
            return false;
        }
    } else {
        auto config_idx = std::distance(camera_config_files.begin(), it);
        std::cout << "Load camera json file: " << camera_config_files[config_idx] << std::endl;
        load_camera_json_config_files(camera_config_files[config_idx], camera_params, camera_idx, num_cameras); 
    }
    return true;
}


#endif