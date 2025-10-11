#include "utils.h"
#include "NvEncoder/NvCodecUtils.h"
#include "json.hpp"
#include <algorithm>
#include <fstream>
#include <iostream>

#if defined(_WIN32)
#include <shlobj.h>
#include <windows.h>
#pragma comment(lib, "shell32.lib")
#else
#include <cstdlib>
#include <pwd.h>
#include <sys/types.h>
#include <unistd.h>
#endif

simplelogger::Logger *logger =
    simplelogger::LoggerFactory::CreateConsoleLogger();

std::string get_home_directory() {
#if defined(_WIN32)
    PWSTR path = nullptr;
    std::string home;
    if (SUCCEEDED(SHGetKnownFolderPath(FOLDERID_Profile, 0, NULL, &path))) {
        char buffer[MAX_PATH];
        WideCharToMultiByte(CP_UTF8, 0, path, -1, buffer, MAX_PATH, NULL, NULL);
        home = buffer;
        CoTaskMemFree(path);
    }
    return home;
#else
    const char *home = std::getenv("HOME");
    if (home != nullptr && std::strcmp(home, "/root") != 0)
        return std::string(home);

    // Check for SUDO_USER — the original username before sudo
    const char *sudo_user = std::getenv("SUDO_USER");
    if (sudo_user != nullptr) {
        struct passwd *pw = getpwnam(sudo_user);
        if (pw != nullptr)
            return std::string(pw->pw_dir);
    }

    // fallback: use real UID (not effective UID)
    struct passwd *pw = getpwuid(getuid());
    if (pw != nullptr)
        return std::string(pw->pw_dir);

    return {};
#endif
}

void create_required_folders(const std::string &base_dir,
                             const std::vector<std::string> &app_folders) {
    for (const auto &folder : app_folders) {
        std::filesystem::path path = std::filesystem::path(base_dir) / folder;

        try {
            if (!std::filesystem::exists(path)) {
                if (std::filesystem::create_directories(path)) {
                    std::cout << "Created folder: " << path << std::endl;
                }
            }
        } catch (const std::filesystem::filesystem_error &e) {
            std::cerr << "Error creating " << path << ": " << e.what()
                      << std::endl;
        }
    }
}

void prepare_application_folders(std::string &orange_root_dir,
                                 std::string &recording_root_dir,
                                 std::string &encoder_codec) {

    std::string home_dir = get_home_directory();

    // check for config.json
    std::filesystem::path config_path =
        std::filesystem::path(home_dir) / ".config/orange/config.json";
    if (std::filesystem::exists(config_path)) {
        try {
            std::ifstream f(config_path);
            nlohmann::json j;
            f >> j;

            if (j.contains("recording_folder") &&
                j["recording_folder"].is_string()) {
                recording_root_dir = j["recording_folder"].get<std::string>();
            }

            if (j.contains("codec") && j["codec"].is_string()) {
                encoder_codec = j["codec"].get<std::string>();
            }
        } catch (const std::exception &e) {
            std::cerr << "Failed to read/parse config.json: " << e.what()
                      << std::endl;
        }
    }

    if (orange_root_dir.empty()) {
        orange_root_dir = home_dir + "/orange_data";
    }
    std::vector<std::string> app_folders = {
        "calib_yaml", "detect", "config/local", "config/network", "pictures"};
    create_required_folders(orange_root_dir, app_folders);

    if (recording_root_dir.empty()) {
        recording_root_dir = orange_root_dir;
    }

    std::vector<std::string> recording_folders = {"exp/unsorted",
                                                  "exp/calibration"};
    create_required_folders(recording_root_dir, recording_folders);
}

std::vector<std::string> string_split(std::string s, std::string delimiter) {
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

std::vector<std::string> string_split_char(char *string_c,
                                           std::string delimiter) {
    std::string s = std::string(string_c);
    size_t pos_start = 0, pos_end, delim_len = delimiter.length();
    std::string token;
    std::vector<std::string> res;

    while ((pos_end = s.find(delimiter, pos_start)) != std::string::npos) {
        token = s.substr(pos_start, pos_end - pos_start);
        pos_start = pos_end + delim_len;
        res.push_back(token);
    }

    res.push_back(s.substr(pos_start));
    return res;
}

std::string get_current_time_milliseconds() {
    // Get the current time
    auto now = std::chrono::system_clock::now();
    auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(
                  now.time_since_epoch()) %
              1000;
    auto time = std::chrono::system_clock::to_time_t(now);
    auto tm = *std::localtime(&time);

    // Format the time
    std::ostringstream oss;
    oss << std::put_time(&tm, "%H_%M_%S_") << std::setfill('0') << std::setw(3)
        << ms.count();

    return oss.str();
}

std::string get_current_date() {
    // Get the current time
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    // Convert to local time
    std::tm local_time = *std::localtime(&time_t_now);

    // Format the date as year_month_day
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y_%m_%d");

    return oss.str();
}

// Get current date/time, format is YYYY_MM_DD_HH_mm_ss
std::string get_current_date_time() {
    // Get the current time
    auto now = std::chrono::system_clock::now();
    auto time_t_now = std::chrono::system_clock::to_time_t(now);

    // Convert to local time
    std::tm local_time = *std::localtime(&time_t_now);

    // Format the date and time as YYYY_MM_DD_HH_mm_ss
    std::ostringstream oss;
    oss << std::put_time(&local_time, "%Y_%m_%d_%H_%M_%S");

    return oss.str();
}

std::string format_elapsed_time(std::chrono::seconds elapsed_seconds) {
    int hours = static_cast<int>(elapsed_seconds.count() / 3600);
    int minutes = static_cast<int>((elapsed_seconds.count() % 3600) / 60);
    int seconds = static_cast<int>(elapsed_seconds.count() % 60);

    std::ostringstream oss;
    oss << std::setfill('0') << std::setw(2) << hours << ":"
        << std::setfill('0') << std::setw(2) << minutes << ":"
        << std::setfill('0') << std::setw(2) << seconds;

    return oss.str();
}

void init_galvo_camera_params(CameraParams *camera_params, int camera_id,
                              int num_cameras, int gain, int exposure) {
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
    camera_params->iris = 0;
}

void init_65MP_camera_params_mono(CameraParams *camera_params, int camera_id,
                                  int num_cameras, int gain, int exposure,
                                  int gpu_id, int frame_rate) {
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
    camera_params->iris = 0;
}

void init_65MP_camera_params_color(CameraParams *camera_params, int camera_id,
                                   int num_cameras, int gain, int exposure,
                                   int gpu_id, int frame_rate) {
    camera_params->width = 512;  // 8192; // 9344;
    camera_params->height = 512; // 7000; // 7000;
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
    camera_params->iris = 0;
}

void init_7MP_camera_params_color(CameraParams *camera_params, int camera_id,
                                  int num_cameras, int gain, int exposure,
                                  int gpu_id, int frame_rate) {
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
    camera_params->iris = 0;
}

void init_7MP_camera_params_mono(CameraParams *camera_params, int camera_id,
                                 int num_cameras, int gain, int exposure,
                                 int gpu_id, int frame_rate) {
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
    camera_params->iris = 0;
}

bool make_folder(std::string folder) {
    namespace fs = std::filesystem;
    fs::path p(folder); // construct a path from the string
    std::error_code ec;

    if (fs::create_directories(p, ec)) // created (including parents)
        return true;

    if (!ec && fs::is_directory(p)) // already exists as a directory
        return true;

    std::cerr << "Error creating folder '" << p.string()
              << "': " << ec.message() << '\n';
    return false;
}

void update_camera_configs(std::vector<std::string> &camera_config_files,
                           std::string input_folder) {
    camera_config_files.clear();
    std::string camera_config_dir = input_folder;
    for (const auto &entry :
         std::filesystem::directory_iterator(camera_config_dir)) {
        std::string entry_str = entry.path().string();
        if (entry_str.find(".json") != std::string::npos)
            camera_config_files.push_back(entry_str);
    }
    std::sort(camera_config_files.begin(), camera_config_files.end());
    // for (int i=0; i < camera_config_files.size(); i++) {
    //     std::cout << camera_config_files[i] << std::endl;
    // }
}

void select_cameras_have_configs(std::vector<std::string> &camera_config_files,
                                 GigEVisionDeviceInfo *device_info,
                                 std::vector<bool> &check, int cam_count) {
    for (int i = 0; i < cam_count; i++) {
        std::string camera_serial = device_info[i].serialNumber;
        std::string sub_str = camera_serial + ".json";
        auto it =
            std::find_if(camera_config_files.begin(), camera_config_files.end(),
                         [&](const std::string &str) {
                             return str.find(sub_str) != std::string::npos;
                         });
        if (it != camera_config_files.end()) {
            check[i] = true;
        } else {
            check[i] = false;
        }
    }
}

void allocate_camera_frame_buffers(CameraEmergent *ecams,
                                   CameraParams *cameras_params,
                                   int evt_buffer_size, int num_cameras) {
    for (int i = 0; i < num_cameras; i++) {
        camera_open_stream(&ecams[i].camera, &cameras_params[i]);
        ecams[i].evt_frame = new Emergent::CEmergentFrame[evt_buffer_size];
        allocate_frame_buffer(&ecams[i].camera, ecams[i].evt_frame,
                              &cameras_params[i], evt_buffer_size);
        if (cameras_params[i].need_reorder && cameras_params[i].gpu_direct) {
            allocate_frame_reorder_buffer(
                &ecams[i].camera, &ecams[i].frame_reorder, &cameras_params[i]);
        }
    }
}
