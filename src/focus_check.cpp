// focus_check.cpp
// Standalone camera focus checker for orange.
//
// Opens each Emergent camera, applies config (including the Focus motor value),
// grabs a test frame, and reports Laplacian variance sharpness per camera.
// Also detects when the configured focus value was silently out-of-range and
// never applied to the hardware.
//
// Exits 0 if all cameras pass, 1 if any fail.
//
// Usage (run from orange source directory):
//   ./build/focus_check
//   ./build/focus_check --config /path/to/config --threshold 150 --save-frames

#include "camera.h"
#include "json.hpp"
#include <filesystem>
#include <fstream>
#include <iostream>
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

using json = nlohmann::json;

static const double DEFAULT_THRESHOLD = 100.0;
static const int    WARMUP_FRAMES     = 10;
static const int    FRAME_BUFFER_SIZE = 4;

// ---------------------------------------------------------------------------
// Config loading (no dependency on project.cpp / ENet / flatbuffers)
// ---------------------------------------------------------------------------
static void load_focus_config(const std::string &filename, CameraParams *p) {
    std::ifstream f(filename);
    if (!f.is_open()) return;
    try {
        json j   = json::parse(f);
        p->width        = j.value("width",        1920u);
        p->height       = j.value("height",       1080u);
        p->frame_rate   = j.value("frame_rate",   30u);
        p->gain         = j.value("gain",         0u);
        p->exposure     = j.value("exposure",     5000u);
        p->pixel_format = j.value("pixel_format", std::string("BayerRG8"));
        p->color_temp   = j.value("color_temp",   std::string("CT_3000K"));
        p->gpu_id       = j.value("gpu_id",       0);
        p->gpu_direct   = j.value("gpu_direct",   false);
        p->color        = j.value("color",        true);
        p->focus        = j.value("focus",        0u);
        p->iris         = j.value("iris",         0u);
        p->camera_name  = j.value("name",         std::string(""));
    } catch (const std::exception &e) {
        fprintf(stderr, "  Warning: failed to parse %s: %s\n",
                filename.c_str(), e.what());
    }
}

// ---------------------------------------------------------------------------
// Sharpness metric: Laplacian variance
// ---------------------------------------------------------------------------
static double laplacian_variance(const cv::Mat &gray) {
    cv::Mat lap;
    cv::Laplacian(gray, lap, CV_64F);
    cv::Scalar mean, stddev;
    cv::meanStdDev(lap, mean, stddev);
    return stddev[0] * stddev[0];
}

// Debayer raw Bayer 8-bit frame to grayscale (CPU)
static cv::Mat bayer_to_gray(const void *data, int w, int h,
                              const std::string &fmt) {
    cv::Mat raw(h, w, CV_8UC1, const_cast<void *>(data));
    cv::Mat bgr;
    if      (fmt == "BayerRG8") cv::cvtColor(raw, bgr, cv::COLOR_BayerRG2BGR);
    else if (fmt == "BayerGB8") cv::cvtColor(raw, bgr, cv::COLOR_BayerGB2BGR);
    else if (fmt == "BayerGR8") cv::cvtColor(raw, bgr, cv::COLOR_BayerGR2BGR);
    else if (fmt == "BayerBG8") cv::cvtColor(raw, bgr, cv::COLOR_BayerBG2BGR);
    else {
        // Mono8 or unknown
        cv::Mat gray;
        cv::cvtColor(raw, gray, cv::COLOR_BGR2GRAY);
        return gray;
    }
    cv::Mat gray;
    cv::cvtColor(bgr, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

// ---------------------------------------------------------------------------
// Per-camera result
// ---------------------------------------------------------------------------
struct CameraResult {
    std::string serial;
    std::string config_file;
    unsigned int focus_configured = 0;
    unsigned int focus_min = 0;
    unsigned int focus_max = 0;
    bool focus_was_applied = false;  // false = silently skipped (out of range)
    double sharpness = 0.0;
    bool passed = false;
    std::string error;
};

// ---------------------------------------------------------------------------
// main
// ---------------------------------------------------------------------------
int main(int argc, char *argv[]) {
    std::string  config_dir  = "./config";
    double       threshold   = DEFAULT_THRESHOLD;
    bool         save_frames = false;
    std::string  save_dir    = "/tmp/focus_check_frames";

    for (int i = 1; i < argc; i++) {
        std::string arg(argv[i]);
        if      (arg == "--config"      && i+1 < argc) config_dir  = argv[++i];
        else if (arg == "--threshold"   && i+1 < argc) threshold   = std::stod(argv[++i]);
        else if (arg == "--save-frames")                save_frames = true;
        else if (arg == "--save-dir"    && i+1 < argc) save_dir    = argv[++i];
        else if (arg == "--help") {
            printf("Usage: focus_check [--config DIR] [--threshold N] "
                   "[--save-frames [--save-dir DIR]]\n");
            return 0;
        }
    }

    // Scan cameras
    const int max_cameras = 20;
    GigEVisionDeviceInfo unsorted_info[max_cameras];
    GigEVisionDeviceInfo device_info[max_cameras];
    int cam_count = scan_cameras(max_cameras, unsorted_info);
    sort_cameras_ip(unsorted_info, device_info, cam_count);

    if (cam_count == 0) {
        fprintf(stderr, "No cameras found.\n");
        return 1;
    }
    printf("Found %d camera(s). Config dir: %s  Threshold: %.0f\n\n",
           cam_count, config_dir.c_str(), threshold);

    // Collect config files once
    std::vector<std::string> config_files;
    if (std::filesystem::exists(config_dir)) {
        for (auto &entry : std::filesystem::directory_iterator(config_dir)) {
            if (entry.path().extension() == ".json")
                config_files.push_back(entry.path().string());
        }
    }

    if (save_frames)
        std::filesystem::create_directories(save_dir);

    std::vector<CameraResult> results;

    for (int i = 0; i < cam_count; i++) {
        CameraResult result;
        result.serial = std::string(device_info[i].serialNumber);
        printf("--- Camera %d  serial=%s ---\n", i, result.serial.c_str());

        // Match config file by serial number
        CameraParams params;
        params.camera_id     = i;
        params.num_cameras   = cam_count;
        params.need_reorder  = false;
        params.gop           = 1;
        params.offsetx       = 0;
        params.offsety       = 0;
        params.camera_serial = result.serial;
        bool found_config    = false;

        for (auto &cf : config_files) {
            if (std::filesystem::path(cf).stem().string() == result.serial) {
                load_focus_config(cf, &params);
                params.camera_serial = result.serial;
                result.config_file   = cf;
                found_config         = true;
                printf("  Config: %s  focus=%u  iris=%u\n",
                       cf.c_str(), params.focus, params.iris);
                break;
            }
        }

        if (!found_config) {
            printf("  WARNING: no config found for serial %s in %s — "
                   "using defaults (focus=0)\n",
                   result.serial.c_str(), config_dir.c_str());
            // Minimal defaults so we can still grab a frame
            params.width        = 1920;
            params.height       = 1080;
            params.frame_rate   = 10;
            params.gain         = 0;
            params.exposure     = 5000;
            params.pixel_format = "BayerRG8";
            params.color_temp   = "CT_3000K";
            params.gpu_id       = 0;
            params.gpu_direct   = false;
            params.color        = true;
            params.focus        = 0;
            params.iris         = 0;
            params.camera_name  = result.serial;
            result.config_file  = "(none)";
        }

        result.focus_configured = params.focus;

        // Open camera with params — this calls update_focus_value internally
        // which will log a warning if the focus value is out of range
        Emergent::CEmergentCamera camera;
        open_camera_with_params(&camera, &device_info[i], &params);

        result.focus_min = params.focus_min;
        result.focus_max = params.focus_max;
        result.focus_was_applied = (params.focus_min == 0 && params.focus_max == 0)
            ? true  // camera doesn't expose focus range — assume applied
            : (result.focus_configured >= result.focus_min &&
               result.focus_configured <= result.focus_max);

        if (!result.focus_was_applied) {
            printf("  WARNING: focus=%u is outside the camera's range [%u, %u] "
                   "— focus was NOT applied, camera uses factory default!\n",
                   result.focus_configured, result.focus_min, result.focus_max);
        }

        // Open stream and allocate buffers
        camera_open_stream(&camera, &params);

        Emergent::CEmergentFrame evt_frames[FRAME_BUFFER_SIZE];
        allocate_frame_buffer(&camera, evt_frames, &params, FRAME_BUFFER_SIZE);

        EVT_CameraExecuteCommand(&camera, "AcquisitionStart");

        // Discard warmup frames to let the sensor and lens motor settle
        Emergent::CEmergentFrame frame_recv;
        for (int f = 0; f < WARMUP_FRAMES; f++) {
            if (EVT_CameraGetFrame(&camera, &frame_recv, 2000) == EVT_SUCCESS) {
                EVT_CameraQueueFrame(&camera, &frame_recv);
            }
        }

        // Grab analysis frame
        int ret = EVT_CameraGetFrame(&camera, &frame_recv, 2000);
        if (ret == EVT_SUCCESS) {
            cv::Mat gray = bayer_to_gray(frame_recv.imagePtr,
                                          frame_recv.size_x,
                                          frame_recv.size_y,
                                          params.pixel_format);
            result.sharpness = laplacian_variance(gray);
            result.passed    = (result.sharpness >= threshold);

            if (save_frames) {
                std::string fname = save_dir + "/" + result.serial + "_focus.png";
                cv::Mat raw(frame_recv.size_y, frame_recv.size_x,
                             CV_8UC1, frame_recv.imagePtr);
                cv::Mat out;
                if      (params.pixel_format == "BayerRG8")
                    cv::cvtColor(raw, out, cv::COLOR_BayerRG2BGR);
                else if (params.pixel_format == "BayerGB8")
                    cv::cvtColor(raw, out, cv::COLOR_BayerGB2BGR);
                else
                    out = raw;
                cv::imwrite(fname, out);
                printf("  Saved: %s\n", fname.c_str());
            }

            EVT_CameraQueueFrame(&camera, &frame_recv);
        } else {
            result.error = "frame grab timeout";
            printf("  ERROR: %s\n", result.error.c_str());
        }

        // Cleanup
        EVT_CameraExecuteCommand(&camera, "AcquisitionStop");
        destroy_frame_buffer(&camera, evt_frames, FRAME_BUFFER_SIZE, &params);
        EVT_CameraCloseStream(&camera);
        close_camera(&camera, &params);
        printf("\n");

        results.push_back(result);
    }

    // Summary table
    const int W = 95;
    printf("%-14s  %-28s  %-10s  %-14s  %-10s  %s\n",
           "Serial", "Config", "Focus", "Range", "Sharpness", "Status");
    printf("%s\n", std::string(W, '-').c_str());

    bool all_passed = true;
    for (auto &r : results) {
        std::string cfg_name =
            std::filesystem::path(r.config_file).filename().string();
        std::string focus_str =
            std::to_string(r.focus_configured) +
            (r.focus_was_applied ? "" : "(!)");
        std::string range_str =
            "[" + std::to_string(r.focus_min) + "," +
            std::to_string(r.focus_max) + "]";
        std::string status;
        if (!r.error.empty()) {
            status     = "ERROR: " + r.error;
            all_passed = false;
        } else if (!r.focus_was_applied) {
            status     = "FOCUS NOT SET(!)";
            all_passed = false;
        } else if (!r.passed) {
            status     = "BLURRY <-- WARNING";
            all_passed = false;
        } else {
            status = "OK";
        }
        printf("%-14s  %-28s  %-10s  %-14s  %-10.1f  %s\n",
               r.serial.c_str(), cfg_name.c_str(), focus_str.c_str(),
               range_str.c_str(), r.sharpness, status.c_str());
    }
    printf("%s\n", std::string(W, '-').c_str());

    if (!all_passed) {
        printf("\n[FAIL] One or more cameras need attention.\n");
        printf("  (!) = focus value was outside camera's min/max range and was\n"
               "        silently skipped in camera.cpp:update_focus_value().\n"
               "        Fix 'focus' in config/<serial>.json to a value inside\n"
               "        the range shown above, then rerun this check.\n");
        return 1;
    }

    printf("\n[PASS] All cameras appear to be in focus. Safe to record.\n");
    return 0;
}
