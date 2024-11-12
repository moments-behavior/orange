#include "video_capture.h"
#include "emergent_camera.h"
#include "camera_params.h"
#include "ptp_manager.h"
#include <thread>
#include <chrono>
#include <iostream>

void report_statistics(CameraParams *camera_params, CameraState *camera_state, double time_diff)
{
    std::string print_out;
    print_out += "\n" + camera_params->camera_serial;
    print_out += ", Frame count: " + std::to_string(camera_state->frame_count);
    print_out += ", Frame received: " + std::to_string(camera_state->frames_recd);
    print_out += ", Dropped Frames: " + std::to_string(camera_state->dropped_frames);
    float calc_frame_rate = camera_state->frames_recd / time_diff;
    print_out += ", Calculated Frame Rate: " + std::to_string(calc_frame_rate);
    std::cout << print_out << std::endl;
}
