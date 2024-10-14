#ifndef ORANGE_DETECTION
#define ORANGE_DETECTION

#include "image_processing.h"
#include <cuda_runtime_api.h>
#include "realtime_tool.h"
#include "camera.h"
#include "aruco_nano.h"
#include "kernel.cuh"
#include <atomic>
#include <thread>

struct SyncDetection {
    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::vector<bool> frame_unread;
    std::vector<bool> frame_ready;
    bool detection_ready;
};

void detection3d_proc(SyncDetection* sync_detection, CameraControl* camera_control);
void detection_proc(SyncDetection* sync_detection, CameraControl* camera_control, int idx);
#endif