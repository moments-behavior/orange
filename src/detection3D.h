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

struct TriangulatePoints {
    int id; 
    std::vector<int> detected_cameras;
    std::vector<std::vector<cv::Point2f>> detected_points;
};

struct CameraEntry {
    void* imagePtr; // source image buffer
    size_t bufferSize; // size of imagePtr in bytes
    int width;
    int height;
    int pixelFormat;
    unsigned long long timestamp;
    unsigned long long frame_id;
};

struct SyncDetection {
    std::vector<int> cam_ids;
    std::condition_variable m_cond;
    std::mutex m_mutex;
    std::vector<bool> frame_unread;
    std::vector<bool> frame_ready;
    std::vector<CameraEntry*> m_frames;
    bool detection_ready;
};

void detection3d_proc(SyncDetection* sync_detection, CameraControl* camera_control, CameraEachSelect *cameras_select, DetectionData* detection_data, int num_cameras);
void detection_proc(SyncDetection* sync_detection, CameraControl* camera_control, CameraParams* camera_params, DetectionData* detection_data, int idx);
#endif