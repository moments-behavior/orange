#ifndef YOLO_WORKER_H
#define YOLO_WORKER_H

#include "threadworker.h"
#include "yolov8_det.h"
#include "image_processing.h"
#include "camera.h"
#include "video_capture.h"
#include "network_base.h"
#include "yolo_payload_generated.h"
#include <chrono>
#include <nppi_geometry_transforms.h> // For NppiSize, NppiRect, nppiResize_8u_C4R

class YOLOv8Worker : public CThreadWorker {
public:
    YOLOv8Worker(const char* name,
                   CameraParams* cam_params,
                   CameraEachSelect* cam_select,
                   unsigned char* display_texture_buffer); // Added display_texture_buffer
    ~YOLOv8Worker() override;

    void SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer);

private:
    bool WorkerFunction(void* f) override;
    void WorkerReset() override;

    YOLOv8* yolov8_instance_;
    CameraParams* associated_camera_params_;
    CameraEachSelect* associated_camera_select_;

    EnetContext* enet_host_context_ = nullptr;
    ENetPeer* enet_target_peer_ = nullptr;
    flatbuffers::FlatBufferBuilder* fb_builder_;

    FrameGPU frame_original_gpu_;
    Debayer debayer_gpu_;
    unsigned char* d_rgb_yolo_input_gpu_ = nullptr;

    // For display output
    unsigned char* display_texture_buffer_ = nullptr;
    unsigned char* d_display_resize_buffer_ = nullptr;
    NppiSize output_display_size_;
    NppiRect input_roi_for_display_resize_;
    NppiRect output_roi_for_display_resize_;

    // For drawing detections
    float *d_points_for_drawing_ = nullptr;
    unsigned int *d_skeleton_for_drawing_ = nullptr;

    // FPS Counter members
    std::chrono::steady_clock::time_point last_fps_update_time_;
    int frame_counter_;
    double current_fps_;
};

#endif // YOLO_WORKER_H