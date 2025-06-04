// src/yolo_worker.h
#ifndef YOLO_WORKER_H
#define YOLO_WORKER_H

#include "threadworker.h"
#include "yolov8_det.h"       // For YOLOv8 class
#include "image_processing.h" // For WORKER_ENTRY, FrameGPU, Debayer
#include "camera.h"           // For CameraParams
#include "video_capture.h"    // For CameraEachSelect
#include "network_base.h"     // For EnetContext and ENetPeer
#include "yolo_payload_generated.h" // For FlatBufferBuilder and YOLO payload messages
#include <chrono> // Required for timing

// Forward declare ENet types if not fully included by network_base.h for this header's needs
// struct ENetHost; // Typically ENetHost is typedef'd from _ENetHost
// struct ENetPeer;

class YOLOv8Worker : public CThreadWorker {
public:
    YOLOv8Worker(const char* name,
                   CameraParams* cam_params,
                   CameraEachSelect* cam_select);
    ~YOLOv8Worker() override;

    // Method to set the ENet target for sending data
    void SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer);

private:
    bool WorkerFunction(void* f) override; // Process one WORKER_ENTRY
    void WorkerReset() override;          // Optional: for resetting worker state

    YOLOv8* yolov8_instance_; // Underscore to denote member
    CameraParams* associated_camera_params_;
    CameraEachSelect* associated_camera_select_;

    // ENet communication members
    EnetContext* enet_host_context_ = nullptr;
    ENetPeer* enet_target_peer_ = nullptr;
    flatbuffers::FlatBufferBuilder* fb_builder_; // For serializing data to send via ENet


    // Internal GPU buffers for YOLO processing
    FrameGPU frame_original_gpu_; // To hold raw frame on GPU
    Debayer debayer_gpu_;         // For debayered/duplicated RGBA output on GPU
    unsigned char* d_rgb_yolo_input_gpu_ = nullptr; // Buffer for RGB data if YOLOv8 needs RGB

    // FPS Counter members
    std::chrono::steady_clock::time_point last_fps_update_time_;
    int frame_counter_;
    double current_fps_;
};

#endif // YOLO_WORKER_H