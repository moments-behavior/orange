// src/yolo_worker.h
#ifndef YOLO_WORKER_H
#define YOLO_WORKER_H

#include "threadworker.h"
#include "yolov8_det.h"
#include "image_processing.h" // For FrameGPU, Debayer
#include "camera.h"           // For CameraParams
#include "video_capture.h"    // For CameraEachSelect, WORKER_ENTRY (if detections are added here)
#include "network_base.h"     // For EnetContext, ENetPeer
#include "shaman.h"           // For shaman::SharedBoxQueue
#include <chrono>
#include <vector>             // For std::vector (if passing detections)
#include "common.hpp"         // For pose::Object (if passing detections)
#include "opengldisplay.h"
#include <chrono>
#include "opengldisplay.h"
#include <cuda.h>
#include <atomic>


class YOLOv8Worker : public CThreadWorker<WORKER_ENTRY>
{
public:
    YOLOv8Worker(const char* name,
                    CameraParams* cam_params,
                    CameraEachSelect* cam_select,
                    SafeQueue<WORKER_ENTRY*>& recycle_queue);
    ~YOLOv8Worker() override;

    void SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer);
    void SetDisplayWorker(COpenGLDisplay* display_worker) { m_display_worker = display_worker; }
    void DumpNextFrame() { m_dump_next_frame.store(true);}

    // New: Define a structure for passing detection results (or use pose::Object directly)
    // This could also be part of WORKER_ENTRY if you modify it globally
    struct YoloDetectionOutput {
        unsigned long long frame_id;
        unsigned long long timestamp;
        uint64_t timestamp_sys;
        std::vector<pose::Object> detections;
    };

    double get_fps() const {
        return current_fps_.load(std::memory_order_relaxed);
    }

    cudaEvent_t m_inference_completed;

    private:
    bool WorkerFunction(WORKER_ENTRY* f) override;
    void WorkerReset() override;

    std::atomic<bool> m_dump_next_frame;

    YOLOv8* yolov8_instance_;
    CameraParams* associated_camera_params_;
    CameraEachSelect* associated_camera_select_;

    EnetContext* enet_host_context_;
    ENetPeer* enet_target_peer_;
    flatbuffers::FlatBufferBuilder* fb_builder_;

    // Buffers needed for YOLO preprocessing
    FrameGPU frame_original_gpu_;
    Debayer debayer_gpu_;
    unsigned char* d_rgb_yolo_input_gpu_;

    std::chrono::steady_clock::time_point last_fps_update_time_;
    int frame_counter_;
    std::atomic<double> current_fps_;

    // Shared memory IPC
    shaman::SharedBoxQueue* shaman_ipc_queue_;
    COpenGLDisplay* m_display_worker = nullptr;
    SafeQueue<WORKER_ENTRY*>& m_recycle_queue;
};

#endif // YOLO_WORKER_H