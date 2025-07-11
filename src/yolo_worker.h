// src/yolo_worker.h
#ifndef YOLO_WORKER_H
#define YOLO_WORKER_H

#include "threadworker.h"
#include "yolov8_det.h"
#include "image_processing.h" // For FrameGPU, Debayer
#include "camera.h"           // For CameraParams
#include "video_capture.h"    // For CameraEachSelect, WORKER_ENTRY
#include "network_base.h"     // For EnetContext, ENetPeer
#include "shaman.h"           // For shaman::SharedBoxQueue
#include <chrono>
#include <vector>
#include <chrono>
#include <cuda.h>
#include <atomic>
#include "common.hpp"         // For pose::Object

class COpenGLDisplay;
class CropAndEncodeWorker;

class YOLOv8Worker : public CThreadWorker<WORKER_ENTRY>
{
public:
    YOLOv8Worker(const char* name,
                    CameraParams* cam_params,
                    CameraEachSelect* cam_select,
                    SafeQueue<WORKER_ENTRY*>& recycle_queue);
    ~YOLOv8Worker() override;

    void SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer);
    void SetDisplayWorker(COpenGLDisplay* display_worker);
    void SetCropAndEncodeWorker(CropAndEncodeWorker* crop_worker);
    void DumpNextFrame() { m_dump_next_frame.store(true);}

    CameraParams* GetCameraParams() const { return associated_camera_params_; }

    struct YoloDetectionOutput {
        unsigned long long frame_id;
        unsigned long long timestamp;
        uint64_t timestamp_sys;
        std::vector<pose::Object> detections;
    };

    double get_fps() const {
        return current_fps_.load(std::memory_order_relaxed);
    }

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

    FrameGPU frame_original_gpu_;
    Debayer debayer_gpu_;
    unsigned char* d_rgb_yolo_input_gpu_;

    std::chrono::steady_clock::time_point last_fps_update_time_;
    int frame_counter_;
    std::atomic<double> current_fps_;

    shaman::SharedBoxQueue* shaman_ipc_queue_;
    COpenGLDisplay* m_display_worker = nullptr;
    CropAndEncodeWorker* m_crop_worker = nullptr; // New pointer to the crop worker
    SafeQueue<WORKER_ENTRY*>& m_recycle_queue;
};

#endif // YOLO_WORKER_H