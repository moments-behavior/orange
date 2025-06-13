// src/yolo_worker.cpp
#include "yolo_worker.h"
#include "kernel.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "yolo_payload_generated.h"
#include "message_wrapper_generated.h"
#include "pose_shaman.h"
#include "thread.h"

YOLOv8Worker::YOLOv8Worker(const char* name,
                           CameraParams* cam_params,
                           CameraEachSelect* cam_select,
                           SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      yolov8_instance_(nullptr),
      associated_camera_params_(cam_params),
      associated_camera_select_(cam_select),
      enet_host_context_(nullptr),
      enet_target_peer_(nullptr),
      fb_builder_(nullptr),
      d_rgb_yolo_input_gpu_(nullptr),
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_counter_(0),
      current_fps_(0.0),
      shaman_ipc_queue_(nullptr),
      m_recycle_queue(recycle_queue)
{
    std::cout << "YOLOv8Worker instance created for: " << name << std::endl;

    if (!associated_camera_params_ || !associated_camera_select_) {
        std::cerr << "YOLOv8Worker Error: CameraParams or CameraEachSelect is null for " << name << "." << std::endl;
        return;
    }

    fb_builder_ = new flatbuffers::FlatBufferBuilder(1024 * 4);

    ck(cudaSetDevice(associated_camera_params_->gpu_id));
    std::cout << "YOLOv8Worker for " << name << " set to CUDA device: " << associated_camera_params_->gpu_id << std::endl;

    if (associated_camera_select_->yolo_model == nullptr || strlen(associated_camera_select_->yolo_model) == 0) {
         std::cerr << "YOLOv8Worker Error: YOLO model path is null or empty for " << name << ". Cannot initialize YOLOv8." << std::endl;
         delete fb_builder_;
         fb_builder_ = nullptr;
         return;
    }

    std::cout << "YOLOv8Worker for " << name << ": Initializing YOLOv8 with model: "
              << associated_camera_select_->yolo_model
              << " for resolution " << associated_camera_params_->width
              << "x" << associated_camera_params_->height << std::endl;

    yolov8_instance_ = new YOLOv8(associated_camera_select_->yolo_model,
                                 associated_camera_params_->width,
                                 associated_camera_params_->height);
    yolov8_instance_->make_pipe(true);

    // Allocate only the buffers needed for inference.
    initalize_gpu_frame(&frame_original_gpu_, associated_camera_params_);
    initialize_gpu_debayer(&debayer_gpu_, associated_camera_params_);
    ck(cudaMalloc((void**)&d_rgb_yolo_input_gpu_, associated_camera_params_->width * associated_camera_params_->height * 3));

    // Initialize Shaman IPC queue if configured
    if (associated_camera_select_->yolo && associated_camera_select_->send_yolo_via_ipc) {
        try {
            std::cout << "YOLOv8Worker (" << name << "): Initializing SharedBoxQueue as writer for IPC." << std::endl;
            shaman_ipc_queue_ = new shaman::SharedBoxQueue(true /* is_writer */);
            std::cout << "YOLOv8Worker (" << name << "): SharedBoxQueue constructor returned." << std::endl;
        } catch (const std::runtime_error& e) {
            std::cerr << "YOLOv8Worker (" << name << ") Error initializing SharedBoxQueue for IPC: " << e.what() << std::endl;
            shaman_ipc_queue_ = nullptr;
        }
    } else {
        std::cout << "YOLOv8Worker (" << name << "): IPC for YOLO detections is disabled by configuration." << std::endl;
    }

    std::cout << "YOLOv8Worker for " << name << " initialized successfully." << std::endl;
}

YOLOv8Worker::~YOLOv8Worker() {
    std::cout << "YOLOv8Worker instance for " << threadName << " being destroyed." << std::endl;

    if (associated_camera_params_) {
        // Set the correct CUDA context for this worker's cleanup
        cudaError_t err = cudaSetDevice(associated_camera_params_->gpu_id);
        if (err != cudaSuccess) {
            std::cerr << "YOLOv8Worker (" << threadName << ") destructor: Failed to set CUDA device. Error: " << cudaGetErrorString(err) << std::endl;
        }
    }

    // Free only the buffers this worker is responsible for
    if (frame_original_gpu_.d_orig) cudaFree(frame_original_gpu_.d_orig);
    if (debayer_gpu_.d_debayer) cudaFree(debayer_gpu_.d_debayer);
    if (d_rgb_yolo_input_gpu_) cudaFree(d_rgb_yolo_input_gpu_);

    // Delete the YOLOv8 instance, which handles its own internal cleanup
    if (yolov8_instance_) {
        delete yolov8_instance_;
    }

    if (shaman_ipc_queue_) {
        delete shaman_ipc_queue_;
    }

    if (fb_builder_) {
        delete fb_builder_;
    }

    std::cout << "YOLOv8Worker for " << threadName << " resources cleaned up." << std::endl;
}

void YOLOv8Worker::SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer) {
    std::cout << "YOLOv8Worker (" << this->threadName
              << "): SetENetTarget called. Host_ctx: " << static_cast<void*>(host_ctx)
              << ". Target_peer: " << static_cast<void*>(target_peer) << std::endl;
    enet_host_context_ = host_ctx;
    enet_target_peer_ = target_peer;
}

bool YOLOv8Worker::WorkerFunction(WORKER_ENTRY* entry) {
    if (!yolov8_instance_ || !entry) {
        return false;
    }

    ck(cudaSetDevice(associated_camera_params_->gpu_id));

    // --- Frame acquisition and YOLO processing ---
    size_t buffer_size = static_cast<size_t>(entry->width) * static_cast<size_t>(entry->height);
    ck(cudaMemcpyAsync(frame_original_gpu_.d_orig,
                       entry->d_image,
                       buffer_size,
                       cudaMemcpyDeviceToDevice,
                       yolov8_instance_->stream));

    if (associated_camera_params_->color) {
        debayer_frame_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_);
    }

    rgba2rgb_convert(d_rgb_yolo_input_gpu_, debayer_gpu_.d_debayer,
                     associated_camera_params_->width, associated_camera_params_->height, yolov8_instance_->stream);

    yolov8_instance_->preprocess_gpu(d_rgb_yolo_input_gpu_);
    yolov8_instance_->infer();

    yolov8_instance_->postprocess(entry->detections);
    entry->has_detections = !entry->detections.empty();
    
    frame_counter_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - last_fps_update_time_;
    if (elapsed.count() >= 1.0) {
        current_fps_ = frame_counter_ / elapsed.count();
        std::cout << threadName << " Inference FPS: " << current_fps_ << std::endl;
        frame_counter_ = 0;
        last_fps_update_time_ = now;
    }
    
    // Always try to send, even if detections are empty.
    if (associated_camera_select_->send_yolo_via_ipc && shaman_ipc_queue_) {
        //std::cout << "[YOLO Worker] Attempting to push " << entry->detections.size() << " detections to IPC queue..." << std::endl;
        std::vector<shaman::Object> shaman_objects = conv_shaman(entry->detections);
        if (!shaman_ipc_queue_->push(shaman_objects)) {
            // This provides a more specific error message.
            std::cerr << "YOLOv8Worker (" << threadName << "): Failed to push to IPC queue (is reader running and queue not full?)." << std::endl;
        }
    }

    if (associated_camera_select_->send_yolo_via_enet && enet_host_context_ && enet_target_peer_ &&
        enet_target_peer_->state == ENET_PEER_STATE_CONNECTED) {
        // ENet transmission logic would go here
    }
    
    // Instead of putting the result on our own output queue, push it to the display worker.
    if (m_display_worker) {
        m_display_worker->PutObjectToQueueIn(entry);
    } else {
        m_recycle_queue.push(entry); // Recycle the entry if no display worker is set
    }

    // Return 'false' to tell the CThreadWorker base class that we have manually
    // handled the buffer and it should NOT place it on our own output queue.
    return false;
}

void YOLOv8Worker::WorkerReset() {
    std::cout << "YOLOv8Worker for " << threadName << " reset." << std::endl;
    last_fps_update_time_ = std::chrono::steady_clock::now();
    frame_counter_ = 0;
    current_fps_ = 0.0;
}