// src/yolo_worker.cpp - Fixed
#include "yolo_worker.h"
#include "kernel.cuh"
#include <cuda_runtime_api.h>
#include <nppi.h>
#include <npp.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "yolo_payload_generated.h"
#include "message_wrapper_generated.h"
#include "pose_shaman.h"
#include "thread.h"
#include "global.h"
#include "cuda_context_debug.h"

YOLOv8Worker::YOLOv8Worker(const char* name,
                           CUcontext cuda_context,
                           CameraParams* cam_params,
                           CameraEachSelect* cam_select,
                           SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      m_cuContext(cuda_context),
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
    std::cout << "YOLOv8Worker constructor for: " << name << std::endl;
    
    if (!cuda_context) {
        throw std::runtime_error("YOLOv8Worker: Null CUDA context provided");
    }
    
    CUDA_CONTEXT_SCOPE_AT(m_cuContext, "YOLOv8Worker constructor");
    
    if (!ctx_manager.is_valid()) {
        throw std::runtime_error("YOLOv8Worker: Failed to set CUDA context");
    }

    try {
        if (!associated_camera_params_ || !associated_camera_select_) {
            throw std::runtime_error("CameraParams or CameraEachSelect is null.");
        }

        ck(cudaSetDevice(associated_camera_params_->gpu_id));
        std::cout << "YOLOv8Worker set to CUDA device: " << associated_camera_params_->gpu_id << std::endl;

        fb_builder_ = new flatbuffers::FlatBufferBuilder(1024 * 4);

        if (associated_camera_select_->yolo_model == nullptr || strlen(associated_camera_select_->yolo_model) == 0) {
            throw std::runtime_error("YOLO model path is null or empty. Cannot initialize YOLOv8.");
        }

        yolov8_instance_ = new YOLOv8(associated_camera_select_->yolo_model,
                                     associated_camera_params_->width,
                                     associated_camera_params_->height);
        yolov8_instance_->make_pipe(true);

        std::cout << "[YOLOv8 MODEL INFO] " << name << " expects input size: " 
                  << yolov8_instance_->inp_w_int << "x" << yolov8_instance_->inp_h_int << std::endl;

        initalize_gpu_frame(&frame_original_gpu_, associated_camera_params_);
        initialize_gpu_debayer(&debayer_gpu_, associated_camera_params_);
        
        if (associated_camera_select_->yolo && associated_camera_select_->send_yolo_via_ipc) {
            shaman_ipc_queue_ = new shaman::SharedBoxQueue(true /* is_writer */);
        }

        std::cout << "YOLOv8Worker for " << name << " initialized successfully." << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "YOLOv8Worker Error for " << name << ": " << e.what() << std::endl;
        
        if (fb_builder_) { delete fb_builder_; fb_builder_ = nullptr; }
        if (yolov8_instance_) { delete yolov8_instance_; yolov8_instance_ = nullptr; }
        if (frame_original_gpu_.d_orig) { cudaFree(frame_original_gpu_.d_orig); frame_original_gpu_.d_orig = nullptr; }
        if (debayer_gpu_.d_debayer) { cudaFree(debayer_gpu_.d_debayer); debayer_gpu_.d_debayer = nullptr; }
        
        throw; 
    }
}

YOLOv8Worker::~YOLOv8Worker() {
    std::cout << "YOLOv8Worker destructor for " << threadName << std::endl;
    
    CUDA_CONTEXT_SCOPE_AT(m_cuContext, "YOLOv8Worker destructor");
    
    if (ctx_manager.is_valid() && associated_camera_params_) {
        ck(cudaSetDevice(associated_camera_params_->gpu_id));
        
        if (debayer_gpu_.d_debayer) { cudaFree(debayer_gpu_.d_debayer); }
        if (frame_original_gpu_.d_orig) { cudaFree(frame_original_gpu_.d_orig); }
        
        if (yolov8_instance_) { delete yolov8_instance_; }
    } else {
        std::cerr << "Warning: Could not set CUDA context in YOLOv8Worker destructor for " << threadName << std::endl;
    }
    
    if (shaman_ipc_queue_) delete shaman_ipc_queue_;
    if (fb_builder_) delete fb_builder_;
    
    std::cout << "YOLOv8Worker destructor complete for " << threadName << std::endl;
}

// --- FIX: Added the missing function definition ---
void YOLOv8Worker::SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer)
{
    std::cout << "YOLOv8Worker (" << this->threadName
              << "): SetENetTarget called. Host_ctx: " << static_cast<void*>(host_ctx)
              << ". Target_peer: " << static_cast<void*>(target_peer) << std::endl;
    enet_host_context_ = host_ctx;
    enet_target_peer_ = target_peer;
}


bool YOLOv8Worker::WorkerFunction(WORKER_ENTRY* entry) {
    if (!yolov8_instance_ || !entry || !entry->d_image) {
        if (entry && entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_recycle_queue.push(entry);
        }
        return false;
    }

    CUDA_CONTEXT_SCOPE_AT(m_cuContext, "YOLOv8Worker::WorkerFunction");
    
    if (!ctx_manager.is_valid()) {
        if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            m_recycle_queue.push(entry);
        }
        return false;
    }

    try {
        const int camera_width = associated_camera_params_->width;
        const int camera_height = associated_camera_params_->height;

        ck(cudaSetDevice(associated_camera_params_->gpu_id));
        nppSetStream(yolov8_instance_->stream);

        size_t buffer_size = static_cast<size_t>(entry->width) * static_cast<size_t>(entry->height);
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, entry->d_image, buffer_size, 
                          cudaMemcpyDeviceToDevice, yolov8_instance_->stream));

        debayer_gpu_.size.width = camera_width;
        debayer_gpu_.size.height = camera_height;

        if (associated_camera_params_->color) {
            debayer_frame_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_);
        } else {
            duplicate_channel_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_);
        }

        yolov8_instance_->preprocess_gpu(debayer_gpu_.d_debayer, camera_width, camera_height);
        
        yolov8_instance_->infer();
        yolov8_instance_->postprocess(entry->detections);
        entry->has_detections = !entry->detections.empty();

        frame_counter_++;
        auto now = std::chrono::steady_clock::now();
        std::chrono::duration<double> elapsed = now - last_fps_update_time_;
        if (elapsed.count() >= 1.0) {
            current_fps_ = frame_counter_ / elapsed.count();
            std::cout << threadName << " Inference FPS: " << current_fps_ 
                      << " (Queue: " << this->GetCountQueueInSize() << ")" << std::endl;
            frame_counter_ = 0;
            last_fps_update_time_ = now;
        }

        ck(cudaStreamSynchronize(yolov8_instance_->stream));

        if (entry->has_detections) {
            if (associated_camera_select_->send_yolo_via_ipc && shaman_ipc_queue_) {
                std::vector<shaman::Object> shaman_objects = conv_shaman(entry->detections);
                if (!shaman_ipc_queue_->push(shaman_objects)) {
                    std::cerr << "[" << threadName << "] Failed to push to IPC queue." << std::endl;
                }
            }
            if (associated_camera_select_->send_yolo_via_enet && enet_host_context_ && enet_target_peer_ &&
                enet_target_peer_->state == ENET_PEER_STATE_CONNECTED) {
                // ENet transmission logic
            }
        }

    } catch (const std::exception& e) {
        std::cerr << "[" << threadName << "] Exception in WorkerFunction: " << e.what() << std::endl;
        if (yolov8_instance_ && yolov8_instance_->stream) {
            cudaStreamSynchronize(yolov8_instance_->stream);
        }
    }

    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        m_recycle_queue.push(entry);
    }
    
    return false;
}

void YOLOv8Worker::WorkerReset() {
    last_fps_update_time_ = std::chrono::steady_clock::now();
    frame_counter_ = 0;
    current_fps_ = 0.0;
}