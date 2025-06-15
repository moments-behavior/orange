// src/yolo_worker.cpp
#include "yolo_worker.h"
#include "kernel.cuh"
#include <cuda_runtime_api.h>
#include <nppi.h> // Include for NPP functions
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "yolo_payload_generated.h"
#include "message_wrapper_generated.h"
#include "pose_shaman.h"
#include "thread.h"
#include "global.h"

YOLOv8Worker::YOLOv8Worker(const char* name,
                           CUcontext cuda_context,
                           CameraParams* cam_params,
                           CameraEachSelect* cam_select,
                           SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      m_cuContext(cuda_context), // <-- Store the context
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
      scaled_width_(0),
      scaled_height_(0),
      d_scaled_mono_buffer_(nullptr),
      shaman_ipc_queue_(nullptr),
      m_recycle_queue(recycle_queue)
{
    std::cout << "YOLOv8Worker instance created for: " << name << std::endl;

    // --- FIX: Push the context to make it active for this thread's setup ---
    ck(cuCtxPushCurrent(m_cuContext));

    // Use a try-catch block to ensure context is popped even on error
    try {
        if (!associated_camera_params_ || !associated_camera_select_) {
            throw std::runtime_error("CameraParams or CameraEachSelect is null.");
        }

        fb_builder_ = new flatbuffers::FlatBufferBuilder(1024 * 4);

        // This is not strictly necessary as the context is already set,
        // but it's good practice for clarity.
        ck(cudaSetDevice(associated_camera_params_->gpu_id));
        std::cout << "YOLOv8Worker for " << name << " set to CUDA device: " << associated_camera_params_->gpu_id << std::endl;

        if (associated_camera_select_->yolo_model == nullptr || strlen(associated_camera_select_->yolo_model) == 0) {
            throw std::runtime_error("YOLO model path is null or empty. Cannot initialize YOLOv8.");
        }

        yolov8_instance_ = new YOLOv8(associated_camera_select_->yolo_model,
                                     associated_camera_params_->width,
                                     associated_camera_params_->height);
        yolov8_instance_->make_pipe(true);

         // *** ADD THIS LINE TO VERIFY THE MODEL'S INPUT SIZE ***
         std::cout << "[YOLOv8 MODEL INFO] " << name << " expects input size: " << yolov8_instance_->inp_w_int << "x" << yolov8_instance_->inp_h_int << std::endl;

        scaled_width_ = yolov8_instance_->inp_w_int;
        scaled_height_ = yolov8_instance_->inp_h_int;

        initalize_gpu_frame(&frame_original_gpu_, associated_camera_params_);
        initialize_gpu_debayer(&debayer_gpu_, associated_camera_params_);
        ck(cudaMalloc((void**)&d_rgb_yolo_input_gpu_, associated_camera_params_->width * associated_camera_params_->height * 3));
        ck(cudaMalloc(&d_scaled_mono_buffer_, scaled_width_ * scaled_height_));
        std::cout << "YOLOv8Worker (" << name << "): Initialized GPU buffers for YOLO processing." << std::endl;

        if (associated_camera_select_->yolo && associated_camera_select_->send_yolo_via_ipc) {
            std::cout << "YOLOv8Worker (" << name << "): Initializing SharedBoxQueue as writer for IPC." << std::endl;
            shaman_ipc_queue_ = new shaman::SharedBoxQueue(true /* is_writer */);
        } else {
            std::cout << "YOLOv8Worker (" << name << "): IPC for YOLO detections is disabled by configuration." << std::endl;
        }

        std::cout << "YOLOv8Worker for " << name << " initialized successfully." << std::endl;

    } catch (const std::runtime_error& e) {
        std::cerr << "YOLOv8Worker Error for " << name << ": " << e.what() << std::endl;

        // --- START: COMPLETE CLEANUP LOGIC ---
        // Clean up any partial allocations before exiting the constructor
        if (fb_builder_) { delete fb_builder_; fb_builder_ = nullptr; }
        if (yolov8_instance_) { delete yolov8_instance_; yolov8_instance_ = nullptr; }
        if (frame_original_gpu_.d_orig) { cudaFree(frame_original_gpu_.d_orig); frame_original_gpu_.d_orig = nullptr; }
        if (debayer_gpu_.d_debayer) { cudaFree(debayer_gpu_.d_debayer); debayer_gpu_.d_debayer = nullptr; }
        if (d_rgb_yolo_input_gpu_) { cudaFree(d_rgb_yolo_input_gpu_); d_rgb_yolo_input_gpu_ = nullptr; }
        if (d_scaled_mono_buffer_) { cudaFree(d_scaled_mono_buffer_); d_scaled_mono_buffer_ = nullptr; }
        
        // IMPORTANT: Pop the context to balance the stack before throwing
        CUcontext popped_context;
        ck(cuCtxPopCurrent(&popped_context));

        // Re-throw the exception to signal that construction failed
        throw; 
        // --- END: COMPLETE CLEANUP LOGIC ---
    }

    // --- FIX: Pop the context when setup is done, ensuring stack balance ---
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
}

YOLOv8Worker::~YOLOv8Worker() {
    std::cout << "YOLOv8Worker instance for " << threadName << " being destroyed." << std::endl;
    // Ensure the CUDA context is set before freeing resources
    ck(cuCtxPushCurrent(m_cuContext)); // Ensure the CUDA context is active for this thread

    if (associated_camera_params_) {
        ck(cudaSetDevice(associated_camera_params_->gpu_id));
    }

    if (frame_original_gpu_.d_orig) cudaFree(frame_original_gpu_.d_orig);
    if (debayer_gpu_.d_debayer) cudaFree(debayer_gpu_.d_debayer);
    if (d_rgb_yolo_input_gpu_) cudaFree(d_rgb_yolo_input_gpu_);
    if (d_scaled_mono_buffer_) cudaFree(d_scaled_mono_buffer_);

    if (yolov8_instance_) delete yolov8_instance_;
    if (shaman_ipc_queue_) delete shaman_ipc_queue_;
    if (fb_builder_) delete fb_builder_;

    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
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
        if (entry) {
             if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                m_recycle_queue.push(entry);
            }
        }
        return false;
    }
    std::lock_guard<std::mutex> lock(g_gpu_camera_mutex);
    // ADD THIS LOG:
    printf("[YOLO_WORKER] ==> START processing entry %p for frame_id %llu\n", (void*)entry, entry->frame_id);
    fflush(stdout);

    // LOG: Received a frame for processing
    std::cout << "[" << threadName << "] ==> "
    << "Received frame_id: " << entry->frame_id
    << " (WORKER_ENTRY: " << entry << ")"
    << ", current ref_count: " << entry->ref_count.load() << std::endl;

    // --- FIX: Push/Pop context around the worker function body ---
    ck(cuCtxPushCurrent(m_cuContext));

    // --- START: DYNAMIC PIPELINE LOGIC ---
    // This section looks correct and doesn't need changes.
    const int model_width = yolov8_instance_->inp_w_int;
    const int camera_width = associated_camera_params_->width;
    
    // --- START: MODIFIED DOWNSAMPLING LOGIC ---
    if (model_width < camera_width) 
    {
        // DOWNSAMPLING PATH (For high-res cameras)
        size_t buffer_size = static_cast<size_t>(entry->width) * static_cast<size_t>(entry->height);
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, entry->d_image, buffer_size, cudaMemcpyDeviceToDevice, yolov8_instance_->stream));
        
        NppiSize oSrcSize = { camera_width, (int)associated_camera_params_->height };
        NppiRect oSrcRect = { 0, 0, camera_width, (int)associated_camera_params_->height };
        NppiSize oDstSize = { yolov8_instance_->inp_w_int, yolov8_instance_->inp_h_int };
        NppiRect oDstRect = { 0, 0, yolov8_instance_->inp_w_int, yolov8_instance_->inp_h_int };
        
        nppiResize_8u_C1R(frame_original_gpu_.d_orig, camera_width, oSrcSize, oSrcRect, d_scaled_mono_buffer_, yolov8_instance_->inp_w_int, oDstSize, oDstRect, NPPI_INTER_LANCZOS);
        
        FrameGPU frame_scaled_gpu;
        frame_scaled_gpu.d_orig = d_scaled_mono_buffer_;
        debayer_gpu_.size = oDstSize; // Use scaled size for debayering
        
        if (associated_camera_params_->color) {
            debayer_frame_gpu(associated_camera_params_, &frame_scaled_gpu, &debayer_gpu_);
        } else {
            duplicate_channel_gpu(associated_camera_params_, &frame_scaled_gpu, &debayer_gpu_);
        }
        
        rgba2rgb_convert(d_rgb_yolo_input_gpu_, debayer_gpu_.d_debayer, oDstSize.width, oDstSize.height, yolov8_instance_->stream);
        yolov8_instance_->preprocess_gpu(d_rgb_yolo_input_gpu_, scaled_width_, scaled_height_);
    } 
    else 
    {
        // FULL-RESOLUTION PATH
        size_t buffer_size = static_cast<size_t>(entry->width) * static_cast<size_t>(entry->height);
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, entry->d_image, buffer_size, cudaMemcpyDeviceToDevice, yolov8_instance_->stream));
        
        debayer_gpu_.size.width = camera_width;
        debayer_gpu_.size.height = associated_camera_params_->height;

        if (associated_camera_params_->color) {
            debayer_frame_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_);
        } else {
            duplicate_channel_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_);
        }
        
        rgba2rgb_convert(d_rgb_yolo_input_gpu_, debayer_gpu_.d_debayer, camera_width, associated_camera_params_->height, yolov8_instance_->stream);
        yolov8_instance_->preprocess_gpu(d_rgb_yolo_input_gpu_, scaled_width_, scaled_height_);
    }

    yolov8_instance_->infer();
    yolov8_instance_->postprocess(entry->detections);
    entry->has_detections = !entry->detections.empty();
    
    // FPS counter and logging
    frame_counter_++;
    auto now = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed = now - last_fps_update_time_;
    if (elapsed.count() >= 1.0) {
        current_fps_ = frame_counter_ / elapsed.count();
        std::cout << threadName << " Inference FPS: " << current_fps_ << " (Queue: " << this->GetCountQueueInSize() << ")" << std::endl;
        frame_counter_ = 0;
        last_fps_update_time_ = now;
    }
    // --- FIX: Add this synchronization call ---
    // This command blocks the CPU thread until all previously issued commands
    // in this specific stream on the GPU have finished. This guarantees that we are
    // done reading from entry->d_image before we allow it to be recycled.
    ck(cudaStreamSynchronize(yolov8_instance_->stream));
    
    if (entry->has_detections) {
        std::cout << threadName << " found " << entry->detections.size() << " objects." << std::endl;
    }

    // --- IPC and ENet logic remains the same ---
    if (associated_camera_select_->send_yolo_via_ipc && shaman_ipc_queue_) {
        std::vector<shaman::Object> shaman_objects = conv_shaman(entry->detections);
        if (!shaman_ipc_queue_->push(shaman_objects)) {
            std::cerr << "YOLOv8Worker (" << threadName << "): Failed to push to IPC queue." << std::endl;
        }
    }

    if (associated_camera_select_->send_yolo_via_enet && enet_host_context_ && enet_target_peer_ &&
        enet_target_peer_->state == ENET_PEER_STATE_CONNECTED) {
        // ENet transmission logic would go here
    }

    // ====================================================================
    // --- START: CORRECTED MEMORY MANAGEMENT LOGIC ---
    //
    // The YOLO worker's job is now done. It should NOT pass the entry to the
    // display worker. Instead, it must handle the reference count.
    //
    // 1. Atomically decrement the reference count.
    // 2. The 'fetch_sub' operation returns the value BEFORE it was decremented.
    // 3. If the value was 1, it means we were the last owner, and the new value is 0.
    //    Therefore, this thread is now responsible for recycling the entry.
    //
    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
         m_recycle_queue.push(entry);
    }
    //
    // --- END: CORRECTED MEMORY MANAGEMENT LOGIC ---
    // ====================================================================

    // Pop the context before the thread sleeps or loops
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));

        // LOG: Finished processing
    std::cout << "[" << this->threadName << "] <== "
              << "Finished frame_id: " << entry->frame_id
              << ". Decrementing ref_count." << std::endl;

    // Decrement ref count and recycle if we're the last owner.
    // ADD THIS LOG before recycling:
    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        printf("[YOLO_WORKER] <== END processing entry %p. Recycling now.\n", (void*)entry);
        fflush(stdout);
        m_recycle_queue.push(entry);
    } else {
        printf("[YOLO_WORKER] <== END processing entry %p. Handing off.\n", (void*)entry);
        fflush(stdout);
    }
    
    return false; 
}

void YOLOv8Worker::WorkerReset() {
    std::cout << "YOLOv8Worker for " << threadName << " reset." << std::endl;
    last_fps_update_time_ = std::chrono::steady_clock::now();
    frame_counter_ = 0;
    current_fps_ = 0.0;
}