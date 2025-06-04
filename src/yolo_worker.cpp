// src/yolo_worker.cpp
#include "yolo_worker.h"
#include "kernel.cuh" 
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono> // Ensure chrono is included

// Include the generated FlatBuffer header
#include "yolo_payload_generated.h" // Assuming it's in the include path

// network_base.h should provide ENet definitions
// fetch_generated.h provides FlatBufferBuilder (though flatbuffers/flatbuffers.h is more direct)

YOLOv8Worker::YOLOv8Worker(const char* name,
                           CameraParams* cam_params,
                           CameraEachSelect* cam_select)
    : CThreadWorker(name),
      yolov8_instance_(nullptr),
      associated_camera_params_(cam_params),
      associated_camera_select_(cam_select),
      enet_host_context_(nullptr),
      enet_target_peer_(nullptr),
      d_rgb_yolo_input_gpu_(nullptr),
      // Initialize FPS counter members
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_counter_(0),
      current_fps_(0.0) {

    std::cout << "YOLOv8Worker instance created for: " << name << std::endl;

    if (!associated_camera_params_ || !associated_camera_select_) {
        std::cerr << "YOLOv8Worker Error: CameraParams or CameraEachSelect is null for " << name << "." << std::endl;
        return;
    }

    fb_builder_ = new flatbuffers::FlatBufferBuilder(1024 * 4); // Increased initial size to 4KB

    ck(cudaSetDevice(associated_camera_params_->gpu_id)); //
    std::cout << "YOLOv8Worker for " << name << " set to CUDA device: " << associated_camera_params_->gpu_id << std::endl; //

    if (associated_camera_select_->yolo_model == nullptr || strlen(associated_camera_select_->yolo_model) == 0) { //
         std::cerr << "YOLOv8Worker Error: YOLO model path is null or empty for " << name << ". Cannot initialize YOLOv8." << std::endl; //
         delete fb_builder_; //
         fb_builder_ = nullptr; //
         return;
    }

    std::cout << "YOLOv8Worker for " << name << ": Initializing YOLOv8 with model: " //
              << associated_camera_select_->yolo_model //
              << " for resolution " << associated_camera_params_->width //
              << "x" << associated_camera_params_->height << std::endl; //

    yolov8_instance_ = new YOLOv8(associated_camera_select_->yolo_model, //
                                 associated_camera_params_->width, //
                                 associated_camera_params_->height); //
    yolov8_instance_->make_pipe(true); //

    initalize_gpu_frame(&frame_original_gpu_, associated_camera_params_); //
    initialize_gpu_debayer(&debayer_gpu_, associated_camera_params_); //
    ck(cudaMalloc((void**)&d_rgb_yolo_input_gpu_, associated_camera_params_->width * associated_camera_params_->height * 3)); //

    std::cout << "YOLOv8Worker for " << name << " initialized successfully." << std::endl; //
}

YOLOv8Worker::~YOLOv8Worker() {
    std::cout << "YOLOv8Worker instance for " << threadName << " being destroyed." << std::endl; //
    delete yolov8_instance_; //
    yolov8_instance_ = nullptr; //

    delete fb_builder_; //
    fb_builder_ = nullptr; //

    if (associated_camera_params_) { //
        cudaSetDevice(associated_camera_params_->gpu_id); //
    }
    if (frame_original_gpu_.d_orig) { //
        cudaFree(frame_original_gpu_.d_orig); //
        frame_original_gpu_.d_orig = nullptr; //
    }
    if (debayer_gpu_.d_debayer) { //
        cudaFree(debayer_gpu_.d_debayer); //
        debayer_gpu_.d_debayer = nullptr; //
    }
    if (d_rgb_yolo_input_gpu_) { //
        cudaFree(d_rgb_yolo_input_gpu_); //
        d_rgb_yolo_input_gpu_ = nullptr; //
    }
    std::cout << "YOLOv8Worker for " << threadName << " resources cleaned up." << std::endl; //
}

void YOLOv8Worker::SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer) {
    enet_host_context_ = host_ctx; //
    enet_target_peer_ = target_peer; //
    if (target_peer) { //
        std::cout << "YOLOv8Worker " << threadName << " ENet target set." << std::endl; //
    } else {
        std::cout << "YOLOv8Worker " << threadName << " ENet target cleared." << std::endl; //
    }
}

// Change return type from void to bool
bool YOLOv8Worker::WorkerFunction(void* f) { //
    if (!yolov8_instance_ || !fb_builder_) { //
        PutObjectToQueueOut(f); //
        return false; 
    }

    WORKER_ENTRY* original_entry = static_cast<WORKER_ENTRY*>(f); //
    ck(cudaSetDevice(associated_camera_params_->gpu_id)); //

    ck(cudaMemcpy2DAsync(frame_original_gpu_.d_orig, //
                    associated_camera_params_->width, //
                    original_entry->imagePtr,        //
                    associated_camera_params_->width, //
                    associated_camera_params_->width, //
                    associated_camera_params_->height, //
                    cudaMemcpyHostToDevice,          //
                    yolov8_instance_->stream));      //

    if (associated_camera_params_->color) { //
        debayer_frame_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_); //
    } else {
        duplicate_channel_gpu(associated_camera_params_, &frame_original_gpu_, &debayer_gpu_); //
    }
    
    rgba2rgb_convert(d_rgb_yolo_input_gpu_, debayer_gpu_.d_debayer, //
                     associated_camera_params_->width, associated_camera_params_->height, yolov8_instance_->stream); //

    yolov8_instance_->preprocess_gpu(d_rgb_yolo_input_gpu_); //
    
    // --- Start FPS timing ---
    // auto inference_start_time = std::chrono::steady_clock::now();
    yolov8_instance_->infer(); //
    // auto inference_end_time = std::chrono::steady_clock::now();
    // std::chrono::duration<double, std::milli> inference_duration = inference_end_time - inference_start_time;
    // --- End FPS timing ---
    
    frame_counter_++;
    auto current_time = std::chrono::steady_clock::now();
    std::chrono::duration<double> elapsed_seconds = current_time - last_fps_update_time_;

    if (elapsed_seconds.count() >= 1.0) {
        current_fps_ = static_cast<double>(frame_counter_) / elapsed_seconds.count();
        std::cout << "YOLOv8 Worker (" << threadName << ") Inference FPS: " << current_fps_ << std::endl;
        frame_counter_ = 0;
        last_fps_update_time_ = current_time;
    }


    std::vector<pose::Object> detections; 
    yolov8_instance_->postprocess(detections); //

    if (enet_host_context_ && enet_target_peer_ &&
        enet_target_peer_->state == ENET_PEER_STATE_CONNECTED) { //
        
        fb_builder_->Clear(); //

        std::vector<flatbuffers::Offset<Orange::VisionData::Detection>> fb_detections_offsets; //

        for (const auto& det : detections) {
            Orange::VisionData::BoundingBox box_struct( //
                det.rect.x, //
                det.rect.y, //
                det.rect.width, //
                det.rect.height, //
                det.label, //
                det.prob //
            );
            
            fb_detections_offsets.push_back(Orange::VisionData::CreateDetection(*fb_builder_, &box_struct)); //
        }

        auto detections_vector = fb_builder_->CreateVector(fb_detections_offsets); //
        auto camera_serial_str = fb_builder_->CreateString(associated_camera_params_->camera_serial); //

        Orange::VisionData::YoloFrameDetectionsBuilder yolo_frame_builder(*fb_builder_); //
        yolo_frame_builder.add_camera_serial(camera_serial_str); //
        yolo_frame_builder.add_timestamp(original_entry->timestamp); //
        yolo_frame_builder.add_frame_id(original_entry->frame_id); //
        yolo_frame_builder.add_detections(detections_vector); //
        
        auto final_payload_offset = yolo_frame_builder.Finish(); //
        fb_builder_->Finish(final_payload_offset); //

        uint8_t* buf = fb_builder_->GetBufferPointer(); //
        size_t size = fb_builder_->GetSize(); //
        
        ENetPacket* packet = enet_packet_create(buf, size, ENET_PACKET_FLAG_RELIABLE); //
        enet_peer_send(enet_target_peer_, 0, packet); //
    }
    
    PutObjectToQueueOut(f); //
    return true; 
}

void YOLOv8Worker::WorkerReset() {
    std::cout << "YOLOv8Worker for " << threadName << " reset." << std::endl; //
    last_fps_update_time_ = std::chrono::steady_clock::now();
    frame_counter_ = 0;
    current_fps_ = 0.0;
}