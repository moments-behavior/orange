#include "yolo_worker.h"
#include "kernel.cuh"
#include <cuda_runtime_api.h>
#include <iostream>
#include <vector>
#include <string>
#include <chrono>
#include "yolo_payload_generated.h"

YOLOv8Worker::YOLOv8Worker(const char* name,
                           CameraParams* cam_params,
                           CameraEachSelect* cam_select,
                           unsigned char* display_texture_buffer) // Added param
    : CThreadWorker(name),
      yolov8_instance_(nullptr),
      associated_camera_params_(cam_params),
      associated_camera_select_(cam_select),
      enet_host_context_(nullptr),
      enet_target_peer_(nullptr),
      d_rgb_yolo_input_gpu_(nullptr),
      display_texture_buffer_(display_texture_buffer), // Initialize member
      d_display_resize_buffer_(nullptr),
      d_points_for_drawing_(nullptr),
      d_skeleton_for_drawing_(nullptr),
      last_fps_update_time_(std::chrono::steady_clock::now()),
      frame_counter_(0),
      current_fps_(0.0) {

    std::cout << "YOLOv8Worker instance created for: " << name << std::endl; //

    if (!associated_camera_params_ || !associated_camera_select_) { //
        std::cerr << "YOLOv8Worker Error: CameraParams or CameraEachSelect is null for " << name << "." << std::endl; //
        return;
    }

    fb_builder_ = new flatbuffers::FlatBufferBuilder(1024 * 4); //

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

    // Initialize resize and drawing resources if a display buffer is provided
    if (display_texture_buffer_ && associated_camera_select_->stream_on) {
        input_roi_for_display_resize_ = {0, 0, associated_camera_params_->width, associated_camera_params_->height}; //
        output_display_size_.width = static_cast<int>(associated_camera_params_->width / associated_camera_select_->downsample); //
        output_display_size_.height = static_cast<int>(associated_camera_params_->height / associated_camera_select_->downsample); //
        output_roi_for_display_resize_ = {0, 0, output_display_size_.width, output_display_size_.height}; //

        if (associated_camera_select_->downsample != 1) { //
            ck(cudaMalloc((void **)&d_display_resize_buffer_, output_display_size_.width * output_display_size_.height * 4 * sizeof(unsigned char)));
        }

        // Resources for drawing detections (e.g., a simple box)
        unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // Box lines: (0,1), (1,2), (2,3), (3,0)
        ck(cudaMalloc((void **)&d_points_for_drawing_, sizeof(float) * 8)); // 4 points (x,y) for a box
        ck(cudaMalloc((void **)&d_skeleton_for_drawing_, sizeof(unsigned int) * 8)); //
        CHECK(cudaMemcpyAsync(d_skeleton_for_drawing_, skeleton, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice, yolov8_instance_->stream)); //
    }

    std::cout << "YOLOv8Worker for " << name << " initialized successfully." << std::endl; //
}

YOLOv8Worker::~YOLOv8Worker() {
    std::cout << "YOLOv8Worker instance for " << threadName << " being destroyed." << std::endl; //
    delete yolov8_instance_; //
    yolov8_instance_ = nullptr; //

    delete fb_builder_; //
    fb_builder_ = nullptr; //

    // Ensure CUDA context is active for freeing GPU memory if params exist
    if (associated_camera_params_ && yolov8_instance_) { // Check yolov8_instance_ to ensure stream is valid
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
    if (d_display_resize_buffer_) {
        cudaFree(d_display_resize_buffer_);
        d_display_resize_buffer_ = nullptr;
    }
    if (d_points_for_drawing_) {
        cudaFree(d_points_for_drawing_);
        d_points_for_drawing_ = nullptr;
    }
    if (d_skeleton_for_drawing_) {
        cudaFree(d_skeleton_for_drawing_);
        d_skeleton_for_drawing_ = nullptr;
    }
    std::cout << "YOLOv8Worker for " << threadName << " resources cleaned up." << std::endl; //
}

void YOLOv8Worker::SetENetTarget(EnetContext* host_ctx, ENetPeer* target_peer) {
    // ... (implementation as before)
    enet_host_context_ = host_ctx; //
    enet_target_peer_ = target_peer; //
    if (target_peer) { //
        std::cout << "YOLOv8Worker " << threadName << " ENet target set." << std::endl; //
    } else {
        std::cout << "YOLOv8Worker " << threadName << " ENet target cleared." << std::endl; //
    }
}

bool YOLOv8Worker::WorkerFunction(void* f) {
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
    yolov8_instance_->infer(); //
    
    frame_counter_++; //
    auto current_time = std::chrono::steady_clock::now(); //
    std::chrono::duration<double> elapsed_seconds = current_time - last_fps_update_time_; //

    if (elapsed_seconds.count() >= 1.0) { //
        current_fps_ = static_cast<double>(frame_counter_) / elapsed_seconds.count(); //
        std::cout << "YOLOv8 Worker (" << threadName << ") Inference FPS: " << current_fps_ << std::endl; //
        frame_counter_ = 0; //
        last_fps_update_time_ = current_time; //
    }

    std::vector<pose::Object> detections; 
    yolov8_instance_->postprocess(detections); //

    // --- Draw detections and copy to display buffer ---
    if (display_texture_buffer_ && associated_camera_select_->stream_on) {
        if (!detections.empty()) {
            // For simplicity, draw the bounding box of the first detection
            // This logic can be expanded to draw all detections or more complex visuals
            const auto& obj = detections[0];
            float points[8] = {
                obj.rect.x, obj.rect.y,                                 // Top-left
                obj.rect.x + obj.rect.width, obj.rect.y,                // Top-right
                obj.rect.x + obj.rect.width, obj.rect.y + obj.rect.height, // Bottom-right
                obj.rect.x, obj.rect.y + obj.rect.height                // Bottom-left
            };
            // The skeleton for a box: 0-1, 1-2, 2-3, 3-0. d_skeleton_for_drawing_ should be set up for this.
            CHECK(cudaMemcpyAsync(d_points_for_drawing_, points, sizeof(float) * 8, cudaMemcpyHostToDevice, yolov8_instance_->stream)); //
            // Assuming debayer_gpu_.d_debayer is RGBA and gpu_draw_rat_pose can draw on RGBA (num_channels=4)
            gpu_draw_rat_pose(debayer_gpu_.d_debayer, associated_camera_params_->width, associated_camera_params_->height, d_points_for_drawing_, d_skeleton_for_drawing_, yolov8_instance_->stream, 4); //
        }

        unsigned char* source_for_display = debayer_gpu_.d_debayer;
        int source_width = associated_camera_params_->width;
        int source_height = associated_camera_params_->height;

        if (associated_camera_select_->downsample != 1) { //
            NppiSize srcSize = {associated_camera_params_->width, associated_camera_params_->height};
            const NppStatus npp_result = nppiResize_8u_C4R( //
                debayer_gpu_.d_debayer, //
                associated_camera_params_->width * 4, // Source step (pitch)
                srcSize,                               
                input_roi_for_display_resize_,       
                d_display_resize_buffer_,
                output_display_size_.width * 4,      
                output_display_size_,                
                output_roi_for_display_resize_,      
                NPPI_INTER_SUPER); //
            if (npp_result != NPP_SUCCESS) { //
                std::cerr << "YOLO Worker: Error executing resize -- code: " << npp_result << std::endl; //
            }
            source_for_display = d_display_resize_buffer_;
            source_width = output_display_size_.width;
            source_height = output_display_size_.height;
        }
        
        ck(cudaMemcpy2DAsync(display_texture_buffer_, //
                         source_width * 4, //
                         source_for_display, //
                         source_width * 4, //
                         source_width * 4, //
                         source_height, //
                         cudaMemcpyDeviceToDevice, //
                         yolov8_instance_->stream));
    }
    // --- End drawing and copying ---


    if (enet_host_context_ && enet_target_peer_ &&
        enet_target_peer_->state == ENET_PEER_STATE_CONNECTED) { //
        
        fb_builder_->Clear(); //
        // ... (rest of FlatBuffer building and sending logic as before) ...
        std::vector<flatbuffers::Offset<Orange::VisionData::Detection>> fb_detections_offsets; //

        for (const auto& det : detections) { //
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
    if (display_texture_buffer_ && associated_camera_select_->stream_on) { //
        cudaStreamSynchronize(yolov8_instance_->stream); //
    }
    return true; 
}

void YOLOv8Worker::WorkerReset() {
    std::cout << "YOLOv8Worker for " << threadName << " reset." << std::endl; //
    last_fps_update_time_ = std::chrono::steady_clock::now(); //
    frame_counter_ = 0; //
    current_fps_ = 0.0; //
}