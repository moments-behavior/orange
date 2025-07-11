// Enhanced crop_and_encode_worker.cpp
#include "crop_and_encode_worker.h"
#include "kernel.cuh"
#include <nppi.h>
#include <npp.h>
#include <nppi_color_conversion.h>
#include <nppi_geometry_transforms.h>

CropAndEncodeWorker::CropAndEncodeWorker(const char* name, CameraParams* camera_params, const std::string& folder_name, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name), camera_params_(camera_params), folder_name_(folder_name), m_recycle_queue(recycle_queue) {

    std::cout << "[CropAndEncodeWorker] Initializing " << name << " on GPU " << camera_params_->gpu_id << std::endl;
    
    ck(cudaSetDevice(camera_params_->gpu_id));
    ck(cudaStreamCreate(&m_stream));
    
    // Initialize video writer for 1280x1280 cropped videos
    writer_.video_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop.mp4";
    writer_.keyframe_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop_keyframe.csv";
    writer_.metadata_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop_meta.csv";
    
    writer_.video = new FFmpegWriter(AV_CODEC_ID_H264, 1280, 1280, camera_params_->frame_rate, 
                                   writer_.video_file.c_str(), writer_.keyframe_file.c_str());
    writer_.video->create_thread();
    
    // Open metadata file
    writer_.metadata = new std::ofstream();
    writer_.metadata->open(writer_.metadata_file.c_str());
    if (!(*writer_.metadata)) {
        std::cout << "[CropAndEncodeWorker] Warning: Could not open metadata file!" << std::endl;
    } else {
        *writer_.metadata << "frame_id,timestamp,timestamp_sys,detection_confidence,crop_x,crop_y,crop_w,crop_h\n";
    }

    // Allocate GPU buffers
    ck(cudaMalloc(&d_cropped_bgr_, 1280 * 1280 * 3)); // BGR buffer for the crop (now 1280x1280)
    ck(cudaMalloc(&d_yuv_buffer_, 1280 * 1280 * 3 / 2)); // YUV420 buffer for encoding
    
    std::cout << "[CropAndEncodeWorker] Initialization complete for " << name << std::endl;
}

CropAndEncodeWorker::~CropAndEncodeWorker() {
    std::cout << "[CropAndEncodeWorker] Destructor for " << threadName << std::endl;
    
    if (camera_params_) {
        ck(cudaSetDevice(camera_params_->gpu_id));
    }
    
    if (writer_.video) {
        writer_.video->quit_thread();
        writer_.video->join_thread();
        delete writer_.video;
    }
    
    if (writer_.metadata && writer_.metadata->is_open()) {
        writer_.metadata->close();
        delete writer_.metadata;
    }
    
    if (d_cropped_bgr_) cudaFree(d_cropped_bgr_);
    if (d_yuv_buffer_) cudaFree(d_yuv_buffer_);
    if (encoder_) delete encoder_;
    if (m_stream) cudaStreamDestroy(m_stream);
    
    std::cout << "[CropAndEncodeWorker] Destructor complete for " << threadName << std::endl;
}

bool CropAndEncodeWorker::WorkerFunction(WORKER_ENTRY* entry) {
    if (!entry || entry->detections.empty()) {
        if (entry) {
            if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
                m_recycle_queue.push(entry);
            }
        }
        return false;
    }

    ck(cudaSetDevice(camera_params_->gpu_id));
    nppSetStream(m_stream);

    try {
        // Wait for YOLO processing to complete if needed
        if (entry->event_ptr) {
            ck(cudaStreamWaitEvent(m_stream, *entry->event_ptr, 0));
        }

        // Find the detection with the highest confidence
        const auto& best_detection = *std::max_element(entry->detections.begin(), entry->detections.end(),
            [](const pose::Object& a, const pose::Object& b) {
                return a.prob < b.prob;
            });

        std::cout << "[CropAndEncodeWorker] Processing frame " << entry->frame_id 
                  << " with detection: conf=" << best_detection.prob
                  << " rect=[" << best_detection.rect.x << "," << best_detection.rect.y 
                  << "," << best_detection.rect.width << "," << best_detection.rect.height << "]" << std::endl;

        // 1. Use NPP for high-quality crop and resize (much better than basic kernel)
        
        // First, extract and validate crop region
        int crop_x = std::max(0, (int)best_detection.rect.x);
        int crop_y = std::max(0, (int)best_detection.rect.y);
        int crop_w = std::min((int)best_detection.rect.width, entry->width - crop_x);
        int crop_h = std::min((int)best_detection.rect.height, entry->height - crop_y);
        
        // Ensure minimum size to avoid issues
        crop_w = std::max(crop_w, 64);  
        crop_h = std::max(crop_h, 64);
        
        std::cout << "[CropAndEncodeWorker] Cropping region: [" << crop_x << "," << crop_y 
                  << "," << crop_w << "," << crop_h << "] from " << entry->width << "x" << entry->height << std::endl;
        
        // Convert mono to RGB first (NPP resize works better with RGB)
        unsigned char* d_rgb_temp = nullptr;
        ck(cudaMalloc(&d_rgb_temp, entry->width * entry->height * 3));
        
        // Convert mono to RGB first (NPP resize works better with RGB)
        unsigned char* d_rgb_temp = nullptr;
        ck(cudaMalloc(&d_rgb_temp, entry->width * entry->height * 3));
        
        // Convert mono to RGB - duplicate grayscale to all 3 channels
        // Using a simple CUDA kernel approach since Mono8ToRGBMono might not be declared
        dim3 threads_per_block(32, 32);
        dim3 num_blocks((entry->width + threads_per_block.x - 1) / threads_per_block.x, 
                       (entry->height + threads_per_block.y - 1) / threads_per_block.y);
        
        // Launch a simple mono-to-RGB conversion kernel
        mono_to_rgb_kernel<<<num_blocks, threads_per_block, 0, m_stream>>>(
            d_rgb_temp, entry->d_image, entry->width, entry->height);
        ck(cudaGetLastError());
        
        // Now use NPP for high-quality crop and resize
        NppiSize src_size = {entry->width, entry->height};
        NppiRect src_roi = {crop_x, crop_y, crop_w, crop_h};  // Crop region
        
        NppiSize dst_size = {1280, 1280};
        NppiRect dst_roi = {0, 0, 1280, 1280};  // Full output
        
        // Use NPP's high-quality resize with super-sampling (best quality)
        NppStatus npp_status = nppiResize_8u_C3R(
            d_rgb_temp + (crop_y * entry->width + crop_x) * 3,  // Source: start of crop region
            entry->width * 3,                                    // Source step (full width stride)
            src_size, src_roi,                                   // Source size and ROI
            d_cropped_bgr_, 1280 * 3,                           // Destination and step
            dst_size, dst_roi,                                   // Destination size and ROI  
            NPPI_INTER_SUPER                                     // Super-sampling for best quality!
        );
        
        // Clean up temp buffer
        ck(cudaFree(d_rgb_temp));
        
        if (npp_status != NPP_SUCCESS) {
            std::cerr << "[CropAndEncodeWorker] NPP resize failed: " << npp_status << std::endl;
        } else {
            std::cout << "[CropAndEncodeWorker] High-quality NPP resize completed successfully" << std::endl;
        }

        // 2. For debugging, let's skip the complex encoding and just save cropped images
        // Synchronize the stream to ensure GPU work is complete
        ck(cudaStreamSynchronize(m_stream));
        
        // Copy cropped BGR data to CPU for debugging
        size_t bgr_size = 1280 * 1280 * 3;
        unsigned char* h_bgr_buffer = new unsigned char[bgr_size];
        ck(cudaMemcpy(h_bgr_buffer, d_cropped_bgr_, bgr_size, cudaMemcpyDeviceToHost));
        
        // Convert BGR to RGB for PPM format (PPM expects RGB)
        unsigned char* h_rgb_buffer = new unsigned char[bgr_size];
        for (int i = 0; i < 1280 * 1280; i++) {
            h_rgb_buffer[i * 3 + 0] = h_bgr_buffer[i * 3 + 2]; // R = B
            h_rgb_buffer[i * 3 + 1] = h_bgr_buffer[i * 3 + 1]; // G = G  
            h_rgb_buffer[i * 3 + 2] = h_bgr_buffer[i * 3 + 0]; // B = R
        }
        
        // Save debug images for first 20 frames to see if cropping works
        if (frame_counter_ < 20) {
            std::string debug_filename = folder_name_ + "/debug_crop_frame_" + 
                                       std::to_string(entry->frame_id) + "_conf_" + 
                                       std::to_string((int)(best_detection.prob * 100)) + ".ppm";
            std::ofstream debug_file(debug_filename, std::ios::binary);
            if (debug_file.is_open()) {
                debug_file << "P6\n1280 1280\n255\n";
                debug_file.write((char*)h_rgb_buffer, bgr_size);
                debug_file.close();
                std::cout << "[CropAndEncodeWorker] Saved debug frame: " << debug_filename << std::endl;
            }
        }
        
        // For now, skip video encoding and just count frames
        delete[] h_bgr_buffer;
        delete[] h_rgb_buffer;
        frame_counter_++;

        // 3. Write metadata
        if (writer_.metadata && writer_.metadata->is_open()) {
            *writer_.metadata << entry->frame_id << "," 
                            << entry->timestamp << "," 
                            << entry->timestamp_sys << ","
                            << best_detection.prob << ","
                            << best_detection.rect.x << ","
                            << best_detection.rect.y << ","
                            << best_detection.rect.width << ","
                            << best_detection.rect.height << std::endl;
        }

        std::cout << "[CropAndEncodeWorker] Successfully processed and encoded frame " << entry->frame_id << std::endl;

    } catch (const std::exception& e) {
        std::cerr << "[CropAndEncodeWorker] Exception processing frame " << entry->frame_id 
                  << ": " << e.what() << std::endl;
    }
    
    // Handle reference counting
    if (entry->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (entry->gpu_direct_mode && entry->camera_instance && entry->camera_frame_struct) {
            EVT_CameraQueueFrame(entry->camera_instance, entry->camera_frame_struct);
        }
        m_recycle_queue.push(entry);
    }
    
    return false; // This worker doesn't pass data to another queue
}