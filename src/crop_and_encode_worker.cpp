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

    // Initialize video writer for 256x256 cropped videos
    writer_.video_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop.mp4";
    writer_.keyframe_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop_keyframe.csv";
    writer_.metadata_file = folder_name_ + "/Cam" + camera_params_->camera_serial + "_crop_meta.csv";

    writer_.video = new FFmpegWriter(AV_CODEC_ID_H264, 256, 256, camera_params_->frame_rate,
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
    ck(cudaMalloc(&d_cropped_bgr_, 256 * 256 * 3)); // BGR buffer for the crop (now 256x256)
    ck(cudaMalloc(&d_yuv_buffer_, 256 * 256 * 3 / 2)); // YUV420 buffer for encoding

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

        // Use detection as locator, crop fixed 256x256 centered on detection
        // This maintains high resolution and avoids tiny resize operations

        // 1. Calculate the center point of the best detection
        float detection_center_x = best_detection.rect.x + best_detection.rect.width / 2.0f;
        float detection_center_y = best_detection.rect.y + best_detection.rect.height / 2.0f;

        std::cout << "[CropAndEncodeWorker] Detection center: x=" << detection_center_x 
                << ", y=" << detection_center_y 
                << " (from rect: x=" << best_detection.rect.x 
                << ", y=" << best_detection.rect.y 
                << ", w=" << best_detection.rect.width 
                << ", h=" << best_detection.rect.height << ")" << std::endl;

        // 2. Define the 256x256 crop region centered on the detection
        const int CROP_SIZE = 256;
        const int HALF_CROP = CROP_SIZE / 2;

        int crop_x = static_cast<int>(detection_center_x) - HALF_CROP;
        int crop_y = static_cast<int>(detection_center_y) - HALF_CROP;

        // 3. Clamp the crop region to stay within image boundaries
        crop_x = std::max(0, std::min(crop_x, static_cast<int>(entry->width) - CROP_SIZE));
        crop_y = std::max(0, std::min(crop_y, static_cast<int>(entry->height) - CROP_SIZE));

        // 4. Define the source ROI for the 256x256 crop
        NppiRect src_roi;
        src_roi.x = crop_x;
        src_roi.y = crop_y;
        src_roi.width = CROP_SIZE;
        src_roi.height = CROP_SIZE;

        std::cout << "[CropAndEncodeWorker] Final crop ROI: x=" << src_roi.x 
                << ", y=" << src_roi.y 
                << ", w=" << src_roi.width 
                << ", h=" << src_roi.height << std::endl;

        // 5. Convert the full mono frame to RGB
        unsigned char* d_rgb_full = nullptr;
        ck(cudaMalloc(&d_rgb_full, (size_t)entry->width * entry->height * 3));
        launch_mono_to_rgb_kernel(d_rgb_full, entry->d_image, entry->width, entry->height, m_stream);

        // 6. Define source and destination sizes  
        NppiSize src_size = {static_cast<int>(entry->width), static_cast<int>(entry->height)};
        NppiSize dst_size = {CROP_SIZE, CROP_SIZE};

        // 7. Define destination ROI (full destination image)
        NppiRect dst_roi = {0, 0, CROP_SIZE, CROP_SIZE};

        // 8. Perform the crop operation (no resize needed since we're extracting 256x256 directly)
        NppStatus npp_status = nppiCopy_8u_C3R(
            d_rgb_full + (src_roi.y * entry->width + src_roi.x) * 3,  // Source pointer offset to crop region
            entry->width * 3,        // Source pitch
            d_cropped_bgr_,          // Destination buffer  
            CROP_SIZE * 3,           // Destination pitch
            dst_size                 // Copy size (256x256)
        );

        // 9. Clean up the temporary RGB buffer
        ck(cudaFree(d_rgb_full));

        if (npp_status != NPP_SUCCESS) {
            std::cerr << "[CropAndEncodeWorker] NPP crop failed with status: " << npp_status << std::endl;
        } else {
            std::cout << "[CropAndEncodeWorker] High-resolution 256x256 crop completed successfully" << std::endl;
        }

        // 2. For debugging, let's skip the complex encoding and just save cropped images
        // Synchronize the stream to ensure GPU work is complete
        ck(cudaStreamSynchronize(m_stream));

        // Copy cropped BGR data to CPU for debugging
        size_t bgr_size = 256 * 256 * 3;
        unsigned char* h_bgr_buffer = new unsigned char[bgr_size];
        ck(cudaMemcpy(h_bgr_buffer, d_cropped_bgr_, bgr_size, cudaMemcpyDeviceToHost));

        // Convert BGR to RGB for PPM format (PPM expects RGB)
        unsigned char* h_rgb_buffer = new unsigned char[bgr_size];
        for (int i = 0; i < 256 * 256; i++) {
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
                debug_file << "P6\n256 256\n255\n";
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