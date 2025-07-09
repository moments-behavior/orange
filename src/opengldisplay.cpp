// src/opengldisplay.cpp

#include "opengldisplay.h"
#include "cuda_context_debug.h"
#include "kernel.cuh"
#include "shaman.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include "global.h"
#include <npp.h> // For nppSetStream

COpenGLDisplay::COpenGLDisplay(const char* name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer_cuda_pbo, INDIGOSignalBuilder* indigo_signal_builder, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      camera_params(camera_params),
      camera_select(camera_select),
      display_buffer_pbo_cuda_ptr_(display_buffer_cuda_pbo),
      indigo_signal_builder_(indigo_signal_builder),
      d_points_for_drawing_(nullptr),
      d_skeleton_for_drawing_(nullptr),
      d_display_resize_buffer_(nullptr),
      m_stream(nullptr),
      m_recycle_queue(recycle_queue)
{
    std::cout << "========================================" << std::endl;
    std::cout << "[OPENGL_DISPLAY] CONSTRUCTOR CALLED" << std::endl;
    std::cout << "  Camera: " << camera_params->camera_name << std::endl;
    std::cout << "  Downsample: " << camera_select->downsample << std::endl;
    std::cout << "  Thread name: " << (name ? name : "null") << std::endl;
    std::cout << "========================================" << std::endl;

    ck(cudaSetDevice(camera_params->gpu_id));
    ck(cudaStreamCreate(&m_stream));

    initalize_gpu_frame(&frame_original_gpu_, camera_params);
    initialize_gpu_debayer(&debayer_gpu_, camera_params);
    ck(cudaMalloc(&d_points_for_drawing_, sizeof(float) * 4 * 2 * shaman::MAX_OBJECTS));
    ck(cudaMalloc(&d_skeleton_for_drawing_, sizeof(unsigned int) * 4 * 2));
    ck(cudaMalloc(&d_display_resize_buffer_, camera_params->width * camera_params->height * 4));

    std::cout << "[OPENGL_DISPLAY] Constructor completed:" << std::endl;
    std::cout << "  - Camera: " << camera_params->camera_name << std::endl;
    std::cout << "  - Resolution: " << camera_params->width << "x" << camera_params->height << std::endl;
    std::cout << "  - Downsample: " << camera_select->downsample << std::endl;
    std::cout << "  - Buffer allocated: " << (d_display_resize_buffer_ != nullptr ? "Yes" : "No") << std::endl;
    std::cout << "  - Stream created: " << (m_stream != nullptr ? "Yes" : "No") << std::endl;
}

COpenGLDisplay::~COpenGLDisplay()
{
    std::cout << "========================================" << std::endl;
    std::cout << "[OPENGL_DISPLAY] DESTRUCTOR CALLED" << std::endl;
    if (camera_params) {
        std::cout << "  Camera: " << camera_params->camera_name << std::endl;
    }
    std::cout << "========================================" << std::endl;
    if (camera_params) {
        ck(cudaSetDevice(camera_params->gpu_id));
    }

    if (m_stream) {
        cudaStreamDestroy(m_stream);
    }

    if (frame_original_gpu_.d_orig) cudaFree(frame_original_gpu_.d_orig);
    if (debayer_gpu_.d_debayer) cudaFree(debayer_gpu_.d_debayer);
    if (d_points_for_drawing_) cudaFree(d_points_for_drawing_);
    if (d_skeleton_for_drawing_) cudaFree(d_skeleton_for_drawing_);
    if (d_display_resize_buffer_) cudaFree(d_display_resize_buffer_);
}


bool COpenGLDisplay::WorkerFunction(WORKER_ENTRY* f)
{
    if (!f) return false;

    ck(cudaSetDevice(camera_params->gpu_id));
    nppSetStream(m_stream); 

    if (f->event_ptr) { // Check if the pointer is not null
        ck(cudaStreamWaitEvent(m_stream, *f->event_ptr, 0)); // Dereference the pointer to get the event
        // The event is now owned by this worker. We can choose to destroy it or recycle it.
        // For simplicity and to match the logic in acquire_frames, let's assume events are single-use for now.
        // The acquire_frames thread will push new events to the pool.
        // NOTE: A more robust system might have this worker push the event back to the free_events_queue.
    }
    frame_original_gpu_.d_orig = f->d_image;
    
    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    }

    if (f->has_detections) {
        std::vector<float> h_points;
        h_points.reserve(f->detections.size() * 4 * 2);

        for(const auto& obj : f->detections) {
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y);
            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y);

            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y);
            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y + obj.rect.height);

            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y + obj.rect.height);
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y + obj.rect.height);

            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y + obj.rect.height);
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y);
        }

        if (!h_points.empty()) {
            ck(cudaMemcpyAsync(d_points_for_drawing_, h_points.data(), h_points.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream));
            gpu_draw_box(debayer_gpu_.d_debayer, camera_params->width, camera_params->height, d_points_for_drawing_, m_stream);
        }
    }

    // Logging for debugging purposes, comment out if not needed
    // std::cout << "[OPENGL_DISPLAY] About to check downsample: " << camera_select->downsample << std::endl;
    // static bool debug_printed = false;
    // if (!debug_printed && camera_select->downsample > 1) {
    //     std::cout << "=== Downsampling Debug Info ===" << std::endl;
    //     std::cout << "Camera: " << camera_params->camera_name << std::endl;
    //     std::cout << "Original resolution: " << camera_params->width << "x" << camera_params->height << std::endl;
    //     std::cout << "Downsample factor: " << camera_select->downsample << std::endl;
    //     std::cout << "Target resolution: " << (camera_params->width / camera_select->downsample) 
    //             << "x" << (camera_params->height / camera_select->downsample) << std::endl;
    //     std::cout << "Input buffer size: " << (camera_params->width * camera_params->height * 4) << " bytes" << std::endl;
    //     std::cout << "Output buffer size: " << ((camera_params->width / camera_select->downsample) * 
    //                                         (camera_params->height / camera_select->downsample) * 4) << " bytes" << std::endl;
    //     std::cout << "Display buffer pointer: " << (void*)display_buffer_pbo_cuda_ptr_ << std::endl;
    //     std::cout << "Resize buffer pointer: " << (void*)d_display_resize_buffer_ << std::endl;
    //     debug_printed = true;
    // }

    if (camera_select->downsample > 1) {
        // std::cout << "[OPENGL_DISPLAY] DOWNSAMPLING PATH - starting resize..." << std::endl;
        output_display_size_.width = camera_params->width / camera_select->downsample;
        output_display_size_.height = camera_params->height / camera_select->downsample;
    
        // Validate buffer size
        size_t required_buffer_size = static_cast<size_t>(output_display_size_.width) * 
                                      static_cast<size_t>(output_display_size_.height) * 4;
        size_t allocated_buffer_size = static_cast<size_t>(camera_params->width) * 
                                       static_cast<size_t>(camera_params->height) * 4;
        
        if (required_buffer_size > allocated_buffer_size) {
            std::cerr << "ERROR: Resize buffer too small! Required: " << required_buffer_size 
                      << ", Allocated: " << allocated_buffer_size << std::endl;
            return false;
        }

        // Input parameters (source image)
        NppiSize input_size = {static_cast<int>(camera_params->width), static_cast<int>(camera_params->height)};
        NppiRect input_roi = {0, 0, static_cast<int>(camera_params->width), static_cast<int>(camera_params->height)};
        
        // Output parameters (destination image)
        NppiRect output_roi = {0, 0, output_display_size_.width, output_display_size_.height};
    
        // Perform the resize operation
        NppStatus npp_status = nppiResize_8u_C4R(
            debayer_gpu_.d_debayer,                    // source data pointer
            camera_params->width * 4,                  // source step (bytes per row)
            input_size,                                // source size
            input_roi,                                 // source ROI
            d_display_resize_buffer_,                  // destination data pointer  
            output_display_size_.width * 4,            // destination step (bytes per row)
            output_display_size_,                      // destination size
            output_roi,                                // destination ROI
            NPPI_INTER_SUPER                          // interpolation method
        );
        
        if (npp_status != NPP_SUCCESS) {
            std::cerr << "NPP Resize failed with error: " << npp_status << std::endl;
            std::cerr << "Input size: " << input_size.width << "x" << input_size.height << std::endl;
            std::cerr << "Output size: " << output_display_size_.width << "x" << output_display_size_.height << std::endl;
            std::cerr << "Input step: " << (camera_params->width * 4) << std::endl;
            std::cerr << "Output step: " << (output_display_size_.width * 4) << std::endl;
            // Continue execution to see if it still works
        }
    
        // Copy resized data to display buffer
        size_t copy_size = static_cast<size_t>(output_display_size_.width) * 
                           static_cast<size_t>(output_display_size_.height) * 4;
        
        ck(cudaMemcpyAsync(display_buffer_pbo_cuda_ptr_, 
                            d_display_resize_buffer_,
                            copy_size,
                            cudaMemcpyDeviceToDevice, 
                            m_stream));
    } else {
        size_t copy_size = static_cast<size_t>(camera_params->width) * static_cast<size_t>(camera_params->height) * 4;
        
        ck(cudaMemcpyAsync(display_buffer_pbo_cuda_ptr_, 
                           debayer_gpu_.d_debayer,
                           copy_size,
                           cudaMemcpyDeviceToDevice, 
                           m_stream));
    }

    ck(cudaStreamSynchronize(m_stream));

    if (f->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (f->gpu_direct_mode && f->camera_instance && f->camera_frame_struct) {
            EVT_CameraQueueFrame(f->camera_instance, f->camera_frame_struct);
        }
        m_recycle_queue.push(f);
    }

    return false;
}