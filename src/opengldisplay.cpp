// src/opengldisplay.cpp

#include "opengldisplay.h"
#include "enet_thread.h"
#include "cuda_context_debug.h"
#include "kernel.cuh"
#include "shaman.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include "global.h"
#include <npp.h> // For nppSetStream
#include "yolo_worker.h"

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

    if (f->event_ptr) { // Wait for acquire_frames to be ready
        ck(cudaStreamWaitEvent(m_stream, *f->event_ptr, 0));
    }
    
    // --- START: MODIFIED LOGIC ---
    // If this frame is supposed to have detections, we MUST wait for the YOLO worker.
    if (f->has_detections && camera_select->yolo) {
        YOLOv8Worker* yolo_worker = nullptr;
        // Find the YOLO worker that corresponds to this display instance
        for (auto* worker : yolo_workers) {
             if (worker && worker->GetCameraParams()->camera_id == this->camera_params->camera_id) {
                yolo_worker = worker;
                break;
            }
        }

        if (yolo_worker) {
            std::cout << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": Waiting for YOLO inference to complete." << std::endl;
            // *** THIS IS THE FIX ***
            // Wait for the specific event that signals YOLO inference is done for this frame.
            ck(cudaStreamWaitEvent(m_stream, yolo_worker->m_inference_completed, 0));
            std::cout << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": YOLO inference complete. Proceeding to draw." << std::endl;
        } else {
             std::cerr << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": has_detections is true, but no corresponding YOLO worker was found!" << std::endl;
        }
    }
    // --- END: MODIFIED LOGIC ---
    
    frame_original_gpu_.d_orig = f->d_image;
    
    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    }

    if (f->has_detections) {
        std::cout << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": Detected " << f->detections.size() << " objects. Preparing to draw boxes." << std::endl;
        std::vector<float> h_points;
        h_points.reserve(f->detections.size() * 4 * 2);

        for(const auto& obj : f->detections) {
            // Top line
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y);
            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y);

            // Right line
            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y);
            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y + obj.rect.height);

            // Bottom line
            h_points.push_back(obj.rect.x + obj.rect.width);
            h_points.push_back(obj.rect.y + obj.rect.height);
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y + obj.rect.height);
            
            // Left line
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y + obj.rect.height);
            h_points.push_back(obj.rect.x);
            h_points.push_back(obj.rect.y);
        }

        if (!h_points.empty()) {
            std::cout << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": Copying " << h_points.size() / 2 << " points to GPU and launching draw kernel." << std::endl;
            ck(cudaMemcpyAsync(d_points_for_drawing_, h_points.data(), h_points.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream));
            gpu_draw_box(debayer_gpu_.d_debayer, camera_params->width, camera_params->height, d_points_for_drawing_, m_stream);
        }
    }

    if (camera_select->downsample > 1) {
        output_display_size_.width = camera_params->width / camera_select->downsample;
        output_display_size_.height = camera_params->height / camera_select->downsample;
    
        NppiSize input_size = {static_cast<int>(camera_params->width), static_cast<int>(camera_params->height)};
        NppiRect input_roi = {0, 0, static_cast<int>(camera_params->width), static_cast<int>(camera_params->height)};
        
        NppiRect output_roi = {0, 0, output_display_size_.width, output_display_size_.height};
    
        nppiResize_8u_C4R(
            debayer_gpu_.d_debayer,
            camera_params->width * 4,
            input_size,
            input_roi,
            d_display_resize_buffer_,
            output_display_size_.width * 4,
            output_display_size_,
            output_roi,
            NPPI_INTER_SUPER
        );
        
        size_t copy_size = static_cast<size_t>(output_display_size_.width) * static_cast<size_t>(output_display_size_.height) * 4;
        
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