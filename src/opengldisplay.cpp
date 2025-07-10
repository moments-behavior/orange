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

#define display_gpu_id 0 // Centralize the display GPU ID for clarity

COpenGLDisplay::COpenGLDisplay(const char* name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer_cuda_pbo, INDIGOSignalBuilder* indigo_signal_builder, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      camera_params(camera_params),
      camera_select(camera_select),
      display_buffer_pbo_cuda_ptr_(display_buffer_cuda_pbo),
      indigo_signal_builder_(indigo_signal_builder),
      h_p2p_copy_buffer_(nullptr), // Initialize the new host buffer
      d_points_for_drawing_(nullptr),
      d_skeleton_for_drawing_(nullptr),
      d_display_resize_buffer_(nullptr),
      m_stream(nullptr),
      m_recycle_queue(recycle_queue)
{
    std::cout << "[OPENGL_DISPLAY] CONSTRUCTOR for " << camera_params->camera_name << " on display GPU " << display_gpu_id << std::endl;

    // *** Set context to the display GPU for all its resources ***
    ck(cudaSetDevice(display_gpu_id));
    ck(cudaStreamCreate(&m_stream));

    // Allocate buffers that will live on the display GPU
    initalize_gpu_frame(&frame_original_gpu_, camera_params); // This will now be on the display GPU
    initialize_gpu_debayer(&debayer_gpu_, camera_params);
    ck(cudaMalloc(&d_points_for_drawing_, sizeof(float) * 4 * 2 * shaman::MAX_OBJECTS));
    ck(cudaMalloc(&d_skeleton_for_drawing_, sizeof(unsigned int) * 4 * 2));
    ck(cudaMalloc(&d_display_resize_buffer_, (size_t)camera_params->width * camera_params->height * 4));

    // *** Allocate PINNED HOST MEMORY for efficient transfers ***
    // This memory is accessible by the CPU and all GPUs
    size_t staging_buffer_size = (size_t)camera_params->width * camera_params->height * 4;
    ck(cudaHostAlloc(&h_p2p_copy_buffer_, staging_buffer_size, cudaHostAllocDefault));

    std::cout << "[OPENGL_DISPLAY] Constructor completed for " << camera_params->camera_name << std::endl;
}

COpenGLDisplay::~COpenGLDisplay()
{
    std::cout << "[OPENGL_DISPLAY] DESTRUCTOR for " << (camera_params ? camera_params->camera_name : "unknown") << std::endl;
    
    // Set context to the display GPU to safely free its resources
    ck(cudaSetDevice(display_gpu_id));

    if (m_stream) cudaStreamDestroy(m_stream);
    if (h_p2p_copy_buffer_) cudaFreeHost(h_p2p_copy_buffer_); // Use cudaFreeHost for pinned memory

    // Free all device memory
    if (frame_original_gpu_.d_orig) cudaFree(frame_original_gpu_.d_orig);
    if (debayer_gpu_.d_debayer) cudaFree(debayer_gpu_.d_debayer);
    if (d_points_for_drawing_) cudaFree(d_points_for_drawing_);
    if (d_skeleton_for_drawing_) cudaFree(d_skeleton_for_drawing_);
    if (d_display_resize_buffer_) cudaFree(d_display_resize_buffer_);
}


bool COpenGLDisplay::WorkerFunction(WORKER_ENTRY* f)
{
    if (!f) return false;

    // --- "LAST FRAME" OPTIMIZATION ---
    // 1. We've been given one frame, 'f'. See if more recent ones are in the queue.
    WORKER_ENTRY* latest_frame = f;
    WORKER_ENTRY* discarded_frame = nullptr;

    // 2. Quickly drain the queue, keeping only the very last item.
    while ((discarded_frame = GetObjectFromQueueIn()) != nullptr) {
    // A newer frame is available. The one we were holding is now outdated.
    // We MUST release the one we are replacing.
    if (latest_frame->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (latest_frame->gpu_direct_mode && latest_frame->camera_instance && latest_frame->camera_frame_struct) {
            EVT_CameraQueueFrame(latest_frame->camera_instance, latest_frame->camera_frame_struct);
        }
        m_recycle_queue.push(latest_frame);
    }
    // The newly popped frame is now the latest one.
    latest_frame = discarded_frame;
}

    // Now, 'latest_frame' holds the most recent frame. We process only this one.
    // --- END OF "LAST FRAME" OPTIMIZATION ---

    // *** Set context to the display GPU for all subsequent operations ***
    ck(cudaSetDevice(display_gpu_id));
    nppSetStream(m_stream);

    // Wait for the data in the latest_frame to be ready on its source GPU
    if (latest_frame->event_ptr) {
        ck(cudaStreamWaitEvent(m_stream, *latest_frame->event_ptr, 0));
    }

    // --- STAGED COPY (DEVICE -> HOST -> DEVICE) ---
    // This section safely moves the image data to the display GPU.
    unsigned char* source_image_ptr_on_display_gpu = nullptr;
    size_t frame_size = (size_t)camera_params->width * camera_params->height;

    if (camera_params->gpu_id != display_gpu_id) {
        // STAGE 1: Copy from Camera GPU's device memory to pinned Host memory.
        ck(cudaMemcpyAsync(h_p2p_copy_buffer_, latest_frame->d_image, frame_size, cudaMemcpyDeviceToHost, m_stream));
        
        // STAGE 2: Copy from pinned Host memory to our local Device memory on the display GPU.
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, h_p2p_copy_buffer_, frame_size, cudaMemcpyHostToDevice, m_stream));
    } else {
        // The frame is already on the display GPU, do a simple device-to-device copy.
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, latest_frame->d_image, frame_size, cudaMemcpyDeviceToDevice, m_stream));
    }
    source_image_ptr_on_display_gpu = frame_original_gpu_.d_orig;
    // --- END OF STAGED COPY ---

    // Now, all subsequent operations use buffers that are guaranteed to be on the display GPU.
    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    }

    if (f->has_detections) {
        std::cout << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": Found " << f->detections.size() << " detections. Preparing to draw boxes." << std::endl;
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
            std::cout << "[OPENGL_DISPLAY] Frame " << f->frame_id << ": Drawing " << f->detections.size() << " boxes." << std::endl;
            ck(cudaMemcpyAsync(d_points_for_drawing_, h_points.data(), h_points.size() * sizeof(float), cudaMemcpyHostToDevice, m_stream));
            gpu_draw_box(
                debayer_gpu_.d_debayer,
                camera_params->width,
                camera_params->height,
                d_points_for_drawing_,
                f->detections.size(),
                m_stream);
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
    
    // Final release of the one frame we actually used.
    if (latest_frame->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (latest_frame->gpu_direct_mode && latest_frame->camera_instance && latest_frame->camera_frame_struct) {
            EVT_CameraQueueFrame(latest_frame->camera_instance, latest_frame->camera_frame_struct);
        }
        m_recycle_queue.push(latest_frame);
    }

    return false; // We handled the object, do not put it on the output queue.
}