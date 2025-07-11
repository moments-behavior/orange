// src/opengldisplay.cpp

#include "opengldisplay.h"
#include "enet_thread.h"
#include "cuda_context_debug.h"
#include <vector>
#include "kernel.cuh"
#include "shaman.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>
#include "global.h"
#include <npp.h> // For nppSetStream
#include "yolo_worker.h"

#define display_gpu_id 0 

COpenGLDisplay::COpenGLDisplay(const char* name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer_cuda_pbo, INDIGOSignalBuilder* indigo_signal_builder, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      camera_params(camera_params),
      camera_select(camera_select),
      display_buffer_pbo_cuda_ptr_(display_buffer_cuda_pbo),
      indigo_signal_builder_(indigo_signal_builder),
      h_p2p_copy_buffer_(nullptr),
      d_detections_for_drawing_(nullptr), // Changed from d_points_for_drawing_
      d_skeleton_for_drawing_(nullptr),
      d_display_resize_buffer_(nullptr),
      m_stream(nullptr),
      m_recycle_queue(recycle_queue)
{
    std::cout << "[OPENGL_DISPLAY] CONSTRUCTOR for " << camera_params->camera_name << " on display GPU " << display_gpu_id << std::endl;
    ck(cudaSetDevice(display_gpu_id));
    ck(cudaStreamCreate(&m_stream));

    initalize_gpu_frame(&frame_original_gpu_, camera_params);
    initialize_gpu_debayer(&debayer_gpu_, camera_params);
    
    // UPDATED: Allocate buffer for pose::Object structs, not floats
    ck(cudaMalloc(&d_detections_for_drawing_, sizeof(pose::Object) * shaman::MAX_OBJECTS));
    
    ck(cudaMalloc(&d_skeleton_for_drawing_, sizeof(unsigned int) * 4 * 2));
    ck(cudaMalloc(&d_display_resize_buffer_, (size_t)camera_params->width * camera_params->height * 4));

    size_t staging_buffer_size = (size_t)camera_params->width * camera_params->height * 4;
    ck(cudaHostAlloc(&h_p2p_copy_buffer_, staging_buffer_size, cudaHostAllocDefault));
    std::cout << "[OPENGL_DISPLAY] Constructor completed for " << camera_params->camera_name << std::endl;
}

COpenGLDisplay::~COpenGLDisplay()
{
    std::cout << "[OPENGL_DISPLAY] DESTRUCTOR for " << (camera_params ? camera_params->camera_name : "unknown") << std::endl;
    ck(cudaSetDevice(display_gpu_id));

    if (m_stream) cudaStreamDestroy(m_stream);
    if (h_p2p_copy_buffer_) cudaFreeHost(h_p2p_copy_buffer_);

    if (frame_original_gpu_.d_orig) cudaFree(frame_original_gpu_.d_orig);
    if (debayer_gpu_.d_debayer) cudaFree(debayer_gpu_.d_debayer);
    if (d_detections_for_drawing_) cudaFree(d_detections_for_drawing_);
    if (d_skeleton_for_drawing_) cudaFree(d_skeleton_for_drawing_);
    if (d_display_resize_buffer_) cudaFree(d_display_resize_buffer_);
}


bool COpenGLDisplay::WorkerFunction(WORKER_ENTRY* f)
{
    if (!f) return false;

    WORKER_ENTRY* latest_frame = f;
    WORKER_ENTRY* discarded_frame = nullptr;

    while ((discarded_frame = GetObjectFromQueueIn()) != nullptr) {
        if (latest_frame->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
            if (latest_frame->gpu_direct_mode && latest_frame->camera_instance && latest_frame->camera_frame_struct) {
                EVT_CameraQueueFrame(latest_frame->camera_instance, latest_frame->camera_frame_struct);
            }
            m_recycle_queue.push(latest_frame);
        }
        latest_frame = discarded_frame;
    }

    ck(cudaSetDevice(display_gpu_id));
    nppSetStream(m_stream);

    if (latest_frame->event_ptr) {
        ck(cudaStreamWaitEvent(m_stream, *latest_frame->event_ptr, 0));
    }

    if (latest_frame->has_detections && latest_frame->yolo_completion_event) {
        ck(cudaStreamWaitEvent(m_stream, *latest_frame->yolo_completion_event, 0));
    }

    while (latest_frame->has_detections && camera_select->yolo && !latest_frame->detections_ready.load(std::memory_order_acquire)) {
        // Spin-wait for CPU-side post-processing to finish
    }

    size_t frame_size = (size_t)camera_params->width * camera_params->height;

    if (camera_params->gpu_id != display_gpu_id) {
        ck(cudaMemcpyAsync(h_p2p_copy_buffer_, latest_frame->d_image, frame_size, cudaMemcpyDeviceToHost, m_stream));
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, h_p2p_copy_buffer_, frame_size, cudaMemcpyHostToDevice, m_stream));
    } else {
        ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, latest_frame->d_image, frame_size, cudaMemcpyDeviceToDevice, m_stream));
    }

    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    }

    // --- GPU-ACCELERATED DRAWING ---
    if (latest_frame->has_detections && !latest_frame->detections.empty()) {
        // 1. Copy the raw detection data (vector of pose::Object) directly to the GPU buffer.
        ck(cudaMemcpyAsync(d_detections_for_drawing_, 
                           latest_frame->detections.data(), 
                           latest_frame->detections.size() * sizeof(pose::Object), 
                           cudaMemcpyHostToDevice, 
                           m_stream));

        // 2. Launch the kernel that reads pose::Object structs and draws rectangles.
        gpu_draw_box(
            debayer_gpu_.d_debayer,
            camera_params->width,
            camera_params->height,
            d_detections_for_drawing_,
            latest_frame->detections.size(),
            m_stream);
    }

    if (camera_select->downsample > 1) {
        output_display_size_.width = camera_params->width / camera_select->downsample;
        output_display_size_.height = camera_params->height / camera_select->downsample;
        NppiSize input_size = {static_cast<int>(camera_params->width), static_cast<int>(camera_params->height)};
        NppiRect input_roi = {0, 0, static_cast<int>(camera_params->width), static_cast<int>(camera_params->height)};
        NppiRect output_roi = {0, 0, output_display_size_.width, output_display_size_.height};
        nppiResize_8u_C4R(debayer_gpu_.d_debayer, camera_params->width * 4, input_size, input_roi,
                            d_display_resize_buffer_, output_display_size_.width * 4, output_display_size_,
                            output_roi, NPPI_INTER_SUPER);
        size_t copy_size = (size_t)output_display_size_.width * (size_t)output_display_size_.height * 4;
        ck(cudaMemcpyAsync(display_buffer_pbo_cuda_ptr_, d_display_resize_buffer_, copy_size, cudaMemcpyDeviceToDevice, m_stream));
    } else {
        size_t copy_size = (size_t)camera_params->width * (size_t)camera_params->height * 4;
        ck(cudaMemcpyAsync(display_buffer_pbo_cuda_ptr_, debayer_gpu_.d_debayer, copy_size, cudaMemcpyDeviceToDevice, m_stream));
    }

    ck(cudaStreamSynchronize(m_stream));
    
    if (latest_frame->ref_count.fetch_sub(1, std::memory_order_acq_rel) == 1) {
        if (latest_frame->gpu_direct_mode && latest_frame->camera_instance && latest_frame->camera_frame_struct) {
            EVT_CameraQueueFrame(latest_frame->camera_instance, latest_frame->camera_frame_struct);
        }
        m_recycle_queue.push(latest_frame);
    }

    return false; 
}