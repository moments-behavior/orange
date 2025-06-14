// src/opengldisplay.cpp

#include "opengldisplay.h"
#include "kernel.cuh"
#include "shaman.h"
#include <cuda_runtime.h>
#include <iostream>
#include <cuda.h>

COpenGLDisplay::COpenGLDisplay(const char* name, CUcontext cuda_context, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer_cuda_pbo, INDIGOSignalBuilder* indigo_signal_builder, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      m_cuContext(cuda_context),
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
    ck(cuCtxPushCurrent(m_cuContext)); // Ensure the CUDA context is active for this thread
    ck(cudaSetDevice(camera_params->gpu_id));
    ck(cudaStreamCreate(&m_stream));

    initalize_gpu_frame(&frame_original_gpu_, camera_params);
    initialize_gpu_debayer(&debayer_gpu_, camera_params);
    // This line will now compile correctly
    ck(cudaMalloc(&d_points_for_drawing_, sizeof(float) * 4 * 2 * shaman::MAX_OBJECTS)); 
    ck(cudaMalloc(&d_skeleton_for_drawing_, sizeof(unsigned int) * 4 * 2));
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
}

COpenGLDisplay::~COpenGLDisplay()
{
    if (camera_params) {
        ck(cudaSetDevice(camera_params->gpu_id));
    }
    
    // --- FIX: Destroy the CUDA stream ---
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

    ck(cuCtxPushCurrent(m_cuContext)); // Ensure the CUDA context is active for this thread
    
    ck(cudaSetDevice(camera_params->gpu_id));

    size_t buffer_size = static_cast<size_t>(f->width) * static_cast<size_t>(f->height);
    // Use the member stream for async operations
    ck(cudaMemcpyAsync(frame_original_gpu_.d_orig, f->d_image, buffer_size, cudaMemcpyDeviceToDevice, m_stream));
    
    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    }

    if (f->has_detections) {
        std::vector<float> h_points;
        h_points.reserve(f->detections.size() * 4 * 2);

        for(const auto& obj : f->detections) {
            // ... (point generation logic remains the same)
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
            
            // --- FIX: Pass the member stream to the drawing kernel ---
            gpu_draw_box(debayer_gpu_.d_debayer, camera_params->width, camera_params->height, d_points_for_drawing_, m_stream);
        }
    }

    ck(cudaMemcpyAsync(display_buffer_pbo_cuda_ptr_, debayer_gpu_.d_debayer, camera_params->width * camera_params->height * 4, cudaMemcpyDeviceToDevice, m_stream));
    
    // Synchronize the stream to ensure all operations are complete before recycling the buffer
    ck(cudaStreamSynchronize(m_stream));
    
    m_recycle_queue.push(f);
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
    return false; 
}