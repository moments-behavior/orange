// src/opengldisplay.cpp

#include "opengldisplay.h"
#include "kernel.cuh"
#include <cuda_runtime.h>
#include <iostream>

COpenGLDisplay::COpenGLDisplay(const char* name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer_cuda_pbo, INDIGOSignalBuilder* indigo_signal_builder, SafeQueue<WORKER_ENTRY*>& recycle_queue)
    : CThreadWorker(name),
      camera_params(camera_params),
      camera_select(camera_select),
      display_buffer_pbo_cuda_ptr_(display_buffer_cuda_pbo),
      indigo_signal_builder_(indigo_signal_builder),
      d_points_for_drawing_(nullptr),
      d_skeleton_for_drawing_(nullptr),
      d_display_resize_buffer_(nullptr),
      m_recycle_queue(recycle_queue) // Initialize the reference
{
    ck(cudaSetDevice(camera_params->gpu_id));
    initalize_gpu_frame(&frame_original_gpu_, camera_params);
    initialize_gpu_debayer(&debayer_gpu_, camera_params);
    ck(cudaMalloc(&d_points_for_drawing_, sizeof(float) * 2 * 17));
    ck(cudaMalloc(&d_skeleton_for_drawing_, sizeof(unsigned int) * 4 * 2));
}

COpenGLDisplay::~COpenGLDisplay()
{
    if (camera_params) {
        ck(cudaSetDevice(camera_params->gpu_id));
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

    // Copy and process the frame data...
    ck(cudaMemcpy(frame_original_gpu_.d_orig, f->imageData.data(), f->bufferSize, cudaMemcpyHostToDevice));
    if (camera_params->color){
        debayer_frame_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    } else {
        duplicate_channel_gpu(camera_params, &frame_original_gpu_, &debayer_gpu_);
    }
    ck(cudaMemcpy(display_buffer_pbo_cuda_ptr_, debayer_gpu_.d_debayer, camera_params->width * camera_params->height * 4, cudaMemcpyDeviceToDevice));
    
    // --- FIX: Return the used entry to the central recycling queue ---
    m_recycle_queue.push(f);
    
    // Return false, as this is the end of the line for this entry
    return false; 
}