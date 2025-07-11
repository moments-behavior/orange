// src/opengldisplay.h

#pragma once
#include "threadworker.h"
#include "image_processing.h"
#include "thread.h" // For SafeQueue
#include <nppi.h>
#include "common.hpp"
#include <cuda.h>

class COpenGLDisplay : public CThreadWorker<WORKER_ENTRY>
{
public:
    COpenGLDisplay(
        const char* name,
        CameraParams *camera_params,
        CameraEachSelect *camera_select,
        unsigned char *display_buffer_cuda_pbo,
        INDIGOSignalBuilder* indigo_signal_builder,
        SafeQueue<WORKER_ENTRY*>& recycle_queue);
    ~COpenGLDisplay() override;

    CameraParams* camera_params;
    CameraEachSelect* camera_select;
    unsigned char* display_buffer_pbo_cuda_ptr_;
    FrameGPU frame_original_gpu_;
    Debayer debayer_gpu_;
    INDIGOSignalBuilder* indigo_signal_builder_;

protected:
    bool WorkerFunction(WORKER_ENTRY* f) override;

private:
    unsigned char* h_p2p_copy_buffer_;
    pose::Object *d_detections_for_drawing_; 
    unsigned int *d_skeleton_for_drawing_;
    unsigned char *d_display_resize_buffer_;
    NppiSize output_display_size_;
    NppiRect input_roi_for_display_resize_;
    NppiRect output_roi_for_display_resize_;

    cudaStream_t m_stream;
    SafeQueue<WORKER_ENTRY*>& m_recycle_queue;
};