// src/acquire_frames.cpp

#include "acquire_frames.h"
#include "NvEncoder/NvCodecUtils.h"
#include "gpu_video_encoder.h"
#include "image_processing.h"
#include "kernel.cuh"
#include "opengldisplay.h"
#include "video_capture.h"
#include "yolo_worker.h"
#include <chrono>
#include <cuda_runtime_api.h>
#include "global.h"
#include "thread.h" // Ensures SafeQueue is included
#include <cuda.h>

#include <iostream>
#include <cuda_runtime.h>

// Macro for CUDA RUNTIME API calls (functions with 'cuda' prefix)
#define CUDA_CHECK(call)                                                      \
do {                                                                          \
    cudaError_t err = call;                                                   \
    if (err != cudaSuccess) {                                                 \
        fprintf(stderr, "[CUDA FATAL] Runtime API error in %s at line %d: %s (%s)\n", \
                __FILE__, __LINE__, cudaGetErrorString(err), #call);           \
        abort();                                                              \
    }                                                                         \
} while (0)

// Macro for CUDA DRIVER API calls (functions with 'cu' prefix)
#define CU_CHECK(call)                                                        \
do {                                                                          \
    CUresult err = call;                                                      \
    if (err != CUDA_SUCCESS) {                                                \
        const char* err_str;                                                  \
        cuGetErrorString(err, &err_str);                                      \
        fprintf(stderr, "[CUDA FATAL] Driver API error in %s at line %d: %s (%s)\n", \
                __FILE__, __LINE__, err_str, #call);                          \
        abort();                                                              \
    }                                                                         \
} while (0)

// Static helper function (PTP_timestamp_checking) remains unchanged...
static inline void PTP_timestamp_checking(PTPState *ptp_state, CameraEmergent *ecam, CameraState *camera_state){
    EVT_CameraExecuteCommand(&ecam->camera, "GevTimestampControlLatch");
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueHigh", &ptp_state->ptp_time_high);
    EVT_CameraGetUInt32Param(&ecam->camera, "GevTimestampValueLow", &ptp_state->ptp_time_low);
    ptp_state->ptp_time = (((unsigned long long)(ptp_state->ptp_time_high)) << 32) | ((unsigned long long)(ptp_state->ptp_time_low));
    ptp_state->frame_ts = ecam->frame_recv.timestamp;
    if (camera_state->frame_count != 0) {
        ptp_state->ptp_time_delta = ptp_state->ptp_time - ptp_state->ptp_time_prev;
        ptp_state->ptp_time_delta_sum += ptp_state->ptp_time_delta;
        ptp_state->frame_ts_delta = ptp_state->frame_ts - ptp_state->frame_ts_prev;
        ptp_state->frame_ts_delta_sum += ptp_state->frame_ts_delta;
    }
    ptp_state->ptp_time_prev = ptp_state->ptp_time;
    ptp_state->frame_ts_prev = ptp_state->frame_ts;
}


void acquire_frames(
    CUcontext cuda_context,
    CameraEmergent *ecam,
    CameraParams *camera_params,
    CameraEachSelect* camera_select,
    CameraControl* camera_control,
    unsigned char *display_buffer,
    std::string encoder_setup,
    std::string folder_name,
    PTPParams* ptp_params,
    INDIGOSignalBuilder* indigo_signal_builder,
    YOLOv8Worker* yolo_worker_for_this_camera,
    GPUVideoEncoder* gpu_encoder,
    SafeQueue<WORKER_ENTRY*>* free_entries_queue,
    SafeQueue<WORKER_ENTRY*>* recycle_queue
){
    // Ensure the CUDA context is set for the thread
    CU_CHECK(cuCtxPushCurrent(cuda_context));

    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

    auto last_acq_time = std::chrono::steady_clock::now();
    int acq_frame_count = 0;
    double acq_fps = 0.0;

    unsigned char* d_temp_unscrambled_buffer = nullptr;
    const size_t frame_size_bytes = camera_params->width * camera_params->height;

    CUDA_CHECK(cudaSetDevice(camera_params->gpu_id));
    // --- MODIFICATION --- Replaced ck() with our ew robust macro
    CUDA_CHECK(cudaMalloc(&d_temp_unscrambled_buffer, frame_size_bytes));

    COpenGLDisplay* openGLDisplay = nullptr;
    if (camera_select->stream_on && display_buffer != nullptr) {
        std::string display_thread_name = "OpenGLDisplay_Cam_" + camera_params->camera_serial;
        openGLDisplay = new COpenGLDisplay(display_thread_name.c_str(), cuda_context, camera_params, camera_select, display_buffer, indigo_signal_builder, *recycle_queue);
        openGLDisplay->StartThread();
    }

    if (camera_control->sync_camera) {
        show_ptp_offset(&ptp_state, ecam);
        start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3);
    }
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"), camera_params->camera_serial.c_str());
    if (camera_control->sync_camera) {
        grab_frames_after_countdown(&ptp_state, ecam);
    } else {
        try_start_timer();
    }
    w.Start();

    std::cout << "Starting acquisition loop for camera " << camera_params->camera_serial
              << " (need_reorder=" << camera_params->need_reorder << ")" << std::endl;

    while (camera_control->subscribe) {
        // Reclaim recycled entries first
        WORKER_ENTRY* recycled_entry = nullptr;
        while(recycle_queue->pop(recycled_entry)) {
            if (recycled_entry) {
                free_entries_queue->push(recycled_entry);
            }
        }

        // Get a free entry from the pool
        WORKER_ENTRY* current_entry = nullptr;
        if (!free_entries_queue->pop(current_entry)) {
            usleep(100);
            continue;
        }

        camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

        if (camera_state.camera_return != EVT_SUCCESS && camera_state.camera_return != EVT_ERROR_TIMEDOUT)
        {
            check_camera_errors((EVT_ERROR)camera_state.camera_return, camera_params->camera_serial.c_str());
        }

        if (camera_state.camera_return == EVT_SUCCESS) {
            acq_frame_count++;
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - last_acq_time;
            if (elapsed.count() >= 1.0) {
                acq_fps = acq_frame_count / elapsed.count();
                acq_frame_count = 0;
                last_acq_time = now;
            }
            
            void* imageDataSource = ecam->frame_recv.imagePtr;
            if (imageDataSource != nullptr && ecam->frame_recv.bufferSize > 0) {
                // Copy data to GPU buffer
                if (camera_params->need_reorder) {
                    GSPRINT4521_Convert(current_entry->d_image, static_cast<const unsigned char*>(imageDataSource), camera_params->width, camera_params->height, camera_params->width, camera_params->width, 0);
                } else {
                    if (ecam->frame_recv.bufferSize < frame_size_bytes) {
                        fprintf(stderr, "[WARNING] Buffer size mismatch! Camera sent %zu bytes, but we expected %zu bytes.\n",
                                ecam->frame_recv.bufferSize, frame_size_bytes);
                    }
                    // --- START: MODIFICATION ---
                    // 1. Get current CUDA device for context verification
                    int current_device_id;
                    cudaGetDevice(&current_device_id);

                    // 2. Add detailed logging BEFORE the call
                    fprintf(stdout, "\n--- [DEBUG | acquire_frames] ---\n");
                    fprintf(stdout, "Attempting cudaMemcpy for camera %s\n", camera_params->camera_serial.c_str());
                    fprintf(stdout, "  > Current CUDA Device ID:      %d\n", current_device_id);
                    fprintf(stdout, "  > Destination (GPU) Pointer:   %p\n", current_entry->d_image);
                    fprintf(stdout, "  > Source (CPU) Pointer:        %p\n", imageDataSource);
                    fprintf(stdout, "  > Transfer Size (bytes):       %zu\n", frame_size_bytes);
                    fprintf(stdout, "---------------------------------\n\n");

                    // 3. Replace the original ck() with our robust CUDA_CHECK() macro
                    CUDA_CHECK(cudaMemcpy(current_entry->d_image, imageDataSource, frame_size_bytes, cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaStreamSynchronize(0)); // Ensure the copy is complete before proceeding
                    // --- END: MODIFICATION ---
                }
                
                // Populate metadata
                current_entry->width = ecam->frame_recv.size_x;
                current_entry->height = ecam->frame_recv.size_y;
                current_entry->pixelFormat = ecam->frame_recv.pixel_type;
                current_entry->timestamp = ecam->frame_recv.timestamp;
                current_entry->has_detections = false;
                current_entry->detections.clear();
                camera_state.frame_count++;
                current_entry->frame_id = camera_state.frame_count;

                bool needs_display = camera_select->stream_on && openGLDisplay != nullptr;
                bool needs_yolo = camera_select->yolo && yolo_worker_for_this_camera != nullptr;
                bool needs_record = camera_control->record_video && camera_select->record && gpu_encoder != nullptr;
                
                int dispatch_count = 0;
                if (needs_yolo) dispatch_count++;
                if (needs_display) dispatch_count++;
                if (needs_record) dispatch_count++;

                if (dispatch_count > 0)
                {
                    current_entry->ref_count.store(dispatch_count);

                    if (needs_yolo) {
                        yolo_worker_for_this_camera->PutObjectToQueueIn(current_entry);
                    }
                    if (needs_display) {
                        openGLDisplay->PutObjectToQueueIn(current_entry);
                    }
                    if (needs_record) {
                        gpu_encoder->PutObjectToQueueIn(current_entry);
                    }
                }
                else
                {
                    free_entries_queue->push(current_entry);
                }
        
            } else {
                std::cerr << "Invalid frame data: imageDataSource=" << imageDataSource
                          << ", frame_bufferSize=" << ecam->frame_recv.bufferSize << std::endl;
                free_entries_queue->push(current_entry);
            }
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
        } else {
            free_entries_queue->push(current_entry);
        }
        
        if (ptp_params->network_sync && ptp_params->network_set_stop_ptp) {
            if (ptp_state.ptp_time > ptp_params->ptp_stop_time) {
                sync_fetch_and_add(&ptp_params->ptp_stop_counter, 1);
                while (ptp_params->ptp_stop_counter != camera_params->num_cameras && camera_control->subscribe) {
                    usleep(10);
                }
                if (camera_control->subscribe) {
                    ptp_params->ptp_stop_reached = true;
                    camera_control->subscribe = false;
                }
            }
        }
    } // End of while(camera_control->subscribe)

    if (d_temp_unscrambled_buffer) {
        // --- MODIFICATION --- Replaced ck() with our new robust macro
        CUDA_CHECK(cudaFree(d_temp_unscrambled_buffer));
    }

    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"), camera_params->camera_serial.c_str());
    if (!ptp_params->network_sync) {
        try_stop_timer();
    }
    double time_diff = w.Stop();

    if (openGLDisplay) {
        openGLDisplay->StopThread();
        delete openGLDisplay;
    }

    report_statistics(camera_params, &camera_state, time_diff);
    std::cout << "Acquire frames thread finished for camera: " << camera_params->camera_serial << std::endl;

    CUcontext popped_context;
    // --- MODIFICATION --- Replaced ck() with our new robust macro
    CU_CHECK(cuCtxPopCurrent(&popped_context));
}