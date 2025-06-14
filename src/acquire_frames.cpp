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
    ck(cuCtxPushCurrent(cuda_context));

    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

    auto last_acq_time = std::chrono::steady_clock::now();
    int acq_frame_count = 0;
    double acq_fps = 0.0;

    unsigned char* d_temp_unscrambled_buffer = nullptr;
    const size_t frame_size_bytes = camera_params->width * camera_params->height;

    ck(cudaSetDevice(camera_params->gpu_id));
    ck(cudaMalloc(&d_temp_unscrambled_buffer, frame_size_bytes));

    COpenGLDisplay* openGLDisplay = nullptr;
    if (camera_select->stream_on && display_buffer != nullptr) {
        std::string display_thread_name = "OpenGLDisplay_Cam_" + camera_params->camera_serial;
        // FIX: Pass the cuda_context to the constructor
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

        // --- START: MODIFIED ERROR HANDLING ---
        camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

        // A timeout is not a fatal error, just continue the loop.
        // Any other error, however, should be treated as fatal to stop the error spam.
        if (camera_state.camera_return != EVT_SUCCESS && camera_state.camera_return != EVT_ERROR_TIMEDOUT)
        {
            // The existing macro throws an exception, which will stop the program
            // and allow you to see the first error message clearly in your debugger.
            check_camera_errors((EVT_ERROR)camera_state.camera_return, camera_params->camera_serial.c_str());
        }
        // --- END: MODIFIED ERROR HANDLING ---

        if (camera_state.camera_return == EVT_SUCCESS) {
            acq_frame_count++;
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - last_acq_time;
            if (elapsed.count() >= 1.0) {
                acq_fps = acq_frame_count / elapsed.count();
                std::cout << camera_params->camera_serial << " Acquisition FPS: " << acq_fps << std::endl;
                acq_frame_count = 0;
                last_acq_time = now;
            }
            
            void* imageDataSource = ecam->frame_recv.imagePtr;
            if (imageDataSource != nullptr && ecam->frame_recv.bufferSize > 0) {
                // Copy data to GPU buffer
                if (camera_params->need_reorder) {
                    GSPRINT4521_Convert(current_entry->d_image, static_cast<const unsigned char*>(imageDataSource), camera_params->width, camera_params->height, camera_params->width, camera_params->width, 0);
                } else {
                    ck(cudaMemcpy(current_entry->d_image, imageDataSource, frame_size_bytes, cudaMemcpyHostToDevice));
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
                if (needs_yolo) {
                    dispatch_count++;
                } else if (needs_display) {
                    dispatch_count++;
                }

                if (needs_record) {
                    dispatch_count++;
                }

                if (dispatch_count > 0)
                {
                    current_entry->ref_count.store(dispatch_count);

                    if (needs_yolo) {
                        yolo_worker_for_this_camera->PutObjectToQueueIn(current_entry);
                    } else if (needs_display) {
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
            // This now only handles EVT_ERROR_TIMEDOUT
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
        ck(cudaFree(d_temp_unscrambled_buffer));
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
    // Pop the context before the thread exits
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
}