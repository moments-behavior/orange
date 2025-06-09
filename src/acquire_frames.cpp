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

#define ACQUIRE_WORK_ENTRIES_MAX 20

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
    CameraEmergent *ecam,
    CameraParams *camera_params,
    CameraEachSelect* camera_select,
    CameraControl* camera_control,
    unsigned char *display_buffer,
    std::string encoder_setup,
    std::string folder_name,
    PTPParams* ptp_params,
    INDIGOSignalBuilder* indigo_signal_builder,
    YOLOv8Worker* yolo_worker_for_this_camera
){
    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

    SafeQueue<WORKER_ENTRY*> recycle_queue;

    unsigned char* d_temp_unscrambled_buffer = nullptr;
    const size_t frame_size_bytes = camera_params->width * camera_params->height;
    
    ck(cudaSetDevice(camera_params->gpu_id));
    ck(cudaMalloc(&d_temp_unscrambled_buffer, frame_size_bytes));

    COpenGLDisplay* openGLDisplay = nullptr;
    if (camera_select->stream_on && display_buffer != nullptr) {
        std::string display_thread_name = "OpenGLDisplay_Cam_" + camera_params->camera_serial;
        // --- FIX: Pass the recycle_queue to the constructor ---
        openGLDisplay = new COpenGLDisplay(display_thread_name.c_str(), camera_params, camera_select, display_buffer, indigo_signal_builder, recycle_queue);
        openGLDisplay->StartThread();
    }

    GPUVideoEncoder* gpu_encoder = nullptr;
    bool encoder_ready_signal = false;
    if (camera_control->record_video && camera_select->record) {
        std::string encoder_thread_name = "GPUEncoder_Cam_" + camera_params->camera_serial;
        gpu_encoder = new GPUVideoEncoder(
            encoder_thread_name.c_str(),
            camera_params,
            encoder_setup,
            folder_name,
            &encoder_ready_signal,
            recycle_queue);
        gpu_encoder->StartThread();
        while(!encoder_ready_signal && camera_control->subscribe) {
            usleep(10);
        }
    }

    WORKER_ENTRY worker_entry_pool[ACQUIRE_WORK_ENTRIES_MAX];
    SafeQueue<WORKER_ENTRY*> free_entries_queue;
    
    for(int i = 0; i < ACQUIRE_WORK_ENTRIES_MAX; ++i) {
        worker_entry_pool[i].imageData.resize(frame_size_bytes);
        worker_entry_pool[i].detections.clear();
        worker_entry_pool[i].has_detections = false;
        free_entries_queue.push(&worker_entry_pool[i]);
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
        // Reclaim recycled entries from the recycle queue
        WORKER_ENTRY* recycled_entry = nullptr;
        while(recycle_queue.pop(recycled_entry)) {
            if (recycled_entry) {
                // Reset the entry before reusing it
                std::cout << "Recycled entry " << recycled_entry << std::endl;
                free_entries_queue.push(recycled_entry);
            }
        }

        // Step 1: Get a free entry from your safe pool
    WORKER_ENTRY* current_entry = nullptr;
    if (!free_entries_queue.pop(current_entry)) {
        usleep(100);
        continue;
    }
    std::cout << "Acquired free entry " << current_entry << std::endl;

    // Step 2: Get the frame from the camera SDK
    camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

    if (camera_state.camera_return == EVT_SUCCESS) {
        void* imageDataSource = ecam->frame_recv.imagePtr;
        size_t frame_bufferSize = ecam->frame_recv.bufferSize;

        if (imageDataSource != nullptr && frame_bufferSize > 0) {

            std::cout << "Cam " << camera_params->camera_serial << ": Got frame " << ecam->frame_recv.frame_id 
                    << ", Addr: " << imageDataSource << ", Size: " << frame_bufferSize << std::endl;

        

            if (camera_params->need_reorder) {
                // CASE 1: The camera data is scrambled.
                // Use the custom kernel to copy and unscramble the data from the
                // volatile hardware buffer to our stable temporary GPU buffer.
                GSPRINT4521_Convert(d_temp_unscrambled_buffer, static_cast<const unsigned char*>(imageDataSource), camera_params->width, camera_params->height, camera_params->width, camera_params->width, 0);
            } else {
                // CASE 2: The camera data is linear, but still in a volatile zero-copy buffer.
                // Use a safe CUDA Host-to-Device copy to snapshot the data
                // into our stable temporary GPU buffer.
                ck(cudaMemcpy(d_temp_unscrambled_buffer, imageDataSource, frame_size_bytes, cudaMemcpyHostToDevice));
            }
            
            // --- This step is now safe for all cameras ---
            // Copy the stable, corrected image from our GPU buffer back to the CPU buffer in the WORKER_ENTRY.
            ck(cudaMemcpy(current_entry->imageData.data(), d_temp_unscrambled_buffer, frame_size_bytes, cudaMemcpyDeviceToHost));
            current_entry->bufferSize = frame_size_bytes;
            
            // The rest of the logic remains the same...

            // Fill metadata
            current_entry->width = ecam->frame_recv.size_x;
            current_entry->height = ecam->frame_recv.size_y;
            current_entry->pixelFormat = ecam->frame_recv.pixel_type;
            current_entry->timestamp = ecam->frame_recv.timestamp;
            
            // Update frame counting
            camera_state.frame_count++;
            if (((ecam->frame_recv.frame_id) != camera_state.id_prev + 1) && 
                (camera_state.id_prev != 65535 && camera_state.frame_count > 1)) {
                camera_state.dropped_frames++;
            } else {
                camera_state.frames_recd++;
            }
            camera_state.id_prev = ecam->frame_recv.frame_id;
            current_entry->frame_id = camera_state.frame_count;

            struct timespec ts_rt1;
            clock_gettime(CLOCK_REALTIME, &ts_rt1);
            current_entry->timestamp_sys = (ts_rt1.tv_sec * 1000000000LL) + ts_rt1.tv_nsec;

            current_entry->has_detections = false;
            current_entry->detections.clear();

            // Dispatch to workers
            bool entry_dispatched = false;
            
            if (camera_control->record_video && camera_select->record && gpu_encoder) {
                gpu_encoder->PutObjectToQueueIn(current_entry);
                entry_dispatched = true;
            }

            if (camera_select->yolo && yolo_worker_for_this_camera != nullptr) {
                yolo_worker_for_this_camera->PutObjectToQueueIn(current_entry);
                entry_dispatched = true;
            } else if (camera_select->stream_on && openGLDisplay != nullptr) {
                openGLDisplay->PutObjectToQueueIn(current_entry);
                entry_dispatched = true;
            }
            
            if (!entry_dispatched) {
                free_entries_queue.push(current_entry);
            }

            } else {
                std::cerr << "Invalid frame data: imageDataSource=" << imageDataSource 
                        << ", frame_bufferSize=" << frame_bufferSize << std::endl;
                free_entries_queue.push(current_entry);
            }
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
        } else {
            if (camera_state.camera_return != EVT_ERROR_TIMEDOUT) {
                camera_state.dropped_frames++;
                std::cerr << "EVT_CameraGetFrame Error: " << camera_state.camera_return
                          << ", camera serial: " << camera_params->camera_serial << std::endl;
            }
            free_entries_queue.push(current_entry);
        }

        WORKER_ENTRY* processed_frame_ptr = nullptr;
        if (camera_select->yolo && yolo_worker_for_this_camera) {
            processed_frame_ptr = yolo_worker_for_this_camera->GetObjectFromQueueOut();
            if (processed_frame_ptr) {
                if (camera_select->stream_on && openGLDisplay) {
                    openGLDisplay->PutObjectToQueueIn(processed_frame_ptr);
                } else {
                    free_entries_queue.push(processed_frame_ptr);
                }
            }
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
    if (gpu_encoder) {
        gpu_encoder->StopThread();
        delete gpu_encoder;
    }

    report_statistics(camera_params, &camera_state, time_diff);
    std::cout << "Acquire frames thread finished for camera: " << camera_params->camera_serial << std::endl;
}