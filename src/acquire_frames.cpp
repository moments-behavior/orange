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
        openGLDisplay = new COpenGLDisplay(display_thread_name.c_str(), camera_params, camera_select, display_buffer, indigo_signal_builder, *recycle_queue);
        openGLDisplay->StartThread();
    }

    // *** The local GPUVideoEncoder creation block has been REMOVED from here. ***

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
        while(recycle_queue->pop(recycled_entry)) {
            if (recycled_entry) {
                free_entries_queue->push(recycled_entry);
            }
        }

        // Step 1: Get a free entry from your safe pool
        WORKER_ENTRY* current_entry = nullptr;
        if (!free_entries_queue->pop(current_entry)) {
            usleep(100);
            continue;
        }

        // Step 2: Get the frame from the camera SDK
        camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

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
            size_t frame_bufferSize = ecam->frame_recv.bufferSize;

            if (imageDataSource != nullptr && frame_bufferSize > 0) {
                if (camera_params->need_reorder) {
                    GSPRINT4521_Convert(current_entry->d_image, static_cast<const unsigned char*>(imageDataSource), camera_params->width, camera_params->height, camera_params->width, camera_params->width, 0);
                } else {
                    ck(cudaMemcpy(current_entry->d_image, imageDataSource, frame_size_bytes, cudaMemcpyHostToDevice));
                }
                
                current_entry->width = ecam->frame_recv.size_x;
                current_entry->height = ecam->frame_recv.size_y;
                current_entry->pixelFormat = ecam->frame_recv.pixel_type;
                current_entry->timestamp = ecam->frame_recv.timestamp;
                
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

                // Identify which workers need this frame
                bool needs_display = camera_select->stream_on && openGLDisplay != nullptr;
                bool needs_yolo = camera_select->yolo && yolo_worker_for_this_camera != nullptr;
                bool needs_record = camera_control->record_video && camera_select->record && gpu_encoder;

                // Keep track of which entries we dispatch
                WORKER_ENTRY* display_entry = nullptr;
                WORKER_ENTRY* yolo_entry = nullptr;
                WORKER_ENTRY* record_entry = nullptr;

                // The first worker can reuse the initial entry we already have.
                if (needs_yolo) {
                    yolo_entry = current_entry;
                } else if (needs_display) {
                    display_entry = current_entry;
                } else if (needs_record) {
                    record_entry = current_entry;
                } else {
                    // No worker needs this frame, so recycle it immediately.
                    free_entries_queue->push(current_entry);
                    current_entry = nullptr;
                }

                // For any other workers that need the frame, get a new entry and copy the GPU data.
                if (yolo_entry) {
                    if (needs_display) {
                        if (free_entries_queue->pop(display_entry)) {
                            ck(cudaMemcpy(display_entry->d_image, yolo_entry->d_image, frame_size_bytes, cudaMemcpyDeviceToDevice));
                            display_entry->frame_id = yolo_entry->frame_id;
                            display_entry->timestamp = yolo_entry->timestamp;
                            display_entry->timestamp_sys = yolo_entry->timestamp_sys;
                        }
                    }
                    if (needs_record) {
                        if (free_entries_queue->pop(record_entry)) {
                            ck(cudaMemcpy(record_entry->d_image, yolo_entry->d_image, frame_size_bytes, cudaMemcpyDeviceToDevice));
                            record_entry->frame_id = yolo_entry->frame_id;
                            record_entry->timestamp = yolo_entry->timestamp;
                            record_entry->timestamp_sys = yolo_entry->timestamp_sys;
                        }
                    }
                } else if (display_entry) {
                    if (needs_record) {
                        if (free_entries_queue->pop(record_entry)) {
                            ck(cudaMemcpy(record_entry->d_image, display_entry->d_image, frame_size_bytes, cudaMemcpyDeviceToDevice));
                            record_entry->frame_id = display_entry->frame_id;
                            record_entry->timestamp = display_entry->timestamp;
                            record_entry->timestamp_sys = display_entry->timestamp_sys;
                        }
                    }
                }

                // Now, dispatch the unique entries to their respective workers.
                if (yolo_entry) {
                    yolo_worker_for_this_camera->PutObjectToQueueIn(yolo_entry);
                }
                if (display_entry) {
                    openGLDisplay->PutObjectToQueueIn(display_entry);
                }
                if (record_entry) {
                    gpu_encoder->PutObjectToQueueIn(record_entry);
                }

            } else {
                std::cerr << "Invalid frame data: imageDataSource=" << imageDataSource
                          << ", frame_bufferSize=" << frame_bufferSize << std::endl;
                free_entries_queue->push(current_entry);
            }
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
        } else {
            if (camera_state.camera_return != EVT_ERROR_TIMEDOUT) {
                camera_state.dropped_frames++;
                std::cerr << "EVT_CameraGetFrame Error: " << camera_state.camera_return
                          << ", camera serial: " << camera_params->camera_serial << std::endl;
            }
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
}