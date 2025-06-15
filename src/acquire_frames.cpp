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
#include <cuda_runtime.h>
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

    // *** 1. ADD THIS: Create a dedicated stream for this thread's operations ***
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreate(&stream));

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

    std::vector<unsigned char> intermediate_cpu_buffer(frame_size_bytes);

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
            // ADD THIS LOG:
            printf("[ACQUIRE] Recycling entry %p\n", (void*)recycled_entry);
            free_entries_queue->push(recycled_entry);
        }
    }
    // Get a free entry from the pool
    WORKER_ENTRY* current_entry = nullptr;
    if (!free_entries_queue->pop(current_entry)) {
        usleep(100);
        continue;
    }
    // ADD THIS LOG:
    printf("[ACQUIRE] Popped free entry %p with d_image %p\n", (void*)current_entry, (void*)current_entry->d_image);
    fflush(stdout);

    // Get the frame from the camera SDK
    camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

    if (camera_state.camera_return != EVT_SUCCESS && camera_state.camera_return != EVT_ERROR_TIMEDOUT)
    {
        check_camera_errors((EVT_ERROR)camera_state.camera_return, camera_params->camera_serial.c_str());
    }

    if (camera_state.camera_return == EVT_SUCCESS) {
        if (((ecam->frame_recv.frame_id) != camera_state.id_prev + 1) && (camera_state.frame_count != 0)) {
            camera_state.dropped_frames++;
        } else {
            camera_state.frames_recd++;
        }
        
        // Handle the 16-bit frame_id wrapping around from 65535 to 1
        if (ecam->frame_recv.frame_id == 65535) {
            camera_state.id_prev = 0;
        } else {
            camera_state.id_prev = ecam->frame_recv.frame_id;
        }
        // --- FPS Counter ---
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
            // --- Step 1: Safely copy the data from the SDK buffer to a CPU buffer. ---
            cudaMemcpy(intermediate_cpu_buffer.data(), imageDataSource, ecam->frame_recv.bufferSize, cudaMemcpyDeviceToHost);

            // --- Step 2: Immediately release the SDK's frame buffer. This is critical. ---
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);

            // --- Step 3: Move the data to the GPU and handle reordering if necessary. ---
            if (camera_params->need_reorder) {
                // For reordering, we need the data on the GPU in a temporary buffer first.
                CUDA_CHECK(cudaMemcpyAsync(d_temp_unscrambled_buffer, intermediate_cpu_buffer.data(), frame_size_bytes, cudaMemcpyHostToDevice, stream));
                // Now call the kernel with two valid *device* pointers.
                GSPRINT4521_Convert(current_entry->d_image, d_temp_unscrambled_buffer, camera_params->width, camera_params->height, camera_params->width, camera_params->width, 0);
            } else {
                // ADD THIS LOG before the failing line:
                printf("[ACQUIRE] Copying to d_image: %p for frame_id %llu\n", (void*)current_entry->d_image, camera_state.frame_count + 1);
                fflush(stdout);
                CUDA_CHECK(cudaMemcpyAsync(current_entry->d_image, intermediate_cpu_buffer.data(), frame_size_bytes, cudaMemcpyHostToDevice, stream));
            }
            CUDA_CHECK(cudaStreamSynchronize(stream)); 
            
            // --- Step 4: Populate metadata and dispatch to workers. ---
            current_entry->width = ecam->frame_recv.size_x;
            current_entry->height = ecam->frame_recv.size_y;
            current_entry->pixelFormat = ecam->frame_recv.pixel_type;
            current_entry->timestamp = ecam->frame_recv.timestamp;
            current_entry->has_detections = false;
            current_entry->detections.clear();
            camera_state.frame_count++;
            current_entry->frame_id = camera_state.frame_count;

            std::cout << "[ACQUIRE " << camera_params->camera_serial << "] "
                        << "Acquired frame_id: " << current_entry->frame_id << std::endl;

            bool needs_display = camera_select->stream_on && openGLDisplay != nullptr;
            bool needs_yolo = camera_select->yolo && yolo_worker_for_this_camera != nullptr;
            bool needs_record = camera_control->record_video && camera_select->record && gpu_encoder != nullptr;
            
            int dispatch_count = (needs_yolo ? 1 : 0) + (needs_display ? 1 : 0) + (needs_record ? 1 : 0);

            if (dispatch_count > 0) {
                current_entry->ref_count.store(dispatch_count);
                std::cout << "[ACQUIRE " << camera_params->camera_serial << "] "
                            << "Dispatching frame_id: " << current_entry->frame_id
                            << " with ref_count: " << dispatch_count << std::endl;

                if (needs_yolo)    yolo_worker_for_this_camera->PutObjectToQueueIn(current_entry);
                if (needs_display) openGLDisplay->PutObjectToQueueIn(current_entry);
                if (needs_record)  gpu_encoder->PutObjectToQueueIn(current_entry);
            } else {
                std::cout << "[ACQUIRE " << camera_params->camera_serial << "] "
                            << "No consumers for frame_id: " << current_entry->frame_id
                            << ". Recycling." << std::endl;
                free_entries_queue->push(current_entry);
            }
    
        } else { // Handle cases where the frame data pointer is invalid
            std::cerr << "Invalid frame data from SDK: imageDataSource is null or bufferSize is zero." << std::endl;
            free_entries_queue->push(current_entry);
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
        }
    } else { // Handle cases where EVT_CameraGetFrame did not return EVT_SUCCESS
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

    CUDA_CHECK(cudaStreamDestroy(stream));
    CUcontext popped_context;
    CU_CHECK(cuCtxPopCurrent(&popped_context));
}