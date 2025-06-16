// src/acquire_frames.cpp - Final Corrected Version

#include "acquire_frames.h"
#include "NvEncoder/NvCodecUtils.h"
#include "image_processing.h"
#include "kernel.cuh"
#include <chrono>
#include <cuda_runtime.h>
#include "global.h"
#include "thread.h"
#include "opengldisplay.h"
#include "gpu_video_encoder.h"
#include "yolo_worker.h"
#include "image_writer_worker.h"
#include "cuda_context_debug.h"
#include "cuda_context_debug.h"

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
    PTPParams* ptp_params,
    INDIGOSignalBuilder* indigo_signal_builder,
    COpenGLDisplay* openGLDisplay,
    GPUVideoEncoder* gpu_encoder,
    YOLOv8Worker* yolo_worker,
    ImageWriterWorker* image_writer,
    SafeQueue<WORKER_ENTRY*>* free_entries_queue,
    SafeQueue<WORKER_ENTRY*>* recycle_queue
){
    std::cout << "Starting acquisition loop for camera " << camera_params->camera_serial << std::endl;

    CUDA_CTX_LOG("=== ACQUIRE FRAMES START ===");
    ck(cuCtxPushCurrent(cuda_context));
    dumpCudaState("Acquire frames startup");
    ck(cudaSetDevice(camera_params->gpu_id));
    CUDA_RT_LOG("Set device to " + std::to_string(camera_params->gpu_id));

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));
    CUDA_STREAM_LOG("Created acquisition stream", stream);

    FrameProcess frame_process_save;
    initalize_gpu_frame(&frame_process_save.frame_original, camera_params);
    initialize_gpu_debayer(&frame_process_save.debayer, camera_params);
    initialize_cpu_frame(&frame_process_save.frame_cpu, camera_params);
    ck(cudaMalloc((void **)&frame_process_save.d_convert, camera_params->width * camera_params->height * 3));

    CameraState camera_state;
    PTPState ptp_state;
    StopWatch w;

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
    
    while (camera_control->subscribe) {
        WORKER_ENTRY* recycled_entry = nullptr;
        while(recycle_queue->pop(recycled_entry)) {
            if (recycled_entry) {
                free_entries_queue->push(recycled_entry);
            }
        }
        
        WORKER_ENTRY* current_entry = nullptr;
        if (!free_entries_queue->pop(current_entry)) {
            usleep(100); 
            continue;
        }

        camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

        if (camera_state.camera_return == EVT_SUCCESS) {
            camera_state.frames_recd++;
            camera_state.frame_count++;
            
            CUDA_CTX_LOG("=== FRAME " + std::to_string(camera_state.frame_count) + " PROCESSING ===");
            
            // --- FIX: Centralized Frame Handling ---

            // 1. Copy the raw frame to our persistent worker buffer ONCE.
            CUDA_MEM_LOG("Copying camera frame to worker buffer", current_entry->d_image, 
                        ecam->frame_recv.bufferSize, camera_state.frame_count);
            ck(cudaMemcpyAsync(current_entry->d_image, ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize, cudaMemcpyDeviceToDevice, stream));
            VALIDATE_CUDA_OP("Camera frame copy", camera_state.frame_count);

            CUDA_SYNC_LOG("Synchronizing stream after camera copy", stream, camera_state.frame_count);
            ck(cudaStreamSynchronize(stream));
            VALIDATE_CUDA_OP("Stream synchronization", camera_state.frame_count);
            
            // 2. We can now immediately requeue the camera buffer.
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
            
            // 3. Populate the metadata for the WORKER_ENTRY.
            current_entry->width = ecam->frame_recv.size_x;
            current_entry->height = ecam->frame_recv.size_y;
            current_entry->pixelFormat = ecam->frame_recv.pixel_type;
            current_entry->timestamp = ecam->frame_recv.timestamp;
            current_entry->frame_id = camera_state.frame_count;
            current_entry->has_detections = false;

            // 4. Handle asynchronous save request (NON-BLOCKING).
            if (camera_select->frame_save_state == State_Write_New_Frame && image_writer) {
                CUDA_CTX_LOG("Processing frame save request for frame " + std::to_string(camera_state.frame_count));
                
                FrameGPU temp_frame_gpu;
                temp_frame_gpu.d_orig = current_entry->d_image;
                temp_frame_gpu.size_pic = current_entry->width * current_entry->height;

                if (camera_params->color){
                    debayer_frame_gpu(camera_params, &temp_frame_gpu, &frame_process_save.debayer);
                } else {
                    duplicate_channel_gpu(camera_params, &temp_frame_gpu, &frame_process_save.debayer);
                }
                rgba2bgr_convert(frame_process_save.d_convert, frame_process_save.debayer.d_debayer, camera_params->width, camera_params->height, stream);
                
                // Asynchronously copy data to the host buffer
                ck(cudaMemcpy2DAsync(frame_process_save.frame_cpu.frame, camera_params->width*3, frame_process_save.d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost, stream));

                // Create and record a CUDA event to mark when the copy is done
                cudaEvent_t event;
                ck(cudaEventCreate(&event));
                ck(cudaEventRecord(event, stream));

                // Create the save job with the necessary info
                ImageWriter_Entry* save_job = new ImageWriter_Entry();
                save_job->event = event;
                save_job->cpu_buffer = frame_process_save.frame_cpu.frame;
                save_job->width = camera_params->width;
                save_job->height = camera_params->height;
                save_job->file_path = camera_select->picture_save_folder + "/" + camera_params->camera_serial + "_" + camera_select->frame_save_name + "." + camera_select->frame_save_format;
                
                // Queue the job; this will no longer block!
                image_writer->PutObjectToQueueIn(save_job);
                
                camera_select->pictures_counter++;
                camera_select->frame_save_state = State_Frame_Idle;
            }

            // 5. Dispatch the WORKER_ENTRY to real-time workers.
            int dispatch_count = 0;
            if (camera_select->stream_on && openGLDisplay) dispatch_count++;
            if (camera_control->record_video && gpu_encoder) dispatch_count++;
            if (camera_select->yolo && yolo_worker) dispatch_count++;

            // Log dispatch analysis with context info
            CUDA_CTX_LOG("Frame " + std::to_string(camera_state.frame_count) + " dispatch analysis");
            dumpCudaState("Before frame dispatch", camera_state.frame_count);

            std::cout << "[ACQUIRE " << camera_params->camera_serial << "] Frame " << current_entry->frame_id 
                    << " dispatch analysis:" << std::endl;
            std::cout << "  - stream_on: " << (camera_select->stream_on ? "true" : "false") 
                    << ", openGLDisplay: " << (openGLDisplay ? "valid" : "null") << std::endl;
            std::cout << "  - record_video: " << (camera_control->record_video ? "true" : "false") 
                    << ", gpu_encoder: " << (gpu_encoder ? "valid" : "null") << std::endl;
            std::cout << "  - yolo: " << (camera_select->yolo ? "true" : "false") 
                    << ", yolo_worker: " << (yolo_worker ? "valid" : "null") << std::endl;
            std::cout << "  - Total dispatch_count: " << dispatch_count << std::endl;

            if (dispatch_count > 0) {
                current_entry->ref_count.store(dispatch_count);
                
                CUDA_CTX_LOG("Setting ref_count to " + std::to_string(dispatch_count) + " for frame " + std::to_string(camera_state.frame_count));
                
                std::cout << "[ACQUIRE " << camera_params->camera_serial << "] Dispatching frame " 
                        << current_entry->frame_id << " to " << dispatch_count << " consumers:" << std::endl;
                
                // Dispatch with detailed logging
                if (camera_select->stream_on && openGLDisplay) {
                    CUDA_CTX_LOG("Dispatching to OpenGL Display");
                    std::cout << "  -> Sending to OpenGL Display" << std::endl;
                    openGLDisplay->PutObjectToQueueIn(current_entry);
                }
                
                if (camera_control->record_video && gpu_encoder) {
                    CUDA_CTX_LOG("Dispatching to GPU Encoder - CRITICAL DISPATCH");
                    dumpCudaState("Pre-GPU-Encoder-Dispatch", camera_state.frame_count);
                    std::cout << "  -> Sending to GPU Encoder" << std::endl;
                    gpu_encoder->PutObjectToQueueIn(current_entry);
                    CUDA_CTX_LOG("GPU Encoder dispatch completed");
                }
                
                if (camera_select->yolo && yolo_worker) {
                    CUDA_CTX_LOG("Dispatching to YOLO Worker");
                    std::cout << "  -> Sending to YOLO Worker" << std::endl;
                    yolo_worker->PutObjectToQueueIn(current_entry);
                }
                
                std::cout << "[ACQUIRE " << camera_params->camera_serial << "] Frame " 
                        << current_entry->frame_id << " dispatched successfully" << std::endl;
                CUDA_CTX_LOG("Frame " + std::to_string(camera_state.frame_count) + " dispatched successfully");
                
            } else {
                std::cout << "[ACQUIRE " << camera_params->camera_serial << "] Frame " 
                        << current_entry->frame_id << " has no consumers - recycling immediately" << std::endl;
                CUDA_CTX_LOG("No consumers, recycling frame " + std::to_string(camera_state.frame_count));
                free_entries_queue->push(current_entry);
            }
        } else {
            CUDA_CTX_LOG("Camera frame acquisition failed");
            free_entries_queue->push(current_entry);
        }
    }

    // Cleanup
    CUDA_CTX_LOG("=== ACQUIRE FRAMES CLEANUP ===");
    check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"), camera_params->camera_serial.c_str());
    if (!ptp_params->network_sync) {
        try_stop_timer();
    }
    double time_diff = w.Stop();
    report_statistics(camera_params, &camera_state, time_diff);
    
    cudaFree(frame_process_save.frame_original.d_orig);
    cudaFree(frame_process_save.debayer.d_debayer);
    cudaFree(frame_process_save.d_convert);
    free(frame_process_save.frame_cpu.frame);
    
    CUDA_STREAM_LOG("Destroying acquisition stream", stream);
    cudaStreamDestroy(stream);
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
    CUDA_CTX_LOG("=== ACQUIRE FRAMES END ===");
    std::cout << "Acquire frames thread finished for camera: " << camera_params->camera_serial << std::endl;
}