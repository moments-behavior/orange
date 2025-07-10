// src/acquire_frames.cpp

#include "acquire_frames.h"
#include "nvtx_profiling.h"
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
    NVTX_RANGE("PTP_Timestamp_Check");
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
    PTPParams* ptp_params,
    INDIGOSignalBuilder* indigo_signal_builder,
    COpenGLDisplay* openGLDisplay,
    GPUVideoEncoder* gpu_encoder,
    YOLOv8Worker* yolo_worker,
    ImageWriterWorker* image_writer,
    SafeQueue<WORKER_ENTRY*>* free_entries_queue,
    SafeQueue<cudaEvent_t*>* free_events_queue,
    SafeQueue<WORKER_ENTRY*>* recycle_queue
){
    ck(cudaSetDevice(camera_params->gpu_id));
    NVTX_CAMERA("AcquireFrames_Main");
    std::cout << "Starting acquisition loop for camera " << camera_params->camera_serial << std::endl;

    {
        NVTX_RANGE("CUDA_Context_Setup");
        CUDA_CTX_LOG("=== ACQUIRE FRAMES START ===");
        dumpCudaState("Acquire frames startup");
        ck(cudaSetDevice(camera_params->gpu_id));
        CUDA_RT_LOG("Set device to " + std::to_string(camera_params->gpu_id));
    }

    cudaStream_t stream;
    FrameProcess frame_process_save;

    {
        NVTX_RANGE("Stream_and_Buffer_Init");
        ck(cudaStreamCreate(&stream));
        CUDA_STREAM_LOG("Created acquisition stream", stream);

        initalize_gpu_frame(&frame_process_save.frame_original, camera_params);
        initialize_gpu_debayer(&frame_process_save.debayer, camera_params);
        initialize_cpu_frame(&frame_process_save.frame_cpu, camera_params);
        ck(cudaMalloc((void **)&frame_process_save.d_convert, (size_t)camera_params->width * camera_params->height * 3));
    }

    CameraState camera_state{};
    PTPState ptp_state{};
    StopWatch w;
    auto last_fps_update_time = std::chrono::steady_clock::now();
    int frame_counter_for_fps = 0;

    {
        NVTX_RANGE("Camera_Initialization");
        if (camera_control->sync_camera) {
            NVTX_RANGE_PUSH("PTP_Sync_Setup");
            show_ptp_offset(&ptp_state, ecam);
            start_ptp_sync(&ptp_state, ptp_params, camera_params, ecam, 3);
            NVTX_RANGE_POP();
        }
        
        NVTX_CAMERA("Camera_Acquisition_Start");
        check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStart"), camera_params->camera_serial.c_str());
        
        if (camera_control->sync_camera) {
            NVTX_RANGE_PUSH("PTP_Countdown");
            grab_frames_after_countdown(&ptp_state, ecam);
            NVTX_RANGE_POP();
        } else {
            try_start_timer();
        }
    }
    
    w.Start();
    
    while (camera_control->subscribe) {
        NVTX_RANGE_PUSH("Frame_Processing_Loop");

        WORKER_ENTRY* recycled_entry = nullptr;
        while(recycle_queue->pop(recycled_entry)) {
            if (recycled_entry) {
                if (recycled_entry->event_ptr) {
                    free_events_queue->push(recycled_entry->event_ptr);
                }
                free_entries_queue->push(recycled_entry);
            }
        }
        
        WORKER_ENTRY* current_entry = nullptr;
        cudaEvent_t* current_event = nullptr;
        if (!free_entries_queue->pop(current_entry) || !free_events_queue->pop(current_event)) {
            if (current_entry) free_entries_queue->push(current_entry);
            if (current_event) free_events_queue->push(current_event);
            NVTX_RANGE_POP();
            usleep(100); 
            continue;
        }

        camera_state.camera_return = EVT_CameraGetFrame(&ecam->camera, &ecam->frame_recv, 1000);

        if (camera_state.camera_return == EVT_SUCCESS) {
            camera_state.frames_recd++;
            camera_state.frame_count++;
            
            cudaPointerAttributes attrs;
            bool use_direct_pointer = (cudaPointerGetAttributes(&attrs, ecam->frame_recv.imagePtr) == cudaSuccess &&
                                       attrs.type == cudaMemoryTypeDevice &&
                                       attrs.device == camera_params->gpu_id);
            
            if (use_direct_pointer) {
                current_entry->d_image = static_cast<unsigned char*>(ecam->frame_recv.imagePtr);
                current_entry->gpu_direct_mode = true;
                current_entry->owns_memory = false; 
            } else {
                ck(cudaMemcpyAsync(current_entry->d_image, ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize, cudaMemcpyDeviceToDevice, stream));
                EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
            }
            
            current_entry->event_ptr = current_event;
            ck(cudaEventRecord(*current_entry->event_ptr, stream));
            
            current_entry->width = ecam->frame_recv.size_x;
            current_entry->height = ecam->frame_recv.size_y;
            current_entry->pixelFormat = ecam->frame_recv.pixel_type;
            current_entry->timestamp = ecam->frame_recv.timestamp;
            current_entry->frame_id = camera_state.frame_count;
            current_entry->has_detections = false;
        
            if (camera_select->frame_save_state == State_Write_New_Frame && image_writer) {
                ImageWriter_Entry* save_job = new ImageWriter_Entry();
                save_job->event_ptr = current_event;
                image_writer->PutObjectToQueueIn(save_job);
            }
        
            int dispatch_count = 0;
            if (camera_select->stream_on && openGLDisplay) dispatch_count++;
            if (camera_control->record_video && gpu_encoder) dispatch_count++;
            if (camera_select->yolo && yolo_worker) dispatch_count++;
            
            if (dispatch_count > 0) {
                current_entry->ref_count.store(dispatch_count);
                
                if (camera_select->stream_on && openGLDisplay) openGLDisplay->PutObjectToQueueIn(current_entry);
                if (camera_control->record_video && gpu_encoder) gpu_encoder->PutObjectToQueueIn(current_entry);
                if (camera_select->yolo && yolo_worker) yolo_worker->PutObjectToQueueIn(current_entry);
                
                if (use_direct_pointer) {
                    current_entry->camera_buffer_ptr = ecam->frame_recv.imagePtr;
                    current_entry->camera_instance = &ecam->camera;
                    current_entry->camera_frame_struct = &ecam->frame_recv;
                }
                
            } else {
                if (use_direct_pointer) {
                    EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);
                }
                free_events_queue->push(current_event);
                free_entries_queue->push(current_entry);
            }
            
            frame_counter_for_fps++;
            auto now = std::chrono::steady_clock::now();
            std::chrono::duration<double> elapsed = now - last_fps_update_time;
            if (elapsed.count() >= 1.0) {
                streaming_fps.store(frame_counter_for_fps / elapsed.count());
                frame_counter_for_fps = 0;
                last_fps_update_time = now;
            }
        }
        NVTX_RANGE_POP();
    }

    // Cleanup
    {
        NVTX_RANGE("Cleanup_and_Shutdown");
        CUDA_CTX_LOG("=== ACQUIRE FRAMES CLEANUP ===");
        
        {
            NVTX_CAMERA("Camera_Acquisition_Stop");
            check_camera_errors(EVT_CameraExecuteCommand(&ecam->camera, "AcquisitionStop"), camera_params->camera_serial.c_str());
        }
        
        if (!ptp_params->network_sync) {
            try_stop_timer();
        }
        double time_diff = w.Stop();
        report_statistics(camera_params, &camera_state, time_diff);

        {
            NVTX_RANGE("Memory_Cleanup");
            cudaFree(frame_process_save.frame_original.d_orig);
            cudaFree(frame_process_save.debayer.d_debayer);
            cudaFree(frame_process_save.d_convert);
            free(frame_process_save.frame_cpu.frame);
        }

        CUDA_STREAM_LOG("Destroying acquisition stream", stream);
        cudaStreamDestroy(stream);

        // 1. Log that the thread is ending (while context is still active)
        CUDA_CTX_LOG("=== ACQUIRE FRAMES END ===");
        std::cout << "Acquire frames thread finished for camera: " << camera_params->camera_serial << std::endl;

        // 2. Now, pop the context as the final step
        CUcontext popped_context;
    }
}