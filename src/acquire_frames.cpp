// src/acquire_frames.cpp - Modernized for true parallelism

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
#include "image_writer_worker.h" // Include the new worker

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
    ImageWriterWorker* image_writer, // Now receives the image writer
    SafeQueue<WORKER_ENTRY*>* free_entries_queue,
    SafeQueue<WORKER_ENTRY*>* recycle_queue
){
    std::cout << "Starting acquisition loop for camera " << camera_params->camera_serial << std::endl;

    ck(cuCtxPushCurrent(cuda_context));
    ck(cudaSetDevice(camera_params->gpu_id));

    cudaStream_t stream;
    ck(cudaStreamCreate(&stream));

    // This struct holds temporary buffers ONLY for the image saving process
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
            
            // --- Asynchronous Save Logic (Non-Blocking) ---
            if (camera_select->frame_save_state == State_Write_New_Frame && image_writer) {
                ck(cudaMemcpyAsync(frame_process_save.frame_original.d_orig, ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize, cudaMemcpyDeviceToDevice, stream));
                if (camera_params->color){
                    debayer_frame_gpu(camera_params, &frame_process_save.frame_original, &frame_process_save.debayer);
                } else {
                    duplicate_channel_gpu(camera_params, &frame_process_save.frame_original, &frame_process_save.debayer);
                }      
                rgba2bgr_convert(frame_process_save.d_convert, frame_process_save.debayer.d_debayer, camera_params->width, camera_params->height, stream);                
                ck(cudaMemcpy2D(frame_process_save.frame_cpu.frame, camera_params->width*3, frame_process_save.d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost));
                ck(cudaStreamSynchronize(stream));

                ImageWriter_Entry* save_job = new ImageWriter_Entry();
                save_job->image = cv::Mat(camera_params->height, camera_params->width, CV_8UC3, frame_process_save.frame_cpu.frame).clone();
                save_job->file_path = camera_select->picture_save_folder + "/" + camera_params->camera_serial + "_" + camera_select->frame_save_name + "." + camera_select->frame_save_format;
                
                image_writer->PutObjectToQueueIn(save_job);
                
                camera_select->pictures_counter++;
                camera_select->frame_save_state = State_Frame_Idle;
            }

            // Immediately requeue the camera's internal buffer for the next frame
            EVT_CameraQueueFrame(&ecam->camera, &ecam->frame_recv);

            // --- Real-time Dispatch Logic ---
            ck(cudaMemcpyAsync(current_entry->d_image, ecam->frame_recv.imagePtr, ecam->frame_recv.bufferSize, cudaMemcpyDeviceToDevice, stream));
            ck(cudaStreamSynchronize(stream));

            current_entry->width = ecam->frame_recv.size_x;
            current_entry->height = ecam->frame_recv.size_y;
            current_entry->pixelFormat = ecam->frame_recv.pixel_type;
            current_entry->timestamp = ecam->frame_recv.timestamp;
            current_entry->frame_id = camera_state.frame_count;
            current_entry->has_detections = false;
            
            int dispatch_count = 0;
            if (camera_select->stream_on && openGLDisplay) dispatch_count++;
            if (camera_select->record && gpu_encoder) dispatch_count++;
            if (camera_select->yolo && yolo_worker) dispatch_count++;

            if (dispatch_count > 0) {
                current_entry->ref_count.store(dispatch_count);

                if (camera_select->stream_on && openGLDisplay) openGLDisplay->PutObjectToQueueIn(current_entry);
                if (camera_select->record && gpu_encoder) gpu_encoder->PutObjectToQueueIn(current_entry);
                if (camera_select->yolo && yolo_worker) yolo_worker->PutObjectToQueueIn(current_entry);
            } else {
                free_entries_queue->push(current_entry);
            }
        } else {
            free_entries_queue->push(current_entry);
        }
    }

    // Cleanup
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
    
    cudaStreamDestroy(stream);
    CUcontext popped_context;
    ck(cuCtxPopCurrent(&popped_context));
    std::cout << "Acquire frames thread finished for camera: " << camera_params->camera_serial << std::endl;
}