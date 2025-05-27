
#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "opengldisplay.h"
#include <cuda_runtime_api.h>
#include "global.h"

COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer, INDIGOSignalBuilder* indigo_signal_builder)
    : CThreadWorker(name), camera_params(camera_params), camera_select(camera_select), display_buffer(display_buffer), indigo_signal_builder(indigo_signal_builder)
{
    input_image_size.width = camera_params->width;
    input_image_size.height = camera_params->height;
    input_image_roi.x = 0;
    input_image_roi.y = 0;
    input_image_roi.width = camera_params->width;
    input_image_roi.height = camera_params->height;

    output_image_size.width = int(camera_params->width / camera_select->downsample);
    output_image_size.height = int(camera_params->height / camera_select->downsample);

    output_image_roi.x = 0;
    output_image_roi.y = 0;
    output_image_roi.width = output_image_size.width;
    output_image_roi.height = output_image_size.height;

    if (camera_select->downsample != 1) {
        cudaMalloc((void **)&d_resize, output_image_size.width * output_image_size.height * 4);
    }

    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = WORK_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++)
    {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
}

COpenGLDisplay::~COpenGLDisplay()
{
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (camera_select->yolo) {
        delete yolov8;
    }
}

void COpenGLDisplay::ThreadRunning()
{
    ck(cudaSetDevice(camera_params->gpu_id));
    // innitialization
    initalize_gpu_frame(&frame_original, camera_params);
    initialize_gpu_debayer(&debayer, camera_params);
    initialize_cpu_frame(&frame_cpu, camera_params);

    ck(cudaMalloc((void **)&d_convert, camera_params->width * camera_params->height * 3)); 
    
    unsigned int skeleton[8] = {0, 1, 1, 2, 2, 3, 3, 0}; // box
    if (camera_select->yolo) {
        printf("YOLO initialization...\n");

        const std::string engine_file_path = camera_select->yolo_model;
        yolov8 = new YOLOv8(engine_file_path, camera_params->width, camera_params->height);
        yolov8->make_pipe(true);

        cudaMalloc((void **)&d_points, sizeof(float) * 8);
        cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
        CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice));
    }

    std::vector<Object> objs;
    std::vector<Object> objs_last_frame; // think about better way that scales with frame
    
    using clock = std::chrono::steady_clock;
   

    int frameCount = 0;
    auto lastFPSUpdate = clock::now();

    while(IsMachineOn())
    {
        auto frameStart = clock::now();
        std::chrono::duration<double, std::milli> targetFrameDuration(1000.0 / streaming_target_fps.load());
        void* f = GetObjectFromQueueIn();
        if(f) {
            WORKER_ENTRY entry = *(WORKER_ENTRY*)f;
            PutObjectToQueueOut(f);
            
            // copy frame from cpu to gpu
            ck(cudaMemcpy2D(frame_original.d_orig, camera_params->width, entry.imagePtr, camera_params->width, camera_params->width, camera_params->height, cudaMemcpyHostToDevice));

            if (camera_params->color){
                debayer_frame_gpu(camera_params, &frame_original, &debayer);
            } else {
                duplicate_channel_gpu(camera_params, &frame_original, &debayer);
            }

            if (camera_select->yolo) {
                rgba2rgb_convert(d_convert, debayer.d_debayer, camera_params->width, camera_params->height, 0);
                yolov8->preprocess_gpu(d_convert);
                yolov8->infer();
                yolov8->postprocess(objs);
                yolov8->copy_keypoints_gpu(d_points, objs);

                if (objs.size() > 0) {
                    // std::cout << objs[0].rect.x << ", " << objs[0].rect.y << std::endl;
                    // f32 bbox_center_x = objs[0].rect.x + objs[0].rect.width / 2.0;
                    // std::cout << bbox_center_x << std::endl;
                    // if (objs[0].rect.x < 2260.41 && objs[0].rect.x < objs_last_frame[0].rect.x) {
                    // if (objs[0].rect.x < 2500.0 && objs[0].rect.x > 2100.0) {
                    if (objs[0].rect.x < 2600.0 && objs[0].rect.x > 2100.0) { // trigger earlier
                        // std::cout << "trigger ball drop" << std::endl;
                        if (indigo_signal_builder->indigo_connection != NULL) {
                            send_indigo_message(indigo_signal_builder->server, indigo_signal_builder->builder, indigo_signal_builder->indigo_connection, FetchGame::SignalType_INDIGO_TRIAL_TRIGGER);
                        }
                    }
                    objs_last_frame.push_back(objs[0]);
                } else {
                    objs_last_frame.clear();
                }
                    
                gpu_draw_rat_pose(debayer.d_debayer, camera_params->width, camera_params->height, d_points, d_skeleton, yolov8->stream, 4);
            }


            if (camera_select->downsample != 1) {
                const NppStatus npp_result = nppiResize_8u_C4R(debayer.d_debayer, 
                    camera_params->width * sizeof(uchar4), 
                    input_image_size, 
                    input_image_roi, 
                    (Npp8u*)d_resize,
                    output_image_size.width * sizeof(uchar4),
                    output_image_size,
                    output_image_roi,
                    NPPI_INTER_SUPER);
                if (npp_result != NPP_SUCCESS) {
                    std::cerr << "Error executing resize in display -- code: " << npp_result << std::endl;
                }
                ck(cudaMemcpy2D(display_buffer,
                    output_image_size.width * 4,
                    d_resize,
                    output_image_size.width * 4,
                    output_image_size.width * 4,
                    output_image_size.height,
                    cudaMemcpyDeviceToDevice));
 
            } else {
               ck(cudaMemcpy2D(display_buffer,
                output_image_size.width * 4,
                debayer.d_debayer,
                output_image_size.width * 4,
                output_image_size.width * 4,
                output_image_size.height,
                cudaMemcpyDeviceToDevice));
            }
            cudaDeviceSynchronize();

            // if (camera_select->frame_save_state==State_Write_New_Frame) {
            //     rgba2bgr_convert(d_convert, debayer.d_debayer, camera_params->width, camera_params->height, 0);
                
            //     // copy frame back to cpu, and then save
            //     cudaMemcpy2D(frame_cpu.frame, camera_params->width*3, d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost);
            //     cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, frame_cpu.frame).reshape(3, camera_params->height);
                
            //     std::string image_name = camera_select->picture_save_folder + "/" + camera_params->camera_serial + "_" + camera_select->frame_save_name + "." + camera_select->frame_save_format;
            //     std::cout << image_name << std::endl;
            //     cv::imwrite(image_name, view);
            //     camera_select->pictures_counter++;
            //     camera_select->frame_save_state = State_Frame_Idle;
            // }          
        }
        // Count frame for FPS
        frameCount++;
        auto now = clock::now();
        std::chrono::duration<double> timeSinceLastFPSUpdate = now - lastFPSUpdate;
        if (timeSinceLastFPSUpdate.count() >= 1.0) {
            streaming_fps.store(frameCount / timeSinceLastFPSUpdate.count());
            frameCount = 0;
            lastFPSUpdate = now;
        }
        // Frame duration (GPU time included)
        std::chrono::duration<double, std::milli> frameDuration = now - frameStart;
        if (frameDuration < targetFrameDuration) {
            std::this_thread::sleep_for(targetFrameDuration - frameDuration);
        }
    }
}


bool COpenGLDisplay::PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id)
{
    WORKER_ENTRY *entriesOut[WORK_ENTRIES_MAX]; // entris got out from saver thread, their frames should be returned to driver queue.
    int entriesOutCount = WORK_ENTRIES_MAX;
    GetObjectsFromQueueOut((void **)entriesOut, &entriesOutCount);
    if (entriesOutCount)
    { // return the frames to driver, and put entries back to frameSaveEntriesFreeQueue
        // printf("++++++++++++++++++++++++ %s %s %d get WORKER_ENTRY from out entriesOutCount: %d\n", __FILE__, __FUNCTION__, __LINE__, entriesOutCount);
        for (int j = 0; j < entriesOutCount; j++)
        {
            workerEntriesFreeQueue[workerEntriesFreeQueueCount] = entriesOut[j];
            workerEntriesFreeQueueCount++;
        }
    }

    // get the free entry if there is one and put in to QueueIn, otherwise EVT_CameraQueueFrame.
    if (workerEntriesFreeQueueCount)
    {
        // printf("++++++++++++++++++++++++ %s %s %d put WORKER_ENTRY to in workerEntriesFreeQueueCount: %d\n", __FILE__, __FUNCTION__, __LINE__, workerEntriesFreeQueueCount);
        WORKER_ENTRY *entry = workerEntriesFreeQueue[workerEntriesFreeQueueCount - 1];
        workerEntriesFreeQueueCount--;
        entry->imagePtr = imagePtr;
        entry->bufferSize = bufferSize;
        entry->width = width;
        entry->height = height;
        entry->pixelFormat = pixelFormat;
        entry->timestamp = timestamp;
        entry->frame_id = frame_id;
        PutObjectToQueueIn(entry);
        return true;
    }
    return false;
}
