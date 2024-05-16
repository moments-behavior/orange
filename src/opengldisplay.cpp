#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "opengldisplay.h"
#include <cuda_runtime_api.h>

COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer, CBOTSignalBuilder* cbot_signal_builder)
    : CThreadWorker(name), camera_params(camera_params), camera_select(camera_select), display_buffer(display_buffer), cbot_signal_builder(cbot_signal_builder)
{
    memset(workerEntries, 0, sizeof(workerEntries));
    workerEntriesFreeQueueCount = WORK_ENTRIES_MAX;
    for (int i = 0; i < workerEntriesFreeQueueCount; i++)
    {
        workerEntriesFreeQueue[i] = &workerEntries[i];
    }
}

COpenGLDisplay::~COpenGLDisplay()
{
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

        const std::string engine_file_path{"/home/ratan/detect/shape_0.engine"};
        yolov8 = new YOLOv8(engine_file_path);
        yolov8->make_pipe(true);

        cudaMalloc((void **)&d_points, sizeof(float) * 10);
        cudaMalloc((void **)&d_skeleton, sizeof(unsigned int) * 8);
        CHECK(cudaMemcpy(d_skeleton, skeleton, sizeof(unsigned int) * 8, cudaMemcpyHostToDevice));
    }

        
    std::vector<Object> objs;
    std::vector<Object> objs_last_frame; // think about better way that scales with frame
 
    while(IsMachineOn())
    {
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
                

                for (auto obj : objs) {
                    std::vector<Object> tmp_obj ;
                    tmp_obj.push_back(obj);
                    yolov8->copy_keypoints_gpu(d_points, tmp_obj);
                    gpu_draw_cicles(debayer.d_debayer, 3208, 2200, d_points, 5, yolov8->stream);
                }

                // gpu_draw_box(debayer.d_debayer, 3208, 2200 , d_points, yolov8->stream);


                if (objs.size() > 0) {
                    
                    
                    // std::cout << objs[0].rect.x << ", " << objs[0].rect.y << std::endl;
                    // f32 bbox_center_x = objs[0].rect.x + objs[0].rect.width / 2.0;
                    // std::cout << bbox_center_x << std::endl;
                    // if (objs[0].rect.x < 2260.41 && objs[0].rect.x < objs_last_frame[0].rect.x) {
                    // if (objs[0].rect.x < 2500.0 && objs[0].rect.x > 2100.0) {
                    if (objs[0].rect.x < 2600.0 && objs[0].rect.x > 2100.0) { // trigger earlier
                        // send a trigger signal to cbot
                        std::cout << "trigger ball drop" << std::endl;
                        // send_cbot_ball_drop_trigger_signal(cbot_signal_builder->server, cbot_signal_builder->builder, cbot_signal_builder->cbot_connection);
                    }
                    objs_last_frame.push_back(objs[0]);
                } else {
                    objs_last_frame.clear();
                }
                    
                // gpu_draw_rat_pose(debayer.d_debayer, 3208, 2200, d_points, d_skeleton, yolov8->stream);
               

            }

            // probably reduandant copy
            ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));

            if (camera_select->frame_save_state==State_Write_New_Frame) {
                // yolo code goes here
                rgba2bgr_convert(d_convert, debayer.d_debayer, camera_params->width, camera_params->height, 0);
                
                // copy frame back to cpu, and then save
                cudaMemcpy2D(frame_cpu.frame, camera_params->width*3, d_convert, camera_params->width*3, camera_params->width*3, camera_params->height, cudaMemcpyDeviceToHost);
                cv::Mat view = cv::Mat(camera_params->width * camera_params->height * 3, 1, CV_8U, frame_cpu.frame).reshape(3, camera_params->height);
                
                std::string image_name = "/home/user/Pictures/Orange/Cam" + camera_params->camera_serial + "_image" + std::to_string(camera_select->frame_save_idx) + ".tif";
                cv::imwrite(image_name, view);
                camera_select->frame_save_idx++;
                camera_select->frame_save_state = State_Frame_Idle;
            }          

        }
        usleep(16000); // sleep for 16ms 
    }
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (camera_select->yolo) {
        delete yolov8;
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