#if defined(__GNUC__)
#include <unistd.h>
#endif
#include <stdio.h>
#include <string.h>
#include "kernel.cuh"
#include "opengldisplay.h"
#include <cuda_runtime_api.h>

COpenGLDisplay::COpenGLDisplay(const char *name, CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer)
    : CThreadWorker(name), camera_params(camera_params), camera_select(camera_select), display_buffer(display_buffer)
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

    if (camera_select->yolo) {
        cudaMalloc((void **)&d_rgb, camera_params->width * camera_params->height * 3); 
        const std::string engine_file_path{"/home/user/detect/rat_pose.engine"};
        yolov8_pose = new YOLOv8_pose(engine_file_path);
        yolov8_pose->make_pipe(true);
    }

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

            // probably reduandant copy
            ck(cudaMemcpy2D(display_buffer, camera_params->width * 4, debayer.d_debayer, camera_params->width * 4, camera_params->width * 4, camera_params->height, cudaMemcpyDeviceToDevice));

            if (camera_select->yolo) {
                // yolo code goes here
                rgba2rgb_convert(d_rgb, debayer.d_debayer, camera_params->width, camera_params->height, 0);

            }


        }
        usleep(16000); // sleep for 16ms 
    }
    cudaFree(frame_original.d_orig);
    cudaFree(debayer.d_debayer);
    if (camera_select->yolo) {
        delete yolov8_pose;
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