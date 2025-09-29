#pragma once
#include "image_processing.h"
#include "threadworker.h"
#include "utils.h"
#include "yolov8_det.h"
#include <nppi.h>
#define WORK_ENTRIES_MAX 1

class COpenGLDisplay : public CThreadWorker {
  public:
    COpenGLDisplay(const char *name, CameraParams *camera_params,
                   CameraEachSelect *camera_select,
                   unsigned char *display_buffer);
    ~COpenGLDisplay();

    bool PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height,
                       int pixelFormat, unsigned long long timestamp,
                       unsigned long long frame_id);

    // open gl dimensions:
    cudaStream_t stream;
    CameraParams *camera_params;
    CameraEachSelect *camera_select;
    unsigned char *display_buffer;
    FrameGPU frame_original;
    Debayer debayer;
    unsigned char *d_convert;
    YOLOv8 *yolov8;
    FrameCPU frame_cpu;
    NppiSize input_image_size;
    NppiRect input_image_roi;
    NppiSize output_image_size;
    NppiRect output_image_roi;
    float *d_points;
    unsigned int *d_skeleton;
    unsigned int *d_resize;

  private:
    virtual void
    ThreadRunning(); // overides of COffThreadMachine for worker thread
  private:
    WORKER_ENTRY workerEntries[WORK_ENTRIES_MAX];
    WORKER_ENTRY *workerEntriesFreeQueue[WORK_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};
