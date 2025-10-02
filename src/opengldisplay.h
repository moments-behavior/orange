#pragma once
#include "image_processing.h"
#include "threadworker.h"
#include "yolov8_det.h"
#include "obb_detector.h"
#include <nppi.h>
#define WORK_ENTRIES_MAX 2

class COpenGLDisplay : public CThreadWorker {
  public:
    COpenGLDisplay(
        const char *name, CameraParams *camera_params,
        CameraEachSelect *camera_select, unsigned char *display_buffer,
        INDIGOSignalBuilder *indigo_signal_builder); // name is the thread name
    ~COpenGLDisplay();

    bool PushToDisplay(void *imagePtr, size_t bufferSize, int width, int height,
                       int pixelFormat, unsigned long long timestamp,
                       unsigned long long frame_id);

    // open gl dimensions:
    CameraParams *camera_params;
    CameraEachSelect *camera_select;
    unsigned char *display_buffer;
    FrameGPU frame_original; // frame on gpu device
    Debayer debayer;
    INDIGOSignalBuilder *indigo_signal_builder;
    // for real time: refactor this
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
    
    // OBB Detection
    OBBDetector *obb_detector = nullptr;
    float *d_obb_points;  // GPU buffer for OBB corner points

  private:
    virtual void
    ThreadRunning(); // overides of COffThreadMachine for worker thread
  private:
    WORKER_ENTRY workerEntries[WORK_ENTRIES_MAX];
    WORKER_ENTRY *workerEntriesFreeQueue[WORK_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};
