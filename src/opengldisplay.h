#pragma once
#include "image_processing.h"
#include "threadworker.h"
#include "utils.h"
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
    NppiSize input_image_size;
    NppiRect input_image_roi;
    NppiSize output_image_size;
    NppiRect output_image_roi;
    unsigned int *d_resize;

    // Average-brightness sampling for the GUI plot during preview (mirrors the
    // encoder path). Reuses this thread's own `stream`; throttled + subsampled.
    unsigned long long *d_bright_sum = nullptr;
    unsigned long long *h_bright_sum = nullptr;
    double bright_next_sample_time = 0.0; // wall-clock throttle (steady seconds)
    int bright_stride = 8;
    long bright_sample_count = 1;

  private:
    virtual void
    ThreadRunning(); // overides of COffThreadMachine for worker thread
  private:
    WORKER_ENTRY workerEntries[WORK_ENTRIES_MAX];
    WORKER_ENTRY *workerEntriesFreeQueue[WORK_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};
