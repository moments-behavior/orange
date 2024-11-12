#pragma once

#include <opencv2/opencv.hpp>
#include <cuda_runtime.h>
#include "threadworker.h"
#include "image_processing.h"
#include "video_capture.h"
#include "yolov8_det.h"
#include "camera_params.h"

struct INDIGOSignalBuilder;
struct CameraEachSelect;

#define WORK_ENTRIES_MAX 2

typedef struct {
	void* imagePtr; // source image buffer
	size_t bufferSize; // size of imagePtr in bytes
	int width;
	int height;
	int pixelFormat;
    unsigned long long timestamp;
    unsigned long long frame_id;
    uint64_t timestamp_sys;
} WORKER_ENTRY;

class COpenGLDisplay : public CThreadWorker
{
public:
    COpenGLDisplay(const char* name, evt::CameraParams *camera_params, CameraEachSelect *camera_select, unsigned char *display_buffer, INDIGOSignalBuilder* indigo_signal_builder); // name is the thread name
    ~COpenGLDisplay ();

    bool PushToDisplay(void* imagePtr, size_t bufferSize, int width, int height, int pixelFormat, unsigned long long timestamp, unsigned long long frame_id);

    // OpenGL dimensions:
    evt::CameraParams* camera_params;
    CameraEachSelect* camera_select;
    unsigned char* display_buffer;
    evt::FrameGPU frame_original; // Frame on GPU device 
    evt::Debayer debayer;
    INDIGOSignalBuilder* indigo_signal_builder;
    // For real time: refactor this
    unsigned char *d_convert;
    YOLOv8* yolov8;
    evt::FrameCPU frame_cpu;

    float *d_points;
    unsigned int *d_skeleton;

private: 
    virtual void ThreadRunning(); // Overrides CThreadWorker for worker thread
private:    
    WORKER_ENTRY workerEntries[WORK_ENTRIES_MAX];
    WORKER_ENTRY* workerEntriesFreeQueue[WORK_ENTRIES_MAX];
    int workerEntriesFreeQueueCount;
};
