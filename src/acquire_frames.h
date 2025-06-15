#ifndef ORANGE_ACQUIRE_FRAMES
#define ORANGE_ACQUIRE_FRAMES

#include "thread.h"
#include "camera.h"
#include <iostream>
#include <fstream>
#include "network_base.h"
#include "image_processing.h"
#include <cuda.h>

// Forward declare worker classes to break include cycles
class COpenGLDisplay;
class GPUVideoEncoder;
class YOLOv8Worker;
class ImageWriterWorker;

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
);
#endif